//! Main `GraphPalace` struct — the unified orchestrator.
//!
//! Coordinates storage, embeddings, pathfinding, stigmergy, and
//! knowledge graph into a single coherent API.

use chrono::{DateTime, Utc};

use gp_core::config::GraphPalaceConfig;
use gp_core::error::{GraphPalaceError, Result};
use gp_core::types::*;
use gp_embeddings::engine::EmbeddingEngine;
use gp_pathfinding::astar::{GraphAccess, SemanticAStar};
use gp_pathfinding::provenance::PathResult;
use gp_storage::backend::StorageBackend;
use gp_storage::memory::InMemoryBackend;

use crate::export::{ImportMode, ImportStats, PalaceExport};
use crate::lifecycle::{ColdSpot, HotPath, KgRelationship, PalaceStatus};
use crate::search::{PheromoneBooster, SearchResult};

// ---------------------------------------------------------------------------
// GraphAccess adapter for InMemoryBackend
// ---------------------------------------------------------------------------

/// Wraps [`InMemoryBackend`] to implement [`GraphAccess`] for A* pathfinding.
///
/// Builds a graph view where nodes are all palace entities (wings, rooms,
/// closets, drawers, KG entities) and edges come from the parent_map,
/// edge_pheromones, and edge_costs tables.
struct BackendGraphAccess<'a> {
    backend: &'a InMemoryBackend,
}

impl<'a> BackendGraphAccess<'a> {
    fn new(backend: &'a InMemoryBackend) -> Self {
        Self { backend }
    }

    /// Build a `GraphNode` from any palace entity.
    fn node_from_wing(w: &Wing, degree: usize) -> GraphNode {
        GraphNode {
            id: w.id.clone(),
            label: w.name.clone(),
            embedding: w.embedding,
            pheromones: w.pheromones.clone(),
            degree,
        }
    }

    fn node_from_room(r: &Room, degree: usize) -> GraphNode {
        GraphNode {
            id: r.id.clone(),
            label: r.name.clone(),
            embedding: r.embedding,
            pheromones: r.pheromones.clone(),
            degree,
        }
    }

    fn node_from_closet(c: &Closet, degree: usize) -> GraphNode {
        GraphNode {
            id: c.id.clone(),
            label: c.name.clone(),
            embedding: c.embedding,
            pheromones: c.pheromones.clone(),
            degree,
        }
    }

    fn node_from_drawer(d: &Drawer, degree: usize) -> GraphNode {
        GraphNode {
            id: d.id.clone(),
            label: d.content.chars().take(40).collect(),
            embedding: d.embedding,
            pheromones: d.pheromones.clone(),
            degree,
        }
    }

    fn node_from_entity(e: &Entity, degree: usize) -> GraphNode {
        GraphNode {
            id: e.id.clone(),
            label: e.name.clone(),
            embedding: e.embedding,
            pheromones: e.pheromones.clone(),
            degree,
        }
    }

    /// Build a `GraphEdge` from a parent→child relationship key.
    fn edge_for_key(
        data: &gp_storage::memory::PalaceData,
        from: &str,
        to: &str,
        relation_type: &str,
    ) -> GraphEdge {
        let key = format!("{from}:{to}");
        let pheromones = data
            .edge_pheromones
            .get(&key)
            .cloned()
            .unwrap_or_default();
        let cost = data
            .edge_costs
            .get(&key)
            .cloned()
            .unwrap_or_else(|| EdgeCost::new(structural_cost(relation_type)));
        GraphEdge {
            from: from.to_string(),
            to: to.to_string(),
            relation_type: relation_type.to_string(),
            cost,
            pheromones,
        }
    }

    /// Count outgoing edges for a given node ID.
    fn degree_of(data: &gp_storage::memory::PalaceData, id: &str) -> usize {
        // Count children (entries in parent_map where parent == id)
        let mut deg = data.parent_map.values().filter(|p| p.as_str() == id).count();
        // Count if this node itself has a parent (edge from parent to this)
        if data.parent_map.contains_key(id) {
            deg += 1;
        }
        deg
    }
}

impl<'a> GraphAccess for BackendGraphAccess<'a> {
    fn get_node(&self, id: &str) -> Option<GraphNode> {
        let d = self.backend.read_data();
        let degree = Self::degree_of(&d, id);

        if let Some(w) = d.wings.get(id) {
            return Some(Self::node_from_wing(w, degree));
        }
        if let Some(r) = d.rooms.get(id) {
            return Some(Self::node_from_room(r, degree));
        }
        if let Some(c) = d.closets.get(id) {
            return Some(Self::node_from_closet(c, degree));
        }
        if let Some(dr) = d.drawers.get(id) {
            return Some(Self::node_from_drawer(dr, degree));
        }
        if let Some(e) = d.entities.get(id) {
            return Some(Self::node_from_entity(e, degree));
        }
        None
    }

    fn get_neighbors(&self, id: &str) -> Vec<(GraphEdge, GraphNode)> {
        let d = self.backend.read_data();
        let mut neighbors = Vec::new();

        // Downward edges: find children where parent_map[child] == id
        for (child_id, parent_id) in &d.parent_map {
            if parent_id == id {
                let rel_type = if d.rooms.contains_key(child_id) {
                    "HAS_ROOM"
                } else if d.closets.contains_key(child_id) {
                    "HAS_CLOSET"
                } else if d.drawers.contains_key(child_id) {
                    "HAS_DRAWER"
                } else {
                    "CONTAINS"
                };
                let degree = Self::degree_of(&d, child_id);
                let edge = Self::edge_for_key(&d, id, child_id, rel_type);
                if let Some(w) = d.wings.get(child_id) {
                    neighbors.push((edge, Self::node_from_wing(w, degree)));
                } else if let Some(r) = d.rooms.get(child_id) {
                    neighbors.push((edge, Self::node_from_room(r, degree)));
                } else if let Some(c) = d.closets.get(child_id) {
                    neighbors.push((edge, Self::node_from_closet(c, degree)));
                } else if let Some(dr) = d.drawers.get(child_id) {
                    neighbors.push((edge, Self::node_from_drawer(dr, degree)));
                }
            }
        }

        // Upward edge: if this node has a parent, add a reverse edge
        if let Some(parent_id) = d.parent_map.get(id) {
            let rel_type = if d.wings.contains_key(parent_id) {
                "HAS_ROOM"
            } else if d.rooms.contains_key(parent_id) {
                "HAS_CLOSET"
            } else if d.closets.contains_key(parent_id) {
                "HAS_DRAWER"
            } else {
                "CONTAINS"
            };
            let degree = Self::degree_of(&d, parent_id);
            // Reverse edge costs are the same
            let edge = Self::edge_for_key(&d, id, parent_id, rel_type);
            if let Some(w) = d.wings.get(parent_id.as_str()) {
                neighbors.push((edge, Self::node_from_wing(w, degree)));
            } else if let Some(r) = d.rooms.get(parent_id.as_str()) {
                neighbors.push((edge, Self::node_from_room(r, degree)));
            } else if let Some(c) = d.closets.get(parent_id.as_str()) {
                neighbors.push((edge, Self::node_from_closet(c, degree)));
            }
        }

        neighbors
    }
}

// ---------------------------------------------------------------------------
// GraphPalace — the unified orchestrator
// ---------------------------------------------------------------------------

/// Unified GraphPalace: memory palace with semantic search, A* navigation,
/// pheromone-guided reinforcement, and knowledge graph.
pub struct GraphPalace {
    /// The storage backend.
    storage: InMemoryBackend,
    /// Embedding engine for encoding text.
    embeddings: Box<dyn EmbeddingEngine>,
    /// Palace configuration.
    config: GraphPalaceConfig,
    /// Pheromone booster for search.
    booster: PheromoneBooster,
    /// Last time pheromone decay was applied.
    last_decay_time: Option<DateTime<Utc>>,
}

impl GraphPalace {
    /// Create a new palace with the given configuration, storage, and embeddings.
    ///
    /// Initialises the storage schema.
    pub fn new(
        config: GraphPalaceConfig,
        storage: InMemoryBackend,
        embeddings: Box<dyn EmbeddingEngine>,
    ) -> Result<Self> {
        storage.init_schema()?;
        Ok(Self {
            storage,
            embeddings,
            config,
            booster: PheromoneBooster::default(),
            last_decay_time: None,
        })
    }

    /// Access the underlying storage backend.
    pub fn storage(&self) -> &InMemoryBackend {
        &self.storage
    }

    /// Access the configuration.
    pub fn config(&self) -> &GraphPalaceConfig {
        &self.config
    }

    // -- Structure management -----------------------------------------------

    /// Create a new wing in the palace.
    pub fn add_wing(
        &mut self,
        name: &str,
        wing_type: WingType,
        description: &str,
    ) -> Result<String> {
        let embedding = self
            .embeddings
            .encode(name)
            .map_err(|e| GraphPalaceError::Embedding(e.to_string()))?;
        self.storage
            .create_wing(name, wing_type, description, embedding)
    }

    /// Create a new room in a wing.
    pub fn add_room(
        &mut self,
        wing_id: &str,
        name: &str,
        hall_type: HallType,
    ) -> Result<String> {
        let embedding = self
            .embeddings
            .encode(name)
            .map_err(|e| GraphPalaceError::Embedding(e.to_string()))?;
        self.storage
            .create_room(wing_id, name, hall_type, name, embedding)
    }

    /// Store a new memory (drawer) in the palace.
    ///
    /// If the specified wing/room don't exist, finds or creates them.
    /// Automatically creates a "General" closet in the room if none exists.
    pub fn add_drawer(
        &mut self,
        content: &str,
        wing_name: &str,
        room_name: &str,
        source: DrawerSource,
    ) -> Result<String> {
        let embedding = self
            .embeddings
            .encode(content)
            .map_err(|e| GraphPalaceError::Embedding(e.to_string()))?;

        // Find or create wing
        let wing_id = match self.storage.find_wing_by_name(wing_name) {
            Some(w) => w.id,
            None => {
                let wing_emb = self
                    .embeddings
                    .encode(wing_name)
                    .map_err(|e| GraphPalaceError::Embedding(e.to_string()))?;
                self.storage
                    .create_wing(wing_name, WingType::Topic, wing_name, wing_emb)?
            }
        };

        // Find or create room
        let rooms = self.storage.list_rooms(&wing_id);
        let room_id = match rooms.iter().find(|r| r.name == room_name) {
            Some(r) => r.id.clone(),
            None => {
                let room_emb = self
                    .embeddings
                    .encode(room_name)
                    .map_err(|e| GraphPalaceError::Embedding(e.to_string()))?;
                self.storage
                    .create_room(&wing_id, room_name, HallType::Facts, room_name, room_emb)?
            }
        };

        // Find or create "General" closet in room
        let closet_id = self.find_or_create_general_closet(&room_id)?;

        // Create drawer
        self.storage
            .create_drawer(&closet_id, content, embedding, source, None, 0.5)
    }

    /// Find the "General" closet in a room, or create one.
    fn find_or_create_general_closet(&mut self, room_id: &str) -> Result<String> {
        let d = self.storage.read_data();
        // Find existing closets in this room
        for (child_id, parent_id) in &d.parent_map {
            if parent_id == room_id
                && let Some(closet) = d.closets.get(child_id)
                && closet.name == "General"
            {
                return Ok(closet.id.clone());
            }
        }
        // Check if any closet exists at all
        let any_closet = d
            .parent_map
            .iter()
            .find(|(child_id, parent_id)| parent_id.as_str() == room_id && d.closets.contains_key(child_id.as_str()))
            .map(|(child_id, _)| child_id.clone());
        drop(d);

        if let Some(cid) = any_closet {
            return Ok(cid);
        }

        // Create a "General" closet
        let closet_emb = self
            .embeddings
            .encode("General")
            .map_err(|e| GraphPalaceError::Embedding(e.to_string()))?;
        self.storage
            .create_closet(room_id, "General", "Default closet", closet_emb)
    }

    // -- Search -------------------------------------------------------------

    /// Semantic search across all drawers, boosted by pheromones.
    /// Semantic search (immutable — requires a pre-computed embedding).
    ///
    /// If you have the raw query text, use [`search_mut`] instead which
    /// encodes the text for you.
    pub fn search(&self, _query: &str, _k: usize) -> Result<Vec<SearchResult>> {
        Err(GraphPalaceError::Embedding(
            "Use search_mut() for mutable borrow, or search_by_embedding()".into(),
        ))
    }

    /// Semantic search (requires &mut self for embedding encoding).
    pub fn search_mut(&mut self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self
            .embeddings
            .encode(query)
            .map_err(|e| GraphPalaceError::Embedding(e.to_string()))?;
        self.search_by_embedding(&query_embedding, k)
    }

    /// Search by a pre-computed embedding vector.
    pub fn search_by_embedding(
        &self,
        query_embedding: &Embedding,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let raw_results = self.storage.search_drawers(query_embedding, k * 2, 0.0);
        let d = self.storage.read_data();

        let mut search_results: Vec<SearchResult> = raw_results
            .into_iter()
            .map(|(drawer, raw_score)| {
                let boosted = self
                    .booster
                    .boost(raw_score, drawer.pheromones.exploitation);

                // Resolve wing/room names from parent chain
                let (wing_name, room_name) = resolve_drawer_location(&d, &drawer.id);

                SearchResult {
                    drawer_id: drawer.id,
                    content: drawer.content,
                    score: boosted,
                    wing_name,
                    room_name,
                }
            })
            .collect();

        // Re-sort by boosted score
        search_results.sort_by(|a, b| b.score.total_cmp(&a.score));
        search_results.truncate(k);
        Ok(search_results)
    }

    // -- Navigation ---------------------------------------------------------

    /// A* pathfinding between two nodes in the palace.
    ///
    /// Uses semantic cost, pheromone cost, and structural cost to find the
    /// optimal path. The `_context` parameter is reserved for future
    /// context-adaptive weight selection.
    pub fn navigate(
        &self,
        from_id: &str,
        to_id: &str,
        _context: Option<&str>,
    ) -> Result<PathResult> {
        let graph = BackendGraphAccess::new(&self.storage);
        let astar = SemanticAStar::new(
            self.config.astar.clone(),
            self.config.cost_weights,
        );
        astar.find_path(&graph, from_id, to_id).ok_or_else(|| {
            GraphPalaceError::Pathfinding(format!(
                "No path found from {from_id} to {to_id}"
            ))
        })
    }

    // -- Pheromone operations -----------------------------------------------

    /// Deposit pheromones along a successful path.
    ///
    /// Deposits success, traversal, and recency on edges, and exploitation
    /// on nodes.
    pub fn deposit_pheromones(&self, path: &[String], reward: f64) -> Result<()> {
        if path.len() < 2 {
            return Ok(());
        }
        for window in path.windows(2) {
            self.storage.deposit_edge_pheromones(
                &window[0],
                &window[1],
                reward,
                gp_stigmergy::rewards::TRAVERSAL_INCREMENT,
                gp_stigmergy::rewards::RECENCY_VALUE,
            );
        }
        for node_id in path {
            self.storage.deposit_node_pheromones(
                node_id,
                gp_stigmergy::rewards::EXPLOITATION_INCREMENT,
                0.0,
            );
        }
        Ok(())
    }

    /// Apply one cycle of pheromone decay across the entire palace.
    pub fn decay_pheromones(&mut self) -> Result<()> {
        self.storage
            .decay_all_pheromones(&self.config.pheromones);
        self.last_decay_time = Some(Utc::now());
        Ok(())
    }

    /// Get the k hottest paths (highest success pheromone).
    pub fn hot_paths(&self, k: usize) -> Result<Vec<HotPath>> {
        let raw = self.storage.hot_paths(k);
        Ok(raw
            .into_iter()
            .map(|(from_id, to_id, success)| HotPath {
                from_id,
                to_id,
                success_pheromone: success,
            })
            .collect())
    }

    /// Get the k coldest spots (lowest total pheromone).
    pub fn cold_spots(&self, k: usize) -> Result<Vec<ColdSpot>> {
        let raw = self.storage.cold_spots(k);
        Ok(raw
            .into_iter()
            .map(|(node_id, name, total)| ColdSpot {
                node_id,
                name,
                total_pheromone: total,
            })
            .collect())
    }

    // -- Knowledge graph ----------------------------------------------------

    /// Add a relationship triple to the knowledge graph.
    ///
    /// Creates entities if they don't exist.
    pub fn kg_add(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<String> {
        // Find or create subject entity
        let subj_id = self.find_or_create_entity(subject)?;
        // Find or create object entity
        let obj_id = self.find_or_create_entity(object)?;
        // Add relationship
        self.storage
            .add_relationship(&subj_id, predicate, &obj_id, 0.5)
    }

    /// Query knowledge graph relationships involving an entity.
    pub fn kg_query(&self, entity: &str) -> Result<Vec<KgRelationship>> {
        // Try to find entity by name
        let entity_id = match self.storage.find_entity_by_name(entity) {
            Some(e) => e.id,
            None => entity.to_string(), // use as ID directly
        };
        let rels = self.storage.query_relationships(&entity_id);
        let d = self.storage.read_data();
        Ok(rels
            .into_iter()
            .map(|r| {
                let subj_name = d
                    .entities
                    .get(&r.subject)
                    .map(|e| e.name.clone())
                    .unwrap_or_else(|| r.subject.clone());
                let obj_name = d
                    .entities
                    .get(&r.object)
                    .map(|e| e.name.clone())
                    .unwrap_or_else(|| r.object.clone());
                KgRelationship {
                    subject: subj_name,
                    predicate: r.predicate,
                    object: obj_name,
                    confidence: r.confidence,
                }
            })
            .collect())
    }

    /// Find an entity by name, or create one.
    fn find_or_create_entity(&mut self, name: &str) -> Result<String> {
        if let Some(e) = self.storage.find_entity_by_name(name) {
            return Ok(e.id);
        }
        let emb = self
            .embeddings
            .encode(name)
            .map_err(|e| GraphPalaceError::Embedding(e.to_string()))?;
        self.storage
            .create_entity(name, EntityType::Concept, name, emb)
    }

    // -- Status -------------------------------------------------------------

    /// Get the current palace status.
    pub fn status(&self) -> Result<PalaceStatus> {
        Ok(PalaceStatus {
            name: self.config.palace.name.clone(),
            wing_count: self.storage.wing_count(),
            room_count: self.storage.room_count(),
            closet_count: self.storage.closet_count(),
            drawer_count: self.storage.drawer_count(),
            entity_count: self.storage.entity_count(),
            relationship_count: self.storage.relationship_count(),
            total_pheromone_mass: self.storage.total_pheromone_mass(),
            last_decay_time: self.last_decay_time,
        })
    }

    // -- Export/Import -------------------------------------------------------

    /// Export the entire palace to a serializable snapshot.
    pub fn export(&self) -> Result<PalaceExport> {
        Ok(PalaceExport {
            version: 1,
            exported_at: Utc::now(),
            data: self.storage.snapshot(),
        })
    }

    /// Import palace data according to the specified mode.
    pub fn import(&self, export: PalaceExport, mode: ImportMode) -> Result<ImportStats> {
        let mut stats = ImportStats::default();
        let incoming = export.data;

        match mode {
            ImportMode::Replace => {
                stats.wings_added = incoming.wings.len();
                stats.rooms_added = incoming.rooms.len();
                stats.closets_added = incoming.closets.len();
                stats.drawers_added = incoming.drawers.len();
                stats.entities_added = incoming.entities.len();
                stats.relationships_added = incoming.relationships.len();
                self.storage.restore(incoming);
            }
            ImportMode::Merge => {
                use std::collections::hash_map::Entry;
                let mut d = self.storage.write_data();
                for (id, wing) in incoming.wings {
                    if let Entry::Vacant(e) = d.wings.entry(id) {
                        e.insert(wing);
                        stats.wings_added += 1;
                    } else {
                        stats.duplicates_skipped += 1;
                    }
                }
                for (id, room) in incoming.rooms {
                    if let Entry::Vacant(e) = d.rooms.entry(id) {
                        e.insert(room);
                        stats.rooms_added += 1;
                    } else {
                        stats.duplicates_skipped += 1;
                    }
                }
                for (id, closet) in incoming.closets {
                    if let Entry::Vacant(e) = d.closets.entry(id) {
                        e.insert(closet);
                        stats.closets_added += 1;
                    } else {
                        stats.duplicates_skipped += 1;
                    }
                }
                for (id, drawer) in incoming.drawers {
                    if let Entry::Vacant(e) = d.drawers.entry(id) {
                        e.insert(drawer);
                        stats.drawers_added += 1;
                    } else {
                        stats.duplicates_skipped += 1;
                    }
                }
                for (id, entity) in incoming.entities {
                    if let Entry::Vacant(e) = d.entities.entry(id) {
                        e.insert(entity);
                        stats.entities_added += 1;
                    } else {
                        stats.duplicates_skipped += 1;
                    }
                }
                for rel in incoming.relationships {
                    d.relationships.push(rel);
                    stats.relationships_added += 1;
                }
                // Merge parent_map
                for (k, v) in incoming.parent_map {
                    d.parent_map.entry(k).or_insert(v);
                }
                // Merge edge pheromones & costs
                for (k, v) in incoming.edge_pheromones {
                    d.edge_pheromones.entry(k).or_insert(v);
                }
                for (k, v) in incoming.edge_costs {
                    d.edge_costs.entry(k).or_insert(v);
                }
                // Advance next_id if needed
                if incoming.next_id > d.next_id {
                    d.next_id = incoming.next_id;
                }
            }
            ImportMode::Overlay => {
                let mut d = self.storage.write_data();
                for (id, wing) in incoming.wings {
                    d.wings.insert(id, wing);
                    stats.wings_added += 1;
                }
                for (id, room) in incoming.rooms {
                    d.rooms.insert(id, room);
                    stats.rooms_added += 1;
                }
                for (id, closet) in incoming.closets {
                    d.closets.insert(id, closet);
                    stats.closets_added += 1;
                }
                for (id, drawer) in incoming.drawers {
                    d.drawers.insert(id, drawer);
                    stats.drawers_added += 1;
                }
                for (id, entity) in incoming.entities {
                    d.entities.insert(id, entity);
                    stats.entities_added += 1;
                }
                for rel in incoming.relationships {
                    d.relationships.push(rel);
                    stats.relationships_added += 1;
                }
                for (k, v) in incoming.parent_map {
                    d.parent_map.insert(k, v);
                }
                for (k, v) in incoming.edge_pheromones {
                    d.edge_pheromones.insert(k, v);
                }
                for (k, v) in incoming.edge_costs {
                    d.edge_costs.insert(k, v);
                }
                if incoming.next_id > d.next_id {
                    d.next_id = incoming.next_id;
                }
            }
        }

        Ok(stats)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk up the parent_map to resolve drawer → closet → room → wing names.
fn resolve_drawer_location(
    data: &gp_storage::memory::PalaceData,
    drawer_id: &str,
) -> (String, String) {
    let mut room_name = String::new();
    let mut wing_name = String::new();

    // drawer → closet → room → wing
    if let Some(closet_id) = data.parent_map.get(drawer_id)
        && let Some(room_id) = data.parent_map.get(closet_id.as_str())
    {
        if let Some(room) = data.rooms.get(room_id.as_str()) {
            room_name = room.name.clone();
        }
        if let Some(wing_id) = data.parent_map.get(room_id.as_str())
            && let Some(wing) = data.wings.get(wing_id.as_str())
        {
            wing_name = wing.name.clone();
        }
    }

    (wing_name, room_name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use gp_embeddings::MockEmbeddingEngine;

    fn make_palace() -> GraphPalace {
        let config = GraphPalaceConfig::default();
        let storage = InMemoryBackend::new();
        let embeddings = Box::new(MockEmbeddingEngine::new());
        GraphPalace::new(config, storage, embeddings).unwrap()
    }

    // ── Construction ──────────────────────────────────────────────────────

    #[test]
    fn new_palace_initialises_schema() {
        let palace = make_palace();
        assert!(palace.storage().read_data().schema_initialized);
    }

    #[test]
    fn new_palace_is_empty() {
        let palace = make_palace();
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 0);
        assert_eq!(status.room_count, 0);
        assert_eq!(status.drawer_count, 0);
        assert_eq!(status.entity_count, 0);
        assert!(status.last_decay_time.is_none());
    }

    #[test]
    fn palace_config_accessible() {
        let palace = make_palace();
        assert_eq!(palace.config().palace.name, "My Palace");
    }

    // ── Wing/Room management ──────────────────────────────────────────────

    #[test]
    fn add_wing() {
        let mut palace = make_palace();
        let id = palace
            .add_wing("Science", WingType::Domain, "All science")
            .unwrap();
        assert!(id.starts_with("wing_"));
        assert_eq!(palace.status().unwrap().wing_count, 1);
    }

    #[test]
    fn add_wing_duplicate_fails() {
        let mut palace = make_palace();
        palace
            .add_wing("Science", WingType::Domain, "desc")
            .unwrap();
        let result = palace.add_wing("Science", WingType::Domain, "desc2");
        assert!(result.is_err());
    }

    #[test]
    fn add_room() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let rid = palace.add_room(&wid, "Physics", HallType::Facts).unwrap();
        assert!(rid.starts_with("room_"));
        assert_eq!(palace.status().unwrap().room_count, 1);
    }

    #[test]
    fn add_room_invalid_wing_fails() {
        let mut palace = make_palace();
        let result = palace.add_room("fake_wing", "R", HallType::Facts);
        assert!(result.is_err());
    }

    // ── Drawer management ─────────────────────────────────────────────────

    #[test]
    fn add_drawer_creates_structure() {
        let mut palace = make_palace();
        let id = palace
            .add_drawer("F = ma", "Science", "Physics", DrawerSource::Conversation)
            .unwrap();
        assert!(id.starts_with("drawer_"));
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 1);
        assert_eq!(status.room_count, 1);
        assert_eq!(status.closet_count, 1);
        assert_eq!(status.drawer_count, 1);
    }

    #[test]
    fn add_drawer_reuses_existing_wing() {
        let mut palace = make_palace();
        palace
            .add_drawer("F = ma", "Science", "Physics", DrawerSource::Conversation)
            .unwrap();
        palace
            .add_drawer("E = mc^2", "Science", "Physics", DrawerSource::Conversation)
            .unwrap();
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 1); // reused
        assert_eq!(status.room_count, 1); // reused
        assert_eq!(status.drawer_count, 2);
    }

    #[test]
    fn add_drawer_new_room_same_wing() {
        let mut palace = make_palace();
        palace
            .add_drawer("F = ma", "Science", "Physics", DrawerSource::Conversation)
            .unwrap();
        palace
            .add_drawer("2+2=4", "Science", "Math", DrawerSource::Conversation)
            .unwrap();
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 1);
        assert_eq!(status.room_count, 2);
        assert_eq!(status.drawer_count, 2);
    }

    #[test]
    fn add_drawer_multiple_wings() {
        let mut palace = make_palace();
        palace
            .add_drawer("F = ma", "Science", "Physics", DrawerSource::Conversation)
            .unwrap();
        palace
            .add_drawer("Monet", "Art", "Painting", DrawerSource::Agent)
            .unwrap();
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 2);
    }

    // ── Search ────────────────────────────────────────────────────────────

    #[test]
    fn search_returns_results() {
        let mut palace = make_palace();
        palace
            .add_drawer(
                "The sky is blue because of Rayleigh scattering",
                "Science",
                "Physics",
                DrawerSource::Conversation,
            )
            .unwrap();
        palace
            .add_drawer(
                "Water boils at 100 degrees Celsius",
                "Science",
                "Chemistry",
                DrawerSource::Conversation,
            )
            .unwrap();

        let results = palace.search_mut("blue sky light", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].wing_name, "Science");
    }

    #[test]
    fn search_respects_k() {
        let mut palace = make_palace();
        for i in 0..10 {
            palace
                .add_drawer(
                    &format!("Memory number {i} about various topics"),
                    "Memories",
                    "General",
                    DrawerSource::Conversation,
                )
                .unwrap();
        }
        let results = palace.search_mut("memory topic", 3).unwrap();
        assert!(results.len() <= 3);
    }

    #[test]
    fn search_empty_palace() {
        let mut palace = make_palace();
        let results = palace.search_mut("anything", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_by_embedding() {
        let mut palace = make_palace();
        palace
            .add_drawer("Test content", "Wing", "Room", DrawerSource::Conversation)
            .unwrap();

        let mut engine = MockEmbeddingEngine::new();
        let emb = engine.encode("Test content").unwrap();
        let results = palace.search_by_embedding(&emb, 5).unwrap();
        assert!(!results.is_empty());
        // Exact match should score very high
        assert!(results[0].score > 0.9);
    }

    // ── Navigation ────────────────────────────────────────────────────────

    #[test]
    fn navigate_parent_to_child() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let rid = palace.add_room(&wid, "Physics", HallType::Facts).unwrap();

        let path = palace.navigate(&wid, &rid, None).unwrap();
        assert_eq!(path.path.len(), 2);
        assert_eq!(path.path[0], wid);
        assert_eq!(path.path[1], rid);
    }

    #[test]
    fn navigate_child_to_parent() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let rid = palace.add_room(&wid, "Physics", HallType::Facts).unwrap();

        // Reverse direction — should work because we add upward edges
        let path = palace.navigate(&rid, &wid, None).unwrap();
        assert_eq!(path.path.len(), 2);
        assert_eq!(path.path[0], rid);
        assert_eq!(path.path[1], wid);
    }

    #[test]
    fn navigate_same_node() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let path = palace.navigate(&wid, &wid, None).unwrap();
        assert_eq!(path.path, vec![wid]);
    }

    #[test]
    fn navigate_no_path_fails() {
        let mut palace = make_palace();
        let w1 = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let w2 = palace
            .add_wing("Art", WingType::Domain, "")
            .unwrap();
        // Wings are not connected to each other
        let result = palace.navigate(&w1, &w2, None);
        assert!(result.is_err());
    }

    #[test]
    fn navigate_through_hierarchy() {
        let mut palace = make_palace();
        palace
            .add_drawer("F = ma", "Science", "Physics", DrawerSource::Conversation)
            .unwrap();

        // Navigate from wing to drawer (3 hops: wing → room → closet → drawer)
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 1);

        let wings = palace.storage().list_wings();
        let wing_id = &wings[0].id;
        let d = palace.storage().read_data();
        // Find the drawer
        let drawer_id = d.drawers.keys().next().unwrap().clone();
        drop(d);

        let path = palace.navigate(wing_id, &drawer_id, None).unwrap();
        assert!(path.path.len() >= 2);
        assert_eq!(path.path[0], *wing_id);
        assert_eq!(*path.path.last().unwrap(), drawer_id);
    }

    // ── Pheromone operations ──────────────────────────────────────────────

    #[test]
    fn deposit_pheromones_along_path() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let rid = palace.add_room(&wid, "Physics", HallType::Facts).unwrap();

        palace
            .deposit_pheromones(&[wid.clone(), rid.clone()], 1.0)
            .unwrap();

        let wing = palace.storage().get_wing(&wid).unwrap();
        assert!(wing.pheromones.exploitation > 0.0);

        let hot = palace.hot_paths(10).unwrap();
        assert!(!hot.is_empty());
    }

    #[test]
    fn deposit_pheromones_single_node_is_noop() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        palace.deposit_pheromones(&[wid], 1.0).unwrap();
        // No edges deposited for single-node path
    }

    #[test]
    fn deposit_pheromones_empty_path() {
        let palace = make_palace();
        palace.deposit_pheromones(&[], 1.0).unwrap();
    }

    #[test]
    fn decay_pheromones() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        palace
            .storage()
            .deposit_node_pheromones(&wid, 1.0, 0.5);

        assert!(palace.status().unwrap().last_decay_time.is_none());
        palace.decay_pheromones().unwrap();

        let wing = palace.storage().get_wing(&wid).unwrap();
        assert!(wing.pheromones.exploitation < 1.0);
        assert!(palace.status().unwrap().last_decay_time.is_some());
    }

    #[test]
    fn hot_paths_empty() {
        let palace = make_palace();
        let hot = palace.hot_paths(10).unwrap();
        assert!(hot.is_empty());
    }

    #[test]
    fn cold_spots_empty() {
        let palace = make_palace();
        let cold = palace.cold_spots(10).unwrap();
        assert!(cold.is_empty());
    }

    #[test]
    fn cold_spots_returns_unvisited() {
        let mut palace = make_palace();
        palace
            .add_wing("Hot", WingType::Domain, "")
            .unwrap();
        let cold_id = palace
            .add_wing("Cold", WingType::Domain, "")
            .unwrap();

        let hot_id = palace
            .storage()
            .list_wings()
            .into_iter()
            .find(|w| w.name == "Hot")
            .unwrap()
            .id;
        palace.storage().deposit_node_pheromones(&hot_id, 10.0, 5.0);

        let cold = palace.cold_spots(1).unwrap();
        assert_eq!(cold.len(), 1);
        assert_eq!(cold[0].node_id, cold_id);
    }

    // ── Knowledge graph ───────────────────────────────────────────────────

    #[test]
    fn kg_add_creates_entities() {
        let mut palace = make_palace();
        let rel_id = palace.kg_add("Einstein", "discovered", "Relativity").unwrap();
        assert!(rel_id.starts_with("rel_"));
        assert_eq!(palace.status().unwrap().entity_count, 2);
    }

    #[test]
    fn kg_add_reuses_existing_entities() {
        let mut palace = make_palace();
        palace.kg_add("Einstein", "discovered", "Relativity").unwrap();
        palace.kg_add("Einstein", "explained", "Photoelectric Effect").unwrap();
        assert_eq!(palace.status().unwrap().entity_count, 3); // Einstein + 2 concepts
    }

    #[test]
    fn kg_query_by_name() {
        let mut palace = make_palace();
        palace.kg_add("Einstein", "discovered", "Relativity").unwrap();
        palace.kg_add("Einstein", "explained", "Photoelectric").unwrap();

        let rels = palace.kg_query("Einstein").unwrap();
        assert_eq!(rels.len(), 2);
        assert!(rels.iter().any(|r| r.predicate == "discovered"));
        assert!(rels.iter().any(|r| r.predicate == "explained"));
    }

    #[test]
    fn kg_query_reverse() {
        let mut palace = make_palace();
        palace.kg_add("Einstein", "discovered", "Relativity").unwrap();

        let rels = palace.kg_query("Relativity").unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].subject, "Einstein");
    }

    #[test]
    fn kg_query_nonexistent() {
        let palace = make_palace();
        let rels = palace.kg_query("Nobody").unwrap();
        assert!(rels.is_empty());
    }

    // ── Export/Import ─────────────────────────────────────────────────────

    #[test]
    fn export_empty_palace() {
        let palace = make_palace();
        let export = palace.export().unwrap();
        assert_eq!(export.version, 1);
        assert!(export.data.wings.is_empty());
    }

    #[test]
    fn export_with_data() {
        let mut palace = make_palace();
        palace
            .add_drawer("F = ma", "Science", "Physics", DrawerSource::Conversation)
            .unwrap();
        palace.kg_add("Newton", "formulated", "F=ma").unwrap();

        let export = palace.export().unwrap();
        assert_eq!(export.data.wings.len(), 1);
        assert_eq!(export.data.drawers.len(), 1);
        assert_eq!(export.data.entities.len(), 2);
    }

    #[test]
    fn import_replace() {
        let mut palace = make_palace();
        palace
            .add_drawer("Old data", "OldWing", "OldRoom", DrawerSource::Conversation)
            .unwrap();

        // Create export from another palace
        let mut palace2 = make_palace();
        palace2
            .add_drawer("New data", "NewWing", "NewRoom", DrawerSource::Agent)
            .unwrap();
        let export = palace2.export().unwrap();

        let stats = palace.import(export, ImportMode::Replace).unwrap();
        assert_eq!(stats.wings_added, 1);
        assert_eq!(stats.drawers_added, 1);

        // Original data should be gone
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 1);
        assert_eq!(status.drawer_count, 1);
    }

    #[test]
    fn import_merge_skips_duplicates() {
        let mut palace = make_palace();
        palace
            .add_drawer("Data", "Wing", "Room", DrawerSource::Conversation)
            .unwrap();

        let export = palace.export().unwrap();

        // Import the same data again — should skip duplicates
        let stats = palace.import(export, ImportMode::Merge).unwrap();
        assert!(stats.duplicates_skipped > 0);
        assert_eq!(palace.status().unwrap().drawer_count, 1);
    }

    #[test]
    fn import_overlay_overwrites() {
        let mut palace = make_palace();
        palace
            .add_drawer("Original", "Wing", "Room", DrawerSource::Conversation)
            .unwrap();

        let mut export = palace.export().unwrap();
        // Modify the exported data
        for drawer in export.data.drawers.values_mut() {
            drawer.content = "Modified".into();
        }

        let stats = palace.import(export, ImportMode::Overlay).unwrap();
        assert_eq!(stats.drawers_added, 1);

        // Content should be modified
        let d = palace.storage().read_data();
        let drawer = d.drawers.values().next().unwrap();
        assert_eq!(drawer.content, "Modified");
    }

    #[test]
    fn export_import_round_trip_json() {
        let mut palace = make_palace();
        palace
            .add_drawer(
                "The sky is blue",
                "Science",
                "Physics",
                DrawerSource::Conversation,
            )
            .unwrap();
        palace.kg_add("Sky", "appears", "Blue").unwrap();

        let export = palace.export().unwrap();
        let json = export.to_json().unwrap();

        let import = PalaceExport::from_json(&json).unwrap();
        let palace2 = make_palace();
        let stats = palace2.import(import, ImportMode::Replace).unwrap();

        assert_eq!(stats.wings_added, 1);
        assert_eq!(stats.drawers_added, 1);
        assert_eq!(stats.entities_added, 2);
        assert_eq!(palace2.status().unwrap().drawer_count, 1);
    }

    // ── Status ────────────────────────────────────────────────────────────

    #[test]
    fn status_reflects_all_operations() {
        let mut palace = make_palace();
        palace
            .add_drawer("Data", "W", "R", DrawerSource::Conversation)
            .unwrap();
        palace.kg_add("A", "knows", "B").unwrap();
        palace
            .storage()
            .deposit_node_pheromones("wing_1", 1.0, 0.5);

        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 1);
        assert_eq!(status.room_count, 1);
        assert_eq!(status.closet_count, 1);
        assert_eq!(status.drawer_count, 1);
        assert_eq!(status.entity_count, 2);
        assert_eq!(status.relationship_count, 1);
        assert!(status.total_pheromone_mass > 0.0);
    }

    // ── GraphAccess adapter ───────────────────────────────────────────────

    #[test]
    fn graph_access_get_node_wing() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();

        let graph = BackendGraphAccess::new(palace.storage());
        let node = graph.get_node(&wid).unwrap();
        assert_eq!(node.id, wid);
        assert_eq!(node.label, "Science");
    }

    #[test]
    fn graph_access_get_node_room() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let rid = palace.add_room(&wid, "Physics", HallType::Facts).unwrap();

        let graph = BackendGraphAccess::new(palace.storage());
        let node = graph.get_node(&rid).unwrap();
        assert_eq!(node.label, "Physics");
    }

    #[test]
    fn graph_access_get_node_nonexistent() {
        let palace = make_palace();
        let graph = BackendGraphAccess::new(palace.storage());
        assert!(graph.get_node("fake").is_none());
    }

    #[test]
    fn graph_access_neighbors_downward() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let rid = palace.add_room(&wid, "Physics", HallType::Facts).unwrap();

        let graph = BackendGraphAccess::new(palace.storage());
        let neighbors = graph.get_neighbors(&wid);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].1.id, rid);
        assert_eq!(neighbors[0].0.relation_type, "HAS_ROOM");
    }

    #[test]
    fn graph_access_neighbors_upward() {
        let mut palace = make_palace();
        let wid = palace
            .add_wing("Science", WingType::Domain, "")
            .unwrap();
        let rid = palace.add_room(&wid, "Physics", HallType::Facts).unwrap();

        let graph = BackendGraphAccess::new(palace.storage());
        let neighbors = graph.get_neighbors(&rid);
        // Should include upward edge to wing
        assert!(neighbors.iter().any(|(_, n)| n.id == wid));
    }

    #[test]
    fn graph_access_neighbors_empty() {
        let palace = make_palace();
        let graph = BackendGraphAccess::new(palace.storage());
        let neighbors = graph.get_neighbors("nonexistent");
        assert!(neighbors.is_empty());
    }

    // ── Full lifecycle ────────────────────────────────────────────────────

    #[test]
    fn full_lifecycle() {
        let mut palace = make_palace();

        // 1. Add structure
        palace
            .add_drawer(
                "Newton's first law: inertia",
                "Science",
                "Physics",
                DrawerSource::Conversation,
            )
            .unwrap();
        palace
            .add_drawer(
                "Newton's second law: F = ma",
                "Science",
                "Physics",
                DrawerSource::Conversation,
            )
            .unwrap();
        palace
            .add_drawer(
                "Van Gogh's Starry Night",
                "Art",
                "Painting",
                DrawerSource::Agent,
            )
            .unwrap();

        // 2. Search
        let results = palace.search_mut("force and motion", 5).unwrap();
        assert!(!results.is_empty());

        // 3. Navigate
        let wings = palace.storage().list_wings();
        if wings.len() >= 2 {
            let w1 = &wings[0].id;
            let rooms1 = palace.storage().list_rooms(w1);
            if !rooms1.is_empty() {
                let r1 = &rooms1[0].id;
                let path = palace.navigate(w1, r1, None).unwrap();
                assert!(path.path.len() >= 2);

                // 4. Deposit pheromones
                palace.deposit_pheromones(&path.path, 1.0).unwrap();
            }
        }

        // 5. Decay
        palace.decay_pheromones().unwrap();

        // 6. Knowledge graph
        palace.kg_add("Newton", "formulated", "Laws of Motion").unwrap();

        // 7. Export
        let export = palace.export().unwrap();
        let json = export.to_json().unwrap();

        // 8. Import into fresh palace
        let import = PalaceExport::from_json(&json).unwrap();
        let palace2 = make_palace();
        palace2.import(import, ImportMode::Replace).unwrap();

        let s = palace2.status().unwrap();
        assert_eq!(s.wing_count, 2);
        assert_eq!(s.drawer_count, 3);
        assert_eq!(s.entity_count, 2);
    }
}
