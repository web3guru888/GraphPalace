//! High-level palace operations for the WASM API.
//!
//! Composes lower-level crate functions into a coherent palace interface
//! that can be called from JavaScript. All operations work on an in-memory
//! palace representation.

use std::collections::HashMap;

use gp_core::config::GraphPalaceConfig;
use gp_core::types::*;
use gp_embeddings::similarity::cosine_similarity;
use gp_stigmergy::cost::recompute_edge_cost;
use gp_stigmergy::decay::{decay_edge_pheromones, decay_node_pheromones};
use gp_stigmergy::rewards::{deposit_exploration, deposit_path_success};
use serde::{Deserialize, Serialize};

/// An in-memory palace graph (WASM-friendly, no external DB needed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InMemoryPalace {
    /// Palace metadata.
    pub palace: Palace,
    /// Wings indexed by ID.
    pub wings: HashMap<String, Wing>,
    /// Rooms indexed by ID.
    pub rooms: HashMap<String, Room>,
    /// Closets indexed by ID.
    pub closets: HashMap<String, Closet>,
    /// Drawers indexed by ID.
    pub drawers: HashMap<String, Drawer>,
    /// Entities indexed by ID.
    pub entities: HashMap<String, Entity>,
    /// Agents indexed by ID.
    pub agents: HashMap<String, Agent>,
    /// Parent relationships: child_id → parent_id.
    pub parent_map: HashMap<String, String>,
    /// Edge pheromones: "from:to" → EdgePheromones.
    pub edge_pheromones: HashMap<String, EdgePheromones>,
    /// Edge costs: "from:to" → EdgeCost.
    pub edge_costs: HashMap<String, EdgeCost>,
    /// Configuration.
    pub config: GraphPalaceConfig,
    /// Auto-incrementing ID counter.
    next_id: u64,
}

impl InMemoryPalace {
    /// Create a new empty palace.
    pub fn new(name: &str) -> Self {
        let palace = Palace {
            id: "palace_root".to_string(),
            name: name.to_string(),
            description: String::new(),
            created_at: chrono::Utc::now(),
        };
        Self {
            palace,
            wings: HashMap::new(),
            rooms: HashMap::new(),
            closets: HashMap::new(),
            drawers: HashMap::new(),
            entities: HashMap::new(),
            agents: HashMap::new(),
            parent_map: HashMap::new(),
            edge_pheromones: HashMap::new(),
            edge_costs: HashMap::new(),
            config: GraphPalaceConfig::default(),
            next_id: 1,
        }
    }

    /// Generate a unique ID.
    fn gen_id(&mut self, prefix: &str) -> String {
        let id = format!("{prefix}_{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Edge key for pheromone/cost lookups.
    fn edge_key(from: &str, to: &str) -> String {
        format!("{from}:{to}")
    }

    // ─── Wing Operations ──────────────────────────────────────────────

    /// Add a wing to the palace.
    pub fn add_wing(&mut self, name: &str, wing_type: WingType, description: &str) -> String {
        let id = self.gen_id("wing");
        let wing = Wing {
            id: id.clone(),
            name: name.to_string(),
            wing_type,
            description: description.to_string(),
            embedding: zero_embedding(),
            pheromones: NodePheromones::default(),
            created_at: chrono::Utc::now(),
        };
        self.wings.insert(id.clone(), wing);
        self.parent_map
            .insert(id.clone(), self.palace.id.clone());
        let key = Self::edge_key(&self.palace.id, &id);
        self.edge_pheromones
            .insert(key.clone(), EdgePheromones::default());
        self.edge_costs.insert(key, EdgeCost::new(0.2));
        id
    }

    /// List all wings.
    pub fn list_wings(&self) -> Vec<&Wing> {
        self.wings.values().collect()
    }

    /// Get a wing by ID.
    pub fn get_wing(&self, id: &str) -> Option<&Wing> {
        self.wings.get(id)
    }

    // ─── Room Operations ──────────────────────────────────────────────

    /// Add a room to a wing.
    pub fn add_room(&mut self, wing_id: &str, name: &str, hall_type: HallType) -> Option<String> {
        if !self.wings.contains_key(wing_id) {
            return None;
        }
        let id = self.gen_id("room");
        let room = Room {
            id: id.clone(),
            name: name.to_string(),
            hall_type,
            description: String::new(),
            embedding: zero_embedding(),
            pheromones: NodePheromones::default(),
            created_at: chrono::Utc::now(),
        };
        self.rooms.insert(id.clone(), room);
        self.parent_map.insert(id.clone(), wing_id.to_string());
        let key = Self::edge_key(wing_id, &id);
        self.edge_pheromones
            .insert(key.clone(), EdgePheromones::default());
        self.edge_costs.insert(key, EdgeCost::new(0.3));
        Some(id)
    }

    /// List rooms in a wing.
    pub fn list_rooms(&self, wing_id: &str) -> Vec<&Room> {
        self.rooms
            .values()
            .filter(|r| self.parent_map.get(&r.id).map(|p| p.as_str()) == Some(wing_id))
            .collect()
    }

    // ─── Closet Operations ────────────────────────────────────────────

    /// Add a closet to a room.
    pub fn add_closet(&mut self, room_id: &str, name: &str, summary: &str) -> Option<String> {
        if !self.rooms.contains_key(room_id) {
            return None;
        }
        let id = self.gen_id("closet");
        let closet = Closet {
            id: id.clone(),
            name: name.to_string(),
            summary: summary.to_string(),
            embedding: zero_embedding(),
            pheromones: NodePheromones::default(),
            drawer_count: 0,
            created_at: chrono::Utc::now(),
        };
        self.closets.insert(id.clone(), closet);
        self.parent_map.insert(id.clone(), room_id.to_string());
        let key = Self::edge_key(room_id, &id);
        self.edge_pheromones
            .insert(key.clone(), EdgePheromones::default());
        self.edge_costs.insert(key, EdgeCost::new(0.3));
        Some(id)
    }

    // ─── Drawer Operations ────────────────────────────────────────────

    /// Add a drawer to a closet.
    pub fn add_drawer(
        &mut self,
        closet_id: &str,
        content: &str,
        source: DrawerSource,
    ) -> Option<String> {
        if !self.closets.contains_key(closet_id) {
            return None;
        }
        let id = self.gen_id("drawer");
        let now = chrono::Utc::now();
        let drawer = Drawer {
            id: id.clone(),
            content: content.to_string(),
            embedding: zero_embedding(),
            source,
            source_file: None,
            importance: 0.5,
            pheromones: NodePheromones::default(),
            created_at: now,
            accessed_at: now,
        };
        self.drawers.insert(id.clone(), drawer);
        self.parent_map.insert(id.clone(), closet_id.to_string());
        let key = Self::edge_key(closet_id, &id);
        self.edge_pheromones
            .insert(key.clone(), EdgePheromones::default());
        self.edge_costs.insert(key, EdgeCost::new(0.3));

        // Update closet drawer count
        if let Some(closet) = self.closets.get_mut(closet_id) {
            closet.drawer_count += 1;
        }
        Some(id)
    }

    /// Delete a drawer by ID.
    pub fn delete_drawer(&mut self, drawer_id: &str) -> bool {
        if let Some(_drawer) = self.drawers.remove(drawer_id) {
            if let Some(parent_id) = self.parent_map.remove(drawer_id) {
                let key = Self::edge_key(&parent_id, drawer_id);
                self.edge_pheromones.remove(&key);
                self.edge_costs.remove(&key);
                if let Some(closet) = self.closets.get_mut(&parent_id) {
                    closet.drawer_count = closet.drawer_count.saturating_sub(1);
                }
            }
            true
        } else {
            false
        }
    }

    /// Get a drawer by ID.
    pub fn get_drawer(&self, id: &str) -> Option<&Drawer> {
        self.drawers.get(id)
    }

    // ─── Entity Operations ────────────────────────────────────────────

    /// Add an entity to the knowledge graph.
    pub fn add_entity(
        &mut self,
        name: &str,
        entity_type: EntityType,
        description: &str,
    ) -> String {
        let id = self.gen_id("entity");
        let entity = Entity {
            id: id.clone(),
            name: name.to_string(),
            entity_type,
            description: description.to_string(),
            embedding: zero_embedding(),
            pheromones: NodePheromones::default(),
            created_at: chrono::Utc::now(),
        };
        self.entities.insert(id.clone(), entity);
        id
    }

    /// Get an entity by ID.
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    // ─── Search Operations ────────────────────────────────────────────

    /// Search drawers by embedding similarity.
    pub fn search_drawers(&self, query_embedding: &Embedding, k: usize) -> Vec<(&Drawer, f32)> {
        let mut results: Vec<_> = self
            .drawers
            .values()
            .map(|d| {
                let sim = cosine_similarity(query_embedding, &d.embedding);
                (d, sim)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    // ─── Pheromone Operations ─────────────────────────────────────────

    /// Deposit pheromones along a path (list of node IDs).
    pub fn deposit_success_on_path(&mut self, path: &[String], base_reward: f64) {
        if path.len() < 2 {
            return;
        }
        // Collect edge pheromones for the path
        let mut edge_keys: Vec<String> = Vec::new();
        for i in 0..path.len() - 1 {
            edge_keys.push(Self::edge_key(&path[i], &path[i + 1]));
        }

        let mut edge_ph: Vec<EdgePheromones> = edge_keys
            .iter()
            .map(|k| {
                self.edge_pheromones
                    .get(k)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        let mut node_ph: Vec<NodePheromones> = path
            .iter()
            .map(|id| {
                self.wings
                    .get(id)
                    .map(|w| w.pheromones.clone())
                    .or_else(|| self.rooms.get(id).map(|r| r.pheromones.clone()))
                    .or_else(|| self.closets.get(id).map(|c| c.pheromones.clone()))
                    .or_else(|| self.drawers.get(id).map(|d| d.pheromones.clone()))
                    .unwrap_or_default()
            })
            .collect();

        deposit_path_success(&mut edge_ph, &mut node_ph, base_reward);

        // Write back edge pheromones
        for (key, ph) in edge_keys.iter().zip(edge_ph.iter()) {
            self.edge_pheromones.insert(key.clone(), ph.clone());
        }
        // Write back node pheromones
        for (id, ph) in path.iter().zip(node_ph.iter()) {
            if let Some(w) = self.wings.get_mut(id) {
                w.pheromones = ph.clone();
            }
            if let Some(r) = self.rooms.get_mut(id) {
                r.pheromones = ph.clone();
            }
            if let Some(c) = self.closets.get_mut(id) {
                c.pheromones = ph.clone();
            }
            if let Some(d) = self.drawers.get_mut(id) {
                d.pheromones = ph.clone();
            }
        }
    }

    /// Deposit exploration pheromones on a single node.
    pub fn deposit_exploration_on_node(&mut self, node_id: &str) {
        let mut ph = self
            .wings
            .get(node_id)
            .map(|w| w.pheromones.clone())
            .or_else(|| self.rooms.get(node_id).map(|r| r.pheromones.clone()))
            .or_else(|| self.closets.get(node_id).map(|c| c.pheromones.clone()))
            .or_else(|| self.drawers.get(node_id).map(|d| d.pheromones.clone()))
            .unwrap_or_default();

        deposit_exploration(&mut ph);

        // Write back
        if let Some(w) = self.wings.get_mut(node_id) {
            w.pheromones = ph.clone();
        }
        if let Some(r) = self.rooms.get_mut(node_id) {
            r.pheromones = ph.clone();
        }
        if let Some(c) = self.closets.get_mut(node_id) {
            c.pheromones = ph.clone();
        }
        if let Some(d) = self.drawers.get_mut(node_id) {
            d.pheromones = ph;
        }
    }

    /// Apply pheromone decay to all nodes and edges.
    pub fn decay_all(&mut self) {
        let config = self.config.pheromones.clone();
        for wing in self.wings.values_mut() {
            decay_node_pheromones(&mut wing.pheromones, &config);
        }
        for room in self.rooms.values_mut() {
            decay_node_pheromones(&mut room.pheromones, &config);
        }
        for closet in self.closets.values_mut() {
            decay_node_pheromones(&mut closet.pheromones, &config);
        }
        for drawer in self.drawers.values_mut() {
            decay_node_pheromones(&mut drawer.pheromones, &config);
        }
        for edge_ph in self.edge_pheromones.values_mut() {
            decay_edge_pheromones(edge_ph, &config);
        }
    }

    /// Recompute all edge costs after pheromone changes.
    pub fn recompute_all_costs(&mut self) {
        for (key, edge_ph) in &self.edge_pheromones {
            if let Some(cost) = self.edge_costs.get_mut(key) {
                recompute_edge_cost(cost, edge_ph);
            }
        }
    }

    // ─── Statistics ───────────────────────────────────────────────────

    /// Get total node count.
    pub fn node_count(&self) -> usize {
        1 + self.wings.len()
            + self.rooms.len()
            + self.closets.len()
            + self.drawers.len()
            + self.entities.len()
            + self.agents.len()
    }

    /// Get total edge count.
    pub fn edge_count(&self) -> usize {
        self.edge_pheromones.len()
    }

    /// Export the palace as a JSON string.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Import a palace from a JSON string.
    pub fn import_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_empty_palace() {
        let palace = InMemoryPalace::new("Test Palace");
        assert_eq!(palace.palace.name, "Test Palace");
        assert_eq!(palace.wings.len(), 0);
        assert_eq!(palace.node_count(), 1); // Just the palace root
    }

    #[test]
    fn add_and_list_wings() {
        let mut palace = InMemoryPalace::new("Test");
        let id1 = palace.add_wing("Research", WingType::Domain, "Research domain");
        let id2 = palace.add_wing("Projects", WingType::Project, "Project wing");
        assert_eq!(palace.wings.len(), 2);
        assert!(palace.get_wing(&id1).is_some());
        assert!(palace.get_wing(&id2).is_some());
        assert_eq!(palace.list_wings().len(), 2);
    }

    #[test]
    fn add_room_to_wing() {
        let mut palace = InMemoryPalace::new("Test");
        let wing_id = palace.add_wing("Research", WingType::Domain, "");
        let room_id = palace.add_room(&wing_id, "Facts Room", HallType::Facts);
        assert!(room_id.is_some());
        let rooms = palace.list_rooms(&wing_id);
        assert_eq!(rooms.len(), 1);
    }

    #[test]
    fn add_room_to_nonexistent_wing_fails() {
        let mut palace = InMemoryPalace::new("Test");
        assert!(palace.add_room("nonexistent", "Room", HallType::Facts).is_none());
    }

    #[test]
    fn add_closet_and_drawer() {
        let mut palace = InMemoryPalace::new("Test");
        let wing_id = palace.add_wing("W", WingType::Domain, "");
        let room_id = palace.add_room(&wing_id, "R", HallType::Facts).unwrap();
        let closet_id = palace.add_closet(&room_id, "C", "Summary").unwrap();
        let drawer_id = palace
            .add_drawer(&closet_id, "Important fact", DrawerSource::Conversation)
            .unwrap();

        assert!(palace.get_drawer(&drawer_id).is_some());
        assert_eq!(
            palace.get_drawer(&drawer_id).unwrap().content,
            "Important fact"
        );
        assert_eq!(palace.closets.get(&closet_id).unwrap().drawer_count, 1);
    }

    #[test]
    fn delete_drawer() {
        let mut palace = InMemoryPalace::new("Test");
        let wing_id = palace.add_wing("W", WingType::Domain, "");
        let room_id = palace.add_room(&wing_id, "R", HallType::Facts).unwrap();
        let closet_id = palace.add_closet(&room_id, "C", "").unwrap();
        let drawer_id = palace
            .add_drawer(&closet_id, "Content", DrawerSource::Api)
            .unwrap();

        assert!(palace.delete_drawer(&drawer_id));
        assert!(palace.get_drawer(&drawer_id).is_none());
        assert_eq!(palace.closets.get(&closet_id).unwrap().drawer_count, 0);
    }

    #[test]
    fn delete_nonexistent_drawer() {
        let mut palace = InMemoryPalace::new("Test");
        assert!(!palace.delete_drawer("nonexistent"));
    }

    #[test]
    fn node_and_edge_counts() {
        let mut palace = InMemoryPalace::new("Test");
        assert_eq!(palace.node_count(), 1);
        assert_eq!(palace.edge_count(), 0);

        let wing_id = palace.add_wing("W", WingType::Domain, "");
        assert_eq!(palace.node_count(), 2);
        assert_eq!(palace.edge_count(), 1); // palace→wing

        let _room_id = palace.add_room(&wing_id, "R", HallType::Facts).unwrap();
        assert_eq!(palace.node_count(), 3);
        assert_eq!(palace.edge_count(), 2); // + wing→room
    }

    #[test]
    fn deposit_and_decay() {
        let mut palace = InMemoryPalace::new("Test");
        let w = palace.add_wing("W", WingType::Domain, "");
        let r = palace.add_room(&w, "R", HallType::Facts).unwrap();

        let path = vec![w.clone(), r.clone()];
        palace.deposit_success_on_path(&path, 1.0);

        // Check edge pheromones were deposited
        let key = InMemoryPalace::edge_key(&w, &r);
        let ph = palace.edge_pheromones.get(&key).unwrap();
        assert!(ph.success > 0.0);
        assert!(ph.recency > 0.0);
        let success_before = ph.success;

        // Decay
        palace.decay_all();
        let ph_after = palace.edge_pheromones.get(&key).unwrap();
        // After decay, values should be less than or equal
        assert!(ph_after.success <= success_before);
    }

    #[test]
    fn recompute_costs() {
        let mut palace = InMemoryPalace::new("Test");
        let w = palace.add_wing("W", WingType::Domain, "");
        let r = palace.add_room(&w, "R", HallType::Facts).unwrap();

        let path = vec![w.clone(), r.clone()];
        palace.deposit_success_on_path(&path, 1.0);
        palace.recompute_all_costs();

        let key = InMemoryPalace::edge_key(&w, &r);
        let cost = palace.edge_costs.get(&key).unwrap();
        assert!(
            cost.current_cost < cost.base_cost,
            "Cost should decrease after pheromone deposit"
        );
    }

    #[test]
    fn search_drawers_empty() {
        let palace = InMemoryPalace::new("Test");
        let results = palace.search_drawers(&zero_embedding(), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn export_import_roundtrip() {
        let mut palace = InMemoryPalace::new("Export Test");
        palace.add_wing("Wing1", WingType::Domain, "Test wing");

        let json = palace.export_json().unwrap();
        let imported = InMemoryPalace::import_json(&json).unwrap();

        assert_eq!(imported.palace.name, "Export Test");
        assert_eq!(imported.wings.len(), 1);
    }

    #[test]
    fn full_hierarchy() {
        let mut palace = InMemoryPalace::new("Full Test");
        let w = palace.add_wing("Research", WingType::Domain, "Research domain");
        let r = palace
            .add_room(&w, "Findings", HallType::Discoveries)
            .unwrap();
        let c = palace
            .add_closet(&r, "Key Findings", "Important discoveries")
            .unwrap();
        let d1 = palace
            .add_drawer(&c, "Finding 1", DrawerSource::Agent)
            .unwrap();
        let _d2 = palace
            .add_drawer(&c, "Finding 2", DrawerSource::Conversation)
            .unwrap();

        assert_eq!(palace.node_count(), 6); // palace + wing + room + closet + 2 drawers
        assert_eq!(palace.drawers.len(), 2);
        assert_eq!(palace.closets.get(&c).unwrap().drawer_count, 2);

        palace.delete_drawer(&d1);
        assert_eq!(palace.closets.get(&c).unwrap().drawer_count, 1);
    }

    #[test]
    fn deposit_on_short_path() {
        let mut palace = InMemoryPalace::new("Test");
        // Path with single node should be no-op
        palace.deposit_success_on_path(&["single".to_string()], 1.0);
        // Empty path
        palace.deposit_success_on_path(&[], 1.0);
    }

    #[test]
    fn add_entity() {
        let mut palace = InMemoryPalace::new("Test");
        let id = palace.add_entity("Rust", EntityType::Concept, "Programming language");
        assert!(palace.get_entity(&id).is_some());
        assert_eq!(palace.get_entity(&id).unwrap().name, "Rust");
        assert_eq!(palace.node_count(), 2); // palace root + entity
    }

    #[test]
    fn deposit_exploration() {
        let mut palace = InMemoryPalace::new("Test");
        let w = palace.add_wing("W", WingType::Domain, "");
        palace.deposit_exploration_on_node(&w);
        let wing = palace.get_wing(&w).unwrap();
        assert!(wing.pheromones.exploration > 0.0);
    }

    #[test]
    fn add_closet_to_nonexistent_room() {
        let mut palace = InMemoryPalace::new("Test");
        assert!(palace.add_closet("nonexistent", "C", "").is_none());
    }

    #[test]
    fn add_drawer_to_nonexistent_closet() {
        let mut palace = InMemoryPalace::new("Test");
        assert!(palace
            .add_drawer("nonexistent", "Content", DrawerSource::Api)
            .is_none());
    }

    #[test]
    fn gen_id_uniqueness() {
        let mut palace = InMemoryPalace::new("Test");
        let id1 = palace.gen_id("test");
        let id2 = palace.gen_id("test");
        assert_ne!(id1, id2);
    }
}
