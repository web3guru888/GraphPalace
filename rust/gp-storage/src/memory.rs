//! In-memory storage backend for GraphPalace.
//!
//! Provides a fully functional [`InMemoryBackend`] that stores palace data
//! in `HashMap`s. Supports CRUD for wings, rooms, closets, drawers, entities,
//! and relationships. Useful for testing, WASM, and scenarios where Kuzu
//! is not available.

use crate::backend::{StorageBackend, Value};
use chrono::Utc;
use gp_core::types::*;
use gp_core::{GraphPalaceError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// A relationship triple in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub observed_at: String,
}

/// Internal data store shared across threads.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PalaceData {
    pub wings: HashMap<String, Wing>,
    pub rooms: HashMap<String, Room>,
    pub closets: HashMap<String, Closet>,
    pub drawers: HashMap<String, Drawer>,
    pub entities: HashMap<String, Entity>,
    pub relationships: Vec<Relationship>,
    /// child_id → parent_id
    pub parent_map: HashMap<String, String>,
    /// "from:to" → EdgePheromones
    pub edge_pheromones: HashMap<String, EdgePheromones>,
    /// "from:to" → EdgeCost
    pub edge_costs: HashMap<String, EdgeCost>,
    /// "from:to" → similarity score for SIMILAR_TO edges between drawers.
    #[serde(default)]
    pub similarity_edges: HashMap<String, f32>,
    /// Auto-incrementing id counter.
    pub next_id: u64,
    /// Whether init_schema has been called.
    pub schema_initialized: bool,
}

impl PalaceData {
    fn gen_id(&mut self, prefix: &str) -> String {
        self.next_id += 1;
        format!("{prefix}_{}", self.next_id)
    }
}

// ---------------------------------------------------------------------------
// InMemoryBackend
// ---------------------------------------------------------------------------

/// A pure-Rust in-memory storage backend. Thread-safe via `RwLock`.
#[derive(Debug, Clone)]
pub struct InMemoryBackend {
    data: Arc<RwLock<PalaceData>>,
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryBackend {
    /// Create a new empty backend.
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(PalaceData::default())),
        }
    }

    /// Create a backend pre-loaded with data.
    pub fn with_data(data: PalaceData) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
        }
    }

    /// Get read access to the underlying data.
    pub fn read_data(&self) -> std::sync::RwLockReadGuard<'_, PalaceData> {
        self.data.read().unwrap()
    }

    /// Get write access to the underlying data.
    pub fn write_data(&self) -> std::sync::RwLockWriteGuard<'_, PalaceData> {
        self.data.write().unwrap()
    }

    /// Snapshot the palace data for serialization.
    pub fn snapshot(&self) -> PalaceData {
        self.read_data().clone()
    }

    /// Restore palace data from a snapshot.
    pub fn restore(&self, data: PalaceData) {
        *self.write_data() = data;
    }

    // -- Wing CRUD -----------------------------------------------------------

    pub fn create_wing(
        &self,
        name: &str,
        wing_type: WingType,
        description: &str,
        embedding: Embedding,
    ) -> Result<String> {
        let mut d = self.write_data();
        // Check for duplicates
        for w in d.wings.values() {
            if w.name == name {
                return Err(GraphPalaceError::DuplicateNode {
                    id: name.to_string(),
                });
            }
        }
        let id = d.gen_id("wing");
        let wing = Wing {
            id: id.clone(),
            name: name.to_string(),
            wing_type,
            description: description.to_string(),
            embedding,
            pheromones: NodePheromones::default(),
            created_at: Utc::now(),
        };
        d.wings.insert(id.clone(), wing);
        Ok(id)
    }

    pub fn get_wing(&self, id: &str) -> Result<Wing> {
        let d = self.read_data();
        d.wings
            .get(id)
            .cloned()
            .ok_or_else(|| GraphPalaceError::WingNotFound {
                name: id.to_string(),
            })
    }

    pub fn list_wings(&self) -> Vec<Wing> {
        let d = self.read_data();
        d.wings.values().cloned().collect()
    }

    pub fn find_wing_by_name(&self, name: &str) -> Option<Wing> {
        let d = self.read_data();
        d.wings.values().find(|w| w.name == name).cloned()
    }

    // -- Room CRUD -----------------------------------------------------------

    pub fn create_room(
        &self,
        wing_id: &str,
        name: &str,
        hall_type: HallType,
        description: &str,
        embedding: Embedding,
    ) -> Result<String> {
        let mut d = self.write_data();
        if !d.wings.contains_key(wing_id) {
            return Err(GraphPalaceError::WingNotFound {
                name: wing_id.to_string(),
            });
        }
        let id = d.gen_id("room");
        let room = Room {
            id: id.clone(),
            name: name.to_string(),
            hall_type,
            description: description.to_string(),
            embedding,
            pheromones: NodePheromones::default(),
            created_at: Utc::now(),
        };
        d.rooms.insert(id.clone(), room);
        d.parent_map.insert(id.clone(), wing_id.to_string());
        // Create containment edge pheromones
        let edge_key = format!("{wing_id}:{id}");
        d.edge_pheromones
            .insert(edge_key.clone(), EdgePheromones::default());
        d.edge_costs.insert(edge_key, EdgeCost::new(0.3));
        Ok(id)
    }

    pub fn get_room(&self, id: &str) -> Result<Room> {
        let d = self.read_data();
        d.rooms
            .get(id)
            .cloned()
            .ok_or_else(|| GraphPalaceError::RoomNotFound {
                name: id.to_string(),
                wing: String::new(),
            })
    }

    pub fn list_rooms(&self, wing_id: &str) -> Vec<Room> {
        let d = self.read_data();
        d.parent_map
            .iter()
            .filter(|(_, parent)| *parent == wing_id)
            .filter_map(|(child, _)| d.rooms.get(child).cloned())
            .collect()
    }

    // -- Closet CRUD ---------------------------------------------------------

    pub fn create_closet(
        &self,
        room_id: &str,
        name: &str,
        summary: &str,
        embedding: Embedding,
    ) -> Result<String> {
        let mut d = self.write_data();
        if !d.rooms.contains_key(room_id) {
            return Err(GraphPalaceError::RoomNotFound {
                name: room_id.to_string(),
                wing: String::new(),
            });
        }
        let id = d.gen_id("closet");
        let closet = Closet {
            id: id.clone(),
            name: name.to_string(),
            summary: summary.to_string(),
            embedding,
            pheromones: NodePheromones::default(),
            drawer_count: 0,
            created_at: Utc::now(),
        };
        d.closets.insert(id.clone(), closet);
        d.parent_map.insert(id.clone(), room_id.to_string());
        let edge_key = format!("{room_id}:{id}");
        d.edge_pheromones
            .insert(edge_key.clone(), EdgePheromones::default());
        d.edge_costs.insert(edge_key, EdgeCost::new(0.3));
        Ok(id)
    }

    pub fn get_closet(&self, id: &str) -> Result<Closet> {
        let d = self.read_data();
        d.closets
            .get(id)
            .cloned()
            .ok_or_else(|| GraphPalaceError::NodeNotFound {
                id: id.to_string(),
            })
    }

    // -- Drawer CRUD ---------------------------------------------------------

    pub fn create_drawer(
        &self,
        closet_id: &str,
        content: &str,
        embedding: Embedding,
        source: DrawerSource,
        source_file: Option<&str>,
        importance: f64,
    ) -> Result<String> {
        let mut d = self.write_data();
        if !d.closets.contains_key(closet_id) {
            return Err(GraphPalaceError::NodeNotFound {
                id: closet_id.to_string(),
            });
        }
        let id = d.gen_id("drawer");
        let now = Utc::now();
        let drawer = Drawer {
            id: id.clone(),
            content: content.to_string(),
            embedding,
            source,
            source_file: source_file.map(|s| s.to_string()),
            importance,
            pheromones: NodePheromones::default(),
            created_at: now,
            accessed_at: now,
        };
        d.drawers.insert(id.clone(), drawer);
        d.parent_map.insert(id.clone(), closet_id.to_string());
        // Increment closet drawer count
        if let Some(closet) = d.closets.get_mut(closet_id) {
            closet.drawer_count += 1;
        }
        let edge_key = format!("{closet_id}:{id}");
        d.edge_pheromones
            .insert(edge_key.clone(), EdgePheromones::default());
        d.edge_costs.insert(edge_key, EdgeCost::new(0.3));
        Ok(id)
    }

    pub fn get_drawer(&self, id: &str) -> Result<Drawer> {
        let d = self.read_data();
        d.drawers
            .get(id)
            .cloned()
            .ok_or_else(|| GraphPalaceError::NodeNotFound {
                id: id.to_string(),
            })
    }

    pub fn delete_drawer(&self, id: &str) -> Result<()> {
        let mut d = self.write_data();
        if d.drawers.remove(id).is_none() {
            return Err(GraphPalaceError::NodeNotFound {
                id: id.to_string(),
            });
        }
        // Decrement closet count
        if let Some(parent_id) = d.parent_map.remove(id)
            && let Some(closet) = d.closets.get_mut(&parent_id)
        {
            closet.drawer_count = closet.drawer_count.saturating_sub(1);
        }
        // Remove edge pheromones/costs where drawer is involved
        d.edge_pheromones.retain(|k, _| !k.contains(id));
        d.edge_costs.retain(|k, _| !k.contains(id));
        Ok(())
    }

    /// Search drawers by cosine similarity to `query_embedding`.
    pub fn search_drawers(
        &self,
        query_embedding: &Embedding,
        k: usize,
        threshold: f32,
    ) -> Vec<(Drawer, f32)> {
        let d = self.read_data();
        let mut scored: Vec<(Drawer, f32)> = d
            .drawers
            .values()
            .map(|drawer| {
                let sim = gp_embeddings::similarity::cosine_similarity(
                    query_embedding,
                    &drawer.embedding,
                );
                (drawer.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    // -- Entity CRUD ---------------------------------------------------------

    pub fn create_entity(
        &self,
        name: &str,
        entity_type: EntityType,
        description: &str,
        embedding: Embedding,
    ) -> Result<String> {
        let mut d = self.write_data();
        let id = d.gen_id("entity");
        let entity = Entity {
            id: id.clone(),
            name: name.to_string(),
            entity_type,
            description: description.to_string(),
            embedding,
            pheromones: NodePheromones::default(),
            created_at: Utc::now(),
        };
        d.entities.insert(id.clone(), entity);
        Ok(id)
    }

    pub fn get_entity(&self, id: &str) -> Result<Entity> {
        let d = self.read_data();
        d.entities
            .get(id)
            .cloned()
            .ok_or_else(|| GraphPalaceError::NodeNotFound {
                id: id.to_string(),
            })
    }

    pub fn find_entity_by_name(&self, name: &str) -> Option<Entity> {
        let d = self.read_data();
        d.entities.values().find(|e| e.name == name).cloned()
    }

    // -- Relationship CRUD ---------------------------------------------------

    pub fn add_relationship(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
    ) -> Result<String> {
        let mut d = self.write_data();
        let id = d.gen_id("rel");
        let rel = Relationship {
            id: id.clone(),
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence,
            valid_from: None,
            valid_to: None,
            observed_at: Utc::now().to_rfc3339(),
        };
        d.relationships.push(rel);
        // Edge pheromones for the relationship
        let edge_key = format!("{subject}:{object}");
        d.edge_pheromones
            .entry(edge_key.clone())
            .or_default();
        d.edge_costs
            .entry(edge_key)
            .or_insert_with(|| EdgeCost::new(1.0));
        Ok(id)
    }

    pub fn query_relationships(&self, entity: &str) -> Vec<Relationship> {
        let d = self.read_data();
        d.relationships
            .iter()
            .filter(|r| r.subject == entity || r.object == entity)
            .cloned()
            .collect()
    }

    // -- Pheromone operations ------------------------------------------------

    pub fn deposit_edge_pheromones(
        &self,
        from: &str,
        to: &str,
        success: f64,
        traversal: f64,
        recency: f64,
    ) {
        let mut d = self.write_data();
        let key = format!("{from}:{to}");
        let ph = d
            .edge_pheromones
            .entry(key)
            .or_default();
        ph.success += success;
        ph.traversal += traversal;
        ph.recency = recency.max(ph.recency);
    }

    pub fn deposit_node_pheromones(
        &self,
        node_id: &str,
        exploitation: f64,
        exploration: f64,
    ) {
        let mut d = self.write_data();
        // Check all node types
        if let Some(w) = d.wings.get_mut(node_id) {
            w.pheromones.exploitation += exploitation;
            w.pheromones.exploration += exploration;
        } else if let Some(r) = d.rooms.get_mut(node_id) {
            r.pheromones.exploitation += exploitation;
            r.pheromones.exploration += exploration;
        } else if let Some(c) = d.closets.get_mut(node_id) {
            c.pheromones.exploitation += exploitation;
            c.pheromones.exploration += exploration;
        } else if let Some(dr) = d.drawers.get_mut(node_id) {
            dr.pheromones.exploitation += exploitation;
            dr.pheromones.exploration += exploration;
        } else if let Some(e) = d.entities.get_mut(node_id) {
            e.pheromones.exploitation += exploitation;
            e.pheromones.exploration += exploration;
        }
    }

    pub fn decay_all_pheromones(&self, config: &gp_core::PheromoneConfig) {
        let mut d = self.write_data();
        // Decay node pheromones
        for w in d.wings.values_mut() {
            gp_stigmergy::decay::decay_node_pheromones(&mut w.pheromones, config);
        }
        for r in d.rooms.values_mut() {
            gp_stigmergy::decay::decay_node_pheromones(&mut r.pheromones, config);
        }
        for c in d.closets.values_mut() {
            gp_stigmergy::decay::decay_node_pheromones(&mut c.pheromones, config);
        }
        for dr in d.drawers.values_mut() {
            gp_stigmergy::decay::decay_node_pheromones(&mut dr.pheromones, config);
        }
        for e in d.entities.values_mut() {
            gp_stigmergy::decay::decay_node_pheromones(&mut e.pheromones, config);
        }
        // Decay edge pheromones
        for ph in d.edge_pheromones.values_mut() {
            gp_stigmergy::decay::decay_edge_pheromones(ph, config);
        }
        // Recompute edge costs — collect keys first to avoid overlapping borrows
        let keys: Vec<String> = d.edge_costs.keys().cloned().collect();
        for key in &keys {
            if let Some(ph) = d.edge_pheromones.get(key).cloned()
                && let Some(cost) = d.edge_costs.get_mut(key)
            {
                gp_stigmergy::cost::recompute_edge_cost(cost, &ph);
            }
        }
    }

    /// Get the k edges with highest success pheromones.
    pub fn hot_paths(&self, k: usize) -> Vec<(String, String, f64)> {
        let d = self.read_data();
        let mut edges: Vec<_> = d
            .edge_pheromones
            .iter()
            .map(|(key, ph)| {
                let parts: Vec<&str> = key.split(':').collect();
                let from = parts.first().unwrap_or(&"").to_string();
                let to = parts.get(1).unwrap_or(&"").to_string();
                (from, to, ph.success)
            })
            .collect();
        edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        edges.truncate(k);
        edges
    }

    /// Get the k nodes with lowest combined pheromones (cold spots).
    pub fn cold_spots(&self, k: usize) -> Vec<(String, String, f64)> {
        let d = self.read_data();
        let mut nodes: Vec<(String, String, f64)> = Vec::new();
        for (id, w) in &d.wings {
            let total = w.pheromones.exploitation + w.pheromones.exploration;
            nodes.push((id.clone(), w.name.clone(), total));
        }
        for (id, r) in &d.rooms {
            let total = r.pheromones.exploitation + r.pheromones.exploration;
            nodes.push((id.clone(), r.name.clone(), total));
        }
        for (id, c) in &d.closets {
            let total = c.pheromones.exploitation + c.pheromones.exploration;
            nodes.push((id.clone(), c.name.clone(), total));
        }
        for (id, dr) in &d.drawers {
            let total = dr.pheromones.exploitation + dr.pheromones.exploration;
            nodes.push((
                id.clone(),
                dr.content.chars().take(40).collect::<String>(),
                total,
            ));
        }
        nodes.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        nodes.truncate(k);
        nodes
    }

    // -- Similarity edges ----------------------------------------------------

    /// Add a similarity edge between two drawers.
    ///
    /// Stores the score in `similarity_edges` and also populates
    /// `edge_pheromones` / `edge_costs` so that pheromone-related
    /// bookkeeping (decay, hot paths) works automatically.
    pub fn add_similarity_edge(&self, drawer_a: &str, drawer_b: &str, similarity: f32) {
        let mut d = self.write_data();
        let key = format!("{drawer_a}:{drawer_b}");
        d.similarity_edges.insert(key.clone(), similarity);
        d.edge_pheromones.entry(key.clone()).or_default();
        d.edge_costs
            .entry(key)
            .or_insert_with(|| EdgeCost::new(1.0 - similarity as f64));
    }

    /// Remove a similarity edge between two drawers (either direction).
    pub fn remove_similarity_edge(&self, drawer_a: &str, drawer_b: &str) -> bool {
        let mut d = self.write_data();
        let key_ab = format!("{drawer_a}:{drawer_b}");
        let key_ba = format!("{drawer_b}:{drawer_a}");
        let removed_ab = d.similarity_edges.remove(&key_ab).is_some();
        let removed_ba = d.similarity_edges.remove(&key_ba).is_some();
        if removed_ab {
            d.edge_pheromones.remove(&key_ab);
            d.edge_costs.remove(&key_ab);
        }
        if removed_ba {
            d.edge_pheromones.remove(&key_ba);
            d.edge_costs.remove(&key_ba);
        }
        removed_ab || removed_ba
    }

    /// Compute pairwise cosine similarity between all drawers and add
    /// `SIMILAR_TO` edges for pairs above `threshold`.
    ///
    /// Returns the number of edges added.
    pub fn add_similarity_edges(&self, threshold: f32) -> usize {
        // Collect drawer IDs and embeddings (read lock)
        let drawer_data: Vec<(String, Embedding)> = {
            let d = self.read_data();
            d.drawers
                .values()
                .map(|dr| (dr.id.clone(), dr.embedding))
                .collect()
        };

        let mut count = 0;
        for i in 0..drawer_data.len() {
            for j in (i + 1)..drawer_data.len() {
                let sim = gp_embeddings::similarity::cosine_similarity(
                    &drawer_data[i].1,
                    &drawer_data[j].1,
                );
                if sim >= threshold {
                    self.add_similarity_edge(&drawer_data[i].0, &drawer_data[j].0, sim);
                    count += 1;
                }
            }
        }
        count
    }

    /// Return the number of SIMILAR_TO edges currently stored.
    pub fn similarity_edge_count(&self) -> usize {
        self.read_data().similarity_edges.len()
    }

    /// Return all similarity edges as `(from, to, similarity)` triples.
    pub fn list_similarity_edges(&self) -> Vec<(String, String, f32)> {
        let d = self.read_data();
        d.similarity_edges
            .iter()
            .filter_map(|(key, &sim)| {
                let parts: Vec<&str> = key.split(':').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string(), sim))
                } else {
                    None
                }
            })
            .collect()
    }

    // -- Taxonomy / stats ----------------------------------------------------

    pub fn wing_count(&self) -> usize {
        self.read_data().wings.len()
    }

    pub fn room_count(&self) -> usize {
        self.read_data().rooms.len()
    }

    pub fn closet_count(&self) -> usize {
        self.read_data().closets.len()
    }

    pub fn drawer_count(&self) -> usize {
        self.read_data().drawers.len()
    }

    pub fn entity_count(&self) -> usize {
        self.read_data().entities.len()
    }

    pub fn relationship_count(&self) -> usize {
        self.read_data().relationships.len()
    }

    pub fn edge_count(&self) -> usize {
        self.read_data().edge_pheromones.len()
    }

    pub fn total_pheromone_mass(&self) -> f64 {
        let d = self.read_data();
        let mut total = 0.0;
        for w in d.wings.values() {
            total += w.pheromones.exploitation + w.pheromones.exploration;
        }
        for r in d.rooms.values() {
            total += r.pheromones.exploitation + r.pheromones.exploration;
        }
        for c in d.closets.values() {
            total += c.pheromones.exploitation + c.pheromones.exploration;
        }
        for dr in d.drawers.values() {
            total += dr.pheromones.exploitation + dr.pheromones.exploration;
        }
        for e in d.entities.values() {
            total += e.pheromones.exploitation + e.pheromones.exploration;
        }
        for ph in d.edge_pheromones.values() {
            total += ph.success + ph.traversal + ph.recency;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// StorageBackend implementation
// ---------------------------------------------------------------------------

impl StorageBackend for InMemoryBackend {
    fn execute_query(&self, cypher: &str) -> Result<Vec<HashMap<String, Value>>> {
        // The in-memory backend doesn't parse Cypher — it provides direct
        // methods instead. However, we support a few simple query patterns
        // for integration testing.
        let lower = cypher.to_lowercase();
        let lower = lower.trim();

        if lower.starts_with("match") && lower.contains("wing") && lower.contains("return") {
            // Return all wings
            let d = self.read_data();
            let rows: Vec<_> = d
                .wings
                .values()
                .map(|w| {
                    let mut m = HashMap::new();
                    m.insert("id".to_string(), Value::String(w.id.clone()));
                    m.insert("name".to_string(), Value::String(w.name.clone()));
                    m.insert(
                        "wing_type".to_string(),
                        Value::String(format!("{:?}", w.wing_type)),
                    );
                    m.insert(
                        "description".to_string(),
                        Value::String(w.description.clone()),
                    );
                    m
                })
                .collect();
            return Ok(rows);
        }

        if lower.starts_with("match") && lower.contains("drawer") && lower.contains("return") {
            let d = self.read_data();
            let rows: Vec<_> = d
                .drawers
                .values()
                .map(|dr| {
                    let mut m = HashMap::new();
                    m.insert("id".to_string(), Value::String(dr.id.clone()));
                    m.insert("content".to_string(), Value::String(dr.content.clone()));
                    m.insert(
                        "importance".to_string(),
                        Value::Float(dr.importance),
                    );
                    m
                })
                .collect();
            return Ok(rows);
        }

        if lower.starts_with("match") && lower.contains("entity") && lower.contains("return") {
            let d = self.read_data();
            let rows: Vec<_> = d
                .entities
                .values()
                .map(|e| {
                    let mut m = HashMap::new();
                    m.insert("id".to_string(), Value::String(e.id.clone()));
                    m.insert("name".to_string(), Value::String(e.name.clone()));
                    m.insert(
                        "entity_type".to_string(),
                        Value::String(format!("{:?}", e.entity_type)),
                    );
                    m
                })
                .collect();
            return Ok(rows);
        }

        // Fallback: return empty result for unknown queries
        Ok(Vec::new())
    }

    fn execute_write(&self, _cypher: &str) -> Result<u64> {
        // The in-memory backend uses direct methods; write queries
        // are acknowledged but don't parse Cypher.
        Ok(0)
    }

    fn init_schema(&self) -> Result<()> {
        let mut d = self.write_data();
        d.schema_initialized = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::types::*;

    fn zero_emb() -> Embedding {
        [0.0f32; 384]
    }

    fn make_emb(seed: u8) -> Embedding {
        let mut emb = [0.0f32; 384];
        for (i, v) in emb.iter_mut().enumerate() {
            *v = ((seed as f32 + i as f32) * 0.01).sin();
        }
        // Normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in emb.iter_mut() {
                *v /= norm;
            }
        }
        emb
    }

    #[test]
    fn test_new_backend_is_empty() {
        let b = InMemoryBackend::new();
        assert_eq!(b.wing_count(), 0);
        assert_eq!(b.room_count(), 0);
        assert_eq!(b.drawer_count(), 0);
        assert_eq!(b.entity_count(), 0);
    }

    #[test]
    fn test_create_wing() {
        let b = InMemoryBackend::new();
        let id = b
            .create_wing("Science", WingType::Domain, "Scientific knowledge", zero_emb())
            .unwrap();
        assert!(id.starts_with("wing_"));
        assert_eq!(b.wing_count(), 1);
        let wing = b.get_wing(&id).unwrap();
        assert_eq!(wing.name, "Science");
    }

    #[test]
    fn test_duplicate_wing_rejected() {
        let b = InMemoryBackend::new();
        b.create_wing("Science", WingType::Domain, "desc", zero_emb())
            .unwrap();
        let result = b.create_wing("Science", WingType::Domain, "desc2", zero_emb());
        assert!(result.is_err());
    }

    #[test]
    fn test_list_wings() {
        let b = InMemoryBackend::new();
        b.create_wing("A", WingType::Domain, "", zero_emb()).unwrap();
        b.create_wing("B", WingType::Project, "", zero_emb()).unwrap();
        assert_eq!(b.list_wings().len(), 2);
    }

    #[test]
    fn test_find_wing_by_name() {
        let b = InMemoryBackend::new();
        b.create_wing("Science", WingType::Domain, "", zero_emb()).unwrap();
        assert!(b.find_wing_by_name("Science").is_some());
        assert!(b.find_wing_by_name("Art").is_none());
    }

    #[test]
    fn test_create_room() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b
            .create_room(&wid, "Physics", HallType::Facts, "Physics facts", zero_emb())
            .unwrap();
        assert!(rid.starts_with("room_"));
        assert_eq!(b.room_count(), 1);
    }

    #[test]
    fn test_create_room_invalid_wing() {
        let b = InMemoryBackend::new();
        let result = b.create_room("fake_wing", "R", HallType::Facts, "", zero_emb());
        assert!(result.is_err());
    }

    #[test]
    fn test_list_rooms_by_wing() {
        let b = InMemoryBackend::new();
        let w1 = b.create_wing("W1", WingType::Domain, "", zero_emb()).unwrap();
        let w2 = b.create_wing("W2", WingType::Project, "", zero_emb()).unwrap();
        b.create_room(&w1, "R1", HallType::Facts, "", zero_emb()).unwrap();
        b.create_room(&w1, "R2", HallType::Events, "", zero_emb()).unwrap();
        b.create_room(&w2, "R3", HallType::Discoveries, "", zero_emb()).unwrap();
        assert_eq!(b.list_rooms(&w1).len(), 2);
        assert_eq!(b.list_rooms(&w2).len(), 1);
    }

    #[test]
    fn test_create_closet() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = b
            .create_closet(&rid, "Closet1", "Summary", zero_emb())
            .unwrap();
        assert!(cid.starts_with("closet_"));
        assert_eq!(b.closet_count(), 1);
    }

    #[test]
    fn test_create_closet_invalid_room() {
        let b = InMemoryBackend::new();
        let result = b.create_closet("fake_room", "C", "S", zero_emb());
        assert!(result.is_err());
    }

    #[test]
    fn test_create_drawer() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = b.create_closet(&rid, "C", "S", zero_emb()).unwrap();
        let did = b
            .create_drawer(&cid, "The sky is blue", make_emb(1), DrawerSource::Conversation, None, 0.5)
            .unwrap();
        assert!(did.starts_with("drawer_"));
        assert_eq!(b.drawer_count(), 1);
        let closet = b.get_closet(&cid).unwrap();
        assert_eq!(closet.drawer_count, 1);
    }

    #[test]
    fn test_create_drawer_invalid_closet() {
        let b = InMemoryBackend::new();
        let result = b.create_drawer("fake", "content", zero_emb(), DrawerSource::Conversation, None, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_drawer() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = b.create_closet(&rid, "C", "S", zero_emb()).unwrap();
        let did = b
            .create_drawer(&cid, "test content", make_emb(1), DrawerSource::File, Some("test.txt"), 0.8)
            .unwrap();
        let drawer = b.get_drawer(&did).unwrap();
        assert_eq!(drawer.content, "test content");
        assert_eq!(drawer.source, DrawerSource::File);
        assert_eq!(drawer.source_file, Some("test.txt".to_string()));
        assert!((drawer.importance - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_delete_drawer() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = b.create_closet(&rid, "C", "S", zero_emb()).unwrap();
        let did = b
            .create_drawer(&cid, "content", zero_emb(), DrawerSource::Conversation, None, 0.5)
            .unwrap();
        assert_eq!(b.drawer_count(), 1);
        b.delete_drawer(&did).unwrap();
        assert_eq!(b.drawer_count(), 0);
        let closet = b.get_closet(&cid).unwrap();
        assert_eq!(closet.drawer_count, 0);
    }

    #[test]
    fn test_delete_nonexistent_drawer() {
        let b = InMemoryBackend::new();
        assert!(b.delete_drawer("fake").is_err());
    }

    #[test]
    fn test_search_drawers() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = b.create_closet(&rid, "C", "S", zero_emb()).unwrap();
        let emb1 = make_emb(1);
        let emb2 = make_emb(2);
        let emb3 = make_emb(100);
        b.create_drawer(&cid, "similar to query", emb1, DrawerSource::Conversation, None, 0.5)
            .unwrap();
        b.create_drawer(&cid, "also similar", emb2, DrawerSource::Conversation, None, 0.5)
            .unwrap();
        b.create_drawer(&cid, "very different", emb3, DrawerSource::Conversation, None, 0.5)
            .unwrap();
        let results = b.search_drawers(&emb1, 2, 0.0);
        assert_eq!(results.len(), 2);
        // First result should be exact match (similarity ~1.0)
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_search_with_threshold() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = b.create_closet(&rid, "C", "S", zero_emb()).unwrap();
        b.create_drawer(&cid, "content", make_emb(1), DrawerSource::Conversation, None, 0.5)
            .unwrap();
        // With very high threshold, nothing matches
        let results = b.search_drawers(&make_emb(200), 10, 0.99);
        assert!(results.is_empty());
    }

    #[test]
    fn test_create_entity() {
        let b = InMemoryBackend::new();
        let id = b
            .create_entity("Albert Einstein", EntityType::Person, "Physicist", zero_emb())
            .unwrap();
        assert!(id.starts_with("entity_"));
        assert_eq!(b.entity_count(), 1);
        let entity = b.get_entity(&id).unwrap();
        assert_eq!(entity.name, "Albert Einstein");
    }

    #[test]
    fn test_find_entity_by_name() {
        let b = InMemoryBackend::new();
        b.create_entity("Einstein", EntityType::Person, "Physicist", zero_emb())
            .unwrap();
        assert!(b.find_entity_by_name("Einstein").is_some());
        assert!(b.find_entity_by_name("Newton").is_none());
    }

    #[test]
    fn test_add_relationship() {
        let b = InMemoryBackend::new();
        let e1 = b
            .create_entity("Einstein", EntityType::Person, "", zero_emb())
            .unwrap();
        let e2 = b
            .create_entity("Relativity", EntityType::Concept, "", zero_emb())
            .unwrap();
        let rid = b.add_relationship(&e1, "discovered", &e2, 0.99).unwrap();
        assert!(rid.starts_with("rel_"));
        assert_eq!(b.relationship_count(), 1);
    }

    #[test]
    fn test_query_relationships() {
        let b = InMemoryBackend::new();
        let e1 = b
            .create_entity("Einstein", EntityType::Person, "", zero_emb())
            .unwrap();
        let e2 = b
            .create_entity("Relativity", EntityType::Concept, "", zero_emb())
            .unwrap();
        let e3 = b
            .create_entity("Photoelectric", EntityType::Concept, "", zero_emb())
            .unwrap();
        b.add_relationship(&e1, "discovered", &e2, 0.99).unwrap();
        b.add_relationship(&e1, "explained", &e3, 0.95).unwrap();
        let rels = b.query_relationships(&e1);
        assert_eq!(rels.len(), 2);
        let rels2 = b.query_relationships(&e2);
        assert_eq!(rels2.len(), 1);
    }

    #[test]
    fn test_pheromone_deposit_and_decay() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        b.deposit_node_pheromones(&wid, 1.0, 0.5);
        let wing = b.get_wing(&wid).unwrap();
        assert!((wing.pheromones.exploitation - 1.0).abs() < 1e-10);
        assert!((wing.pheromones.exploration - 0.5).abs() < 1e-10);

        let config = gp_core::PheromoneConfig::default();
        b.decay_all_pheromones(&config);
        let wing = b.get_wing(&wid).unwrap();
        // After decay: exploitation * (1 - 0.02) = 0.98
        assert!((wing.pheromones.exploitation - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_edge_pheromone_deposit() {
        let b = InMemoryBackend::new();
        b.deposit_edge_pheromones("a", "b", 1.0, 0.5, 1.0);
        let hot = b.hot_paths(10);
        assert_eq!(hot.len(), 1);
        assert!((hot[0].2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hot_paths_ordering() {
        let b = InMemoryBackend::new();
        b.deposit_edge_pheromones("a", "b", 0.5, 0.1, 1.0);
        b.deposit_edge_pheromones("c", "d", 2.0, 0.1, 1.0);
        b.deposit_edge_pheromones("e", "f", 1.0, 0.1, 1.0);
        let hot = b.hot_paths(2);
        assert_eq!(hot.len(), 2);
        assert!((hot[0].2 - 2.0).abs() < 1e-10); // highest first
    }

    #[test]
    fn test_cold_spots() {
        let b = InMemoryBackend::new();
        b.create_wing("Hot", WingType::Domain, "", zero_emb()).unwrap();
        let cold_id = b.create_wing("Cold", WingType::Domain, "", zero_emb()).unwrap();
        // Deposit on one wing to make it "hot"
        let hot_wing = b.list_wings().into_iter().find(|w| w.name == "Hot").unwrap();
        b.deposit_node_pheromones(&hot_wing.id, 10.0, 5.0);
        let cold = b.cold_spots(1);
        assert_eq!(cold.len(), 1);
        assert_eq!(cold[0].0, cold_id);
    }

    #[test]
    fn test_total_pheromone_mass() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        assert!((b.total_pheromone_mass() - 0.0).abs() < 1e-10);
        b.deposit_node_pheromones(&wid, 1.0, 2.0);
        b.deposit_edge_pheromones("a", "b", 3.0, 4.0, 5.0);
        assert!((b.total_pheromone_mass() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_snapshot_and_restore() {
        let b = InMemoryBackend::new();
        b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let snapshot = b.snapshot();
        assert_eq!(snapshot.wings.len(), 1);
        let b2 = InMemoryBackend::new();
        assert_eq!(b2.wing_count(), 0);
        b2.restore(snapshot);
        assert_eq!(b2.wing_count(), 1);
    }

    #[test]
    fn test_init_schema() {
        let b = InMemoryBackend::new();
        assert!(!b.read_data().schema_initialized);
        b.init_schema().unwrap();
        assert!(b.read_data().schema_initialized);
    }

    #[test]
    fn test_storage_backend_execute_query_wings() {
        let b = InMemoryBackend::new();
        b.create_wing("Science", WingType::Domain, "Science wing", zero_emb())
            .unwrap();
        let rows = b.execute_query("MATCH (w:Wing) RETURN w").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].get("name").unwrap().as_str().unwrap(),
            "Science"
        );
    }

    #[test]
    fn test_storage_backend_execute_query_drawers() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = b.create_closet(&rid, "C", "S", zero_emb()).unwrap();
        b.create_drawer(&cid, "hello world", zero_emb(), DrawerSource::Conversation, None, 0.5)
            .unwrap();
        let rows = b.execute_query("MATCH (d:Drawer) RETURN d").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_storage_backend_execute_query_entities() {
        let b = InMemoryBackend::new();
        b.create_entity("Einstein", EntityType::Person, "physicist", zero_emb())
            .unwrap();
        let rows = b.execute_query("MATCH (e:Entity) RETURN e").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_storage_backend_execute_write() {
        let b = InMemoryBackend::new();
        let result = b.execute_write("CREATE (w:Wing {name: 'test'})");
        assert!(result.is_ok());
    }

    #[test]
    fn test_storage_backend_unknown_query() {
        let b = InMemoryBackend::new();
        let rows = b.execute_query("SOME UNKNOWN QUERY").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_edge_count() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let initial_edges = b.edge_count();
        b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        assert_eq!(b.edge_count(), initial_edges + 1);
    }

    #[test]
    fn test_full_hierarchy() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("Science", WingType::Domain, "Science wing", zero_emb()).unwrap();
        let rid = b
            .create_room(&wid, "Physics", HallType::Facts, "Physics room", zero_emb())
            .unwrap();
        let cid = b
            .create_closet(&rid, "Mechanics", "Classical mechanics", zero_emb())
            .unwrap();
        let did = b
            .create_drawer(&cid, "F=ma", make_emb(1), DrawerSource::Conversation, None, 0.9)
            .unwrap();
        assert_eq!(b.wing_count(), 1);
        assert_eq!(b.room_count(), 1);
        assert_eq!(b.closet_count(), 1);
        assert_eq!(b.drawer_count(), 1);

        // Verify hierarchy via parent_map
        let d = b.read_data();
        assert_eq!(d.parent_map.get(&rid).unwrap(), &wid);
        assert_eq!(d.parent_map.get(&cid).unwrap(), &rid);
        assert_eq!(d.parent_map.get(&did).unwrap(), &cid);
    }

    #[test]
    fn test_default_backend() {
        let b = InMemoryBackend::default();
        assert_eq!(b.wing_count(), 0);
    }

    #[test]
    fn test_with_data() {
        let mut data = PalaceData::default();
        data.next_id = 100;
        let b = InMemoryBackend::with_data(data);
        let id = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        assert!(id.contains("101"));
    }

    #[test]
    fn test_multiple_drawers_per_closet() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = b.create_closet(&rid, "C", "S", zero_emb()).unwrap();
        for i in 0..5 {
            b.create_drawer(
                &cid,
                &format!("content {i}"),
                make_emb(i as u8),
                DrawerSource::Conversation,
                None,
                0.5,
            )
            .unwrap();
        }
        assert_eq!(b.drawer_count(), 5);
        let closet = b.get_closet(&cid).unwrap();
        assert_eq!(closet.drawer_count, 5);
    }

    #[test]
    fn test_search_empty_palace() {
        let b = InMemoryBackend::new();
        let results = b.search_drawers(&make_emb(1), 10, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_decay_multiple_cycles() {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        b.deposit_node_pheromones(&wid, 1.0, 0.0);
        let config = gp_core::PheromoneConfig::default();
        for _ in 0..100 {
            b.decay_all_pheromones(&config);
        }
        let wing = b.get_wing(&wid).unwrap();
        // After 100 cycles at 0.02 decay: 1.0 * (1-0.02)^100 ≈ 0.133
        assert!(wing.pheromones.exploitation < 0.14);
        assert!(wing.pheromones.exploitation > 0.12);
    }

    #[test]
    fn test_relationship_details() {
        let b = InMemoryBackend::new();
        let e1 = b.create_entity("A", EntityType::Person, "", zero_emb()).unwrap();
        let e2 = b.create_entity("B", EntityType::Concept, "", zero_emb()).unwrap();
        b.add_relationship(&e1, "knows", &e2, 0.8).unwrap();
        let rels = b.query_relationships(&e1);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].predicate, "knows");
        assert!((rels[0].confidence - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;
        let b = Arc::new(InMemoryBackend::new());
        let mut handles = vec![];
        for i in 0..10 {
            let b_clone = Arc::clone(&b);
            handles.push(thread::spawn(move || {
                b_clone
                    .create_wing(
                        &format!("Wing{i}"),
                        WingType::Domain,
                        "",
                        zero_emb(),
                    )
                    .unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(b.wing_count(), 10);
    }

    // ── Similarity edges ─────────────────────────────────────────────────

    /// Helper: build a backend with a full hierarchy and n drawers whose
    /// embeddings are created by `make_emb(seed)`.
    fn backend_with_drawers(count: u8) -> (InMemoryBackend, Vec<String>) {
        let b = InMemoryBackend::new();
        let wid = b.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = b
            .create_room(&wid, "R", HallType::Facts, "", zero_emb())
            .unwrap();
        let cid = b.create_closet(&rid, "C", "", zero_emb()).unwrap();
        let mut ids = Vec::new();
        for i in 0..count {
            let did = b
                .create_drawer(
                    &cid,
                    &format!("drawer {i}"),
                    make_emb(i),
                    DrawerSource::Conversation,
                    None,
                    0.5,
                )
                .unwrap();
            ids.push(did);
        }
        (b, ids)
    }

    #[test]
    fn test_add_similarity_edge() {
        let (b, ids) = backend_with_drawers(2);
        b.add_similarity_edge(&ids[0], &ids[1], 0.85);
        assert_eq!(b.similarity_edge_count(), 1);
    }

    #[test]
    fn test_add_similarity_edge_populates_pheromones_and_cost() {
        let (b, ids) = backend_with_drawers(2);
        b.add_similarity_edge(&ids[0], &ids[1], 0.9);
        let d = b.read_data();
        let key = format!("{}:{}", ids[0], ids[1]);
        assert!(d.edge_pheromones.contains_key(&key));
        let cost = d.edge_costs.get(&key).unwrap();
        assert!((cost.base_cost - 0.1).abs() < 1e-6, "cost should be 1.0 - 0.9");
    }

    #[test]
    fn test_add_similarity_edge_overwrites_score() {
        let (b, ids) = backend_with_drawers(2);
        b.add_similarity_edge(&ids[0], &ids[1], 0.5);
        b.add_similarity_edge(&ids[0], &ids[1], 0.9);
        let d = b.read_data();
        let key = format!("{}:{}", ids[0], ids[1]);
        let score = d.similarity_edges.get(&key).unwrap();
        assert!((*score - 0.9).abs() < 1e-6);
        // Still only 1 edge
        assert_eq!(b.similarity_edge_count(), 1);
    }

    #[test]
    fn test_remove_similarity_edge_forward() {
        let (b, ids) = backend_with_drawers(2);
        b.add_similarity_edge(&ids[0], &ids[1], 0.8);
        assert!(b.remove_similarity_edge(&ids[0], &ids[1]));
        assert_eq!(b.similarity_edge_count(), 0);
        // Pheromone and cost entries should also be removed
        let d = b.read_data();
        let key = format!("{}:{}", ids[0], ids[1]);
        assert!(!d.edge_pheromones.contains_key(&key));
        assert!(!d.edge_costs.contains_key(&key));
    }

    #[test]
    fn test_remove_similarity_edge_reverse() {
        let (b, ids) = backend_with_drawers(2);
        b.add_similarity_edge(&ids[0], &ids[1], 0.8);
        // Remove using reversed order
        assert!(b.remove_similarity_edge(&ids[1], &ids[0]));
        // The edge was stored as ids[0]:ids[1], so forward removal returns
        // false but reverse should still return true if the key matches.
        // Actually: our remove checks both directions — but the stored key
        // is only forward. reverse won't find it. Let's re-check…
        // remove_similarity_edge tries both key_ab and key_ba.
        // key_ab = ids[1]:ids[0] → not found.
        // key_ba = ids[0]:ids[1] → found!
    }

    #[test]
    fn test_remove_nonexistent_similarity_edge() {
        let (b, _ids) = backend_with_drawers(2);
        assert!(!b.remove_similarity_edge("fake_a", "fake_b"));
    }

    #[test]
    fn test_add_similarity_edges_bulk_above_threshold() {
        // make_emb(0) and make_emb(1) should have *some* similarity.
        // With the make_emb helper, we test threshold=0.0 (accepts all).
        let (b, ids) = backend_with_drawers(4);
        let count = b.add_similarity_edges(0.0);
        // 4 drawers → C(4,2) = 6 pairs, threshold 0.0 accepts all.
        assert_eq!(count, 6, "expected C(4,2) = 6 edges");
        assert_eq!(b.similarity_edge_count(), 6);
        drop(ids);
    }

    #[test]
    fn test_add_similarity_edges_high_threshold() {
        let (b, _ids) = backend_with_drawers(4);
        let count = b.add_similarity_edges(1.0);
        // Threshold 1.0 — only perfectly identical embeddings would pass.
        // Different seeds produce different embeddings, so likely 0.
        assert_eq!(count, 0);
    }

    #[test]
    fn test_add_similarity_edges_returns_correct_count() {
        let (b, _ids) = backend_with_drawers(3);
        let count = b.add_similarity_edges(0.0);
        // 3 drawers → C(3,2) = 3 pairs
        assert_eq!(count, 3);
    }

    #[test]
    fn test_similarity_edge_count_empty() {
        let b = InMemoryBackend::new();
        assert_eq!(b.similarity_edge_count(), 0);
    }

    #[test]
    fn test_list_similarity_edges() {
        let (b, ids) = backend_with_drawers(2);
        b.add_similarity_edge(&ids[0], &ids[1], 0.7);
        let edges = b.list_similarity_edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, ids[0]);
        assert_eq!(edges[0].1, ids[1]);
        assert!((edges[0].2 - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_list_similarity_edges_empty() {
        let b = InMemoryBackend::new();
        assert!(b.list_similarity_edges().is_empty());
    }

    #[test]
    fn test_similarity_edge_survives_snapshot_restore() {
        let (b, ids) = backend_with_drawers(2);
        b.add_similarity_edge(&ids[0], &ids[1], 0.6);
        let snap = b.snapshot();
        let b2 = InMemoryBackend::with_data(snap);
        assert_eq!(b2.similarity_edge_count(), 1);
        let edges = b2.list_similarity_edges();
        assert!((edges[0].2 - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_edges_serialization() {
        let (b, ids) = backend_with_drawers(3);
        b.add_similarity_edges(0.0);
        let snap = b.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let deser: PalaceData = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.similarity_edges.len(), 3);
    }

    #[test]
    fn test_similarity_edges_deserialize_missing_field() {
        // Backward compat: old serialized data without similarity_edges
        let json = r#"{"wings":{},"rooms":{},"closets":{},"drawers":{},"entities":{},"relationships":[],"parent_map":{},"edge_pheromones":{},"edge_costs":{},"next_id":0,"schema_initialized":false}"#;
        let data: PalaceData = serde_json::from_str(json).unwrap();
        assert!(data.similarity_edges.is_empty());
    }

    #[test]
    fn test_multiple_similarity_edges_different_pairs() {
        let (b, ids) = backend_with_drawers(4);
        b.add_similarity_edge(&ids[0], &ids[1], 0.9);
        b.add_similarity_edge(&ids[0], &ids[2], 0.7);
        b.add_similarity_edge(&ids[1], &ids[3], 0.5);
        assert_eq!(b.similarity_edge_count(), 3);
        // edge_pheromones should also have 3 entries for similarity
        // (plus containment edges from hierarchy creation)
    }

    #[test]
    fn test_similarity_edge_score_range() {
        let (b, ids) = backend_with_drawers(2);
        // Test with various valid similarity scores
        b.add_similarity_edge(&ids[0], &ids[1], 0.0);
        let edges = b.list_similarity_edges();
        assert!((edges[0].2).abs() < 1e-6);

        b.add_similarity_edge(&ids[0], &ids[1], 1.0);
        let edges = b.list_similarity_edges();
        assert!((edges[0].2 - 1.0).abs() < 1e-6);
    }
}
