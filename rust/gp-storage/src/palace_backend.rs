//! The `PalaceBackend` trait — pluggable storage for GraphPalace.
//!
//! Abstracts all high-level palace operations so alternative backends
//! (e.g., LadybugDB disk storage) can be swapped in.

use gp_core::config::PheromoneConfig;
use gp_core::types::*;
use gp_core::Result;

use crate::memory::Relationship;

/// Comprehensive palace storage trait.
///
/// Covers all CRUD, search, pheromone, knowledge graph, similarity,
/// navigation, and agent operations needed by `GraphPalace`.
///
/// The bundled implementation is [`InMemoryBackend`](crate::InMemoryBackend).
pub trait PalaceBackend: Send + Sync {
    // ── Palace hierarchy CRUD ────────────────────────────────────────────

    fn create_wing(
        &self, name: &str, wing_type: WingType, description: &str, embedding: Embedding,
    ) -> Result<String>;

    fn get_wing(&self, id: &str) -> Result<Wing>;
    fn list_wings(&self) -> Vec<Wing>;
    fn find_wing_by_name(&self, name: &str) -> Option<Wing>;

    fn create_room(
        &self, wing_id: &str, name: &str, hall_type: HallType,
        description: &str, embedding: Embedding,
    ) -> Result<String>;

    fn get_room(&self, id: &str) -> Result<Room>;
    fn list_rooms(&self, wing_id: &str) -> Vec<Room>;

    fn create_closet(
        &self, room_id: &str, name: &str, summary: &str, embedding: Embedding,
    ) -> Result<String>;

    fn get_closet(&self, id: &str) -> Result<Closet>;

    fn create_drawer(
        &self, closet_id: &str, content: &str, embedding: Embedding,
        source: DrawerSource, source_file: Option<&str>, importance: f64,
    ) -> Result<String>;

    fn get_drawer(&self, id: &str) -> Result<Drawer>;
    fn delete_drawer(&self, id: &str) -> Result<()>;

    // ── Search ───────────────────────────────────────────────────────────

    fn search_drawers(
        &self, query_embedding: &Embedding, k: usize, threshold: f32,
    ) -> Vec<(Drawer, f32)>;

    fn rebuild_hnsw_index(&self);

    // ── Knowledge graph ──────────────────────────────────────────────────

    fn create_entity(
        &self, name: &str, entity_type: EntityType, description: &str, embedding: Embedding,
    ) -> Result<String>;

    fn get_entity(&self, id: &str) -> Result<Entity>;
    fn find_entity_by_name(&self, name: &str) -> Option<Entity>;

    fn add_relationship(
        &self, subject: &str, predicate: &str, object: &str, confidence: f64,
    ) -> Result<String>;

    fn add_relationship_temporal(
        &self, subject: &str, predicate: &str, object: &str, confidence: f64,
        valid_from: Option<String>, valid_to: Option<String>,
        statement_type: StatementType,
    ) -> Result<String>;

    fn query_relationships(&self, entity: &str) -> Vec<Relationship>;
    fn invalidate_relationship(&self, subject: &str, predicate: &str, object: &str) -> bool;
    fn find_contradictions(&self, entity: &str) -> Vec<(Relationship, Relationship)>;

    // ── Pheromones ───────────────────────────────────────────────────────

    fn deposit_edge_pheromones(
        &self, from: &str, to: &str, success: f64, traversal: f64, recency: f64,
    );

    fn deposit_node_pheromones(
        &self, node_id: &str, exploitation: f64, exploration: f64,
    );

    fn deposit_failure_edge_pheromones(&self, from: &str, to: &str, penalty: f64);
    fn deposit_failure_node_pheromones(&self, node_id: &str, exploitation_penalty: f64);

    fn decay_all_pheromones(&self, config: &PheromoneConfig);
    fn hot_paths(&self, k: usize) -> Vec<(String, String, f64)>;
    fn cold_spots(&self, k: usize) -> Vec<(String, String, f64)>;
    fn total_pheromone_mass(&self) -> f64;

    // ── Similarity graph ─────────────────────────────────────────────────

    fn add_similarity_edge(&self, drawer_a: &str, drawer_b: &str, similarity: f32);
    fn remove_similarity_edge(&self, drawer_a: &str, drawer_b: &str) -> bool;
    fn add_similarity_edges(&self, threshold: f32) -> usize;
    fn similarity_edge_count(&self) -> usize;
    fn list_similarity_edges(&self) -> Vec<(String, String, f32)>;

    // ── Navigation structures ────────────────────────────────────────────

    fn create_hall(&self, from_room_id: &str, to_room_id: &str, wing_id: &str);
    fn create_tunnel(&self, from_room_id: &str, to_room_id: &str);
    fn list_halls(&self) -> Vec<(String, String, String)>;
    fn list_tunnels(&self) -> Vec<(String, String)>;

    // ── Agents ───────────────────────────────────────────────────────────

    fn create_agent(
        &self, name: &str, domain: &str, focus: &str,
        goal_embedding: Embedding, temperature: f64,
    ) -> Result<String>;

    fn get_agent(&self, id: &str) -> Result<Agent>;
    fn list_agents(&self) -> Vec<Agent>;
    fn find_agent_by_name(&self, name: &str) -> Option<Agent>;
    fn append_diary(&self, agent_id: &str, entry: &str) -> Result<()>;
    fn agent_count(&self) -> usize;

    // ── Counts ───────────────────────────────────────────────────────────

    fn wing_count(&self) -> usize;
    fn room_count(&self) -> usize;
    fn closet_count(&self) -> usize;
    fn drawer_count(&self) -> usize;
    fn entity_count(&self) -> usize;
    fn relationship_count(&self) -> usize;
    fn edge_count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemoryBackend;

    fn zero_emb() -> Embedding { [0.0f32; 384] }

    #[test]
    fn test_inmemory_implements_palace_backend() {
        let backend: Box<dyn PalaceBackend> = Box::new(InMemoryBackend::new());
        let id = backend.create_wing("Test", WingType::Domain, "desc", zero_emb()).unwrap();
        assert!(id.starts_with("wing_"));
        assert_eq!(backend.wing_count(), 1);
    }

    #[test]
    fn test_palace_backend_full_hierarchy() {
        let backend: Box<dyn PalaceBackend> = Box::new(InMemoryBackend::new());
        let wid = backend.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = backend.create_room(&wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = backend.create_closet(&rid, "C", "S", zero_emb()).unwrap();
        let did = backend.create_drawer(&cid, "content", zero_emb(), DrawerSource::Conversation, None, 0.5).unwrap();
        assert_eq!(backend.wing_count(), 1);
        assert_eq!(backend.room_count(), 1);
        assert_eq!(backend.closet_count(), 1);
        assert_eq!(backend.drawer_count(), 1);
        let drawer = backend.get_drawer(&did).unwrap();
        assert_eq!(drawer.content, "content");
    }

    #[test]
    fn test_palace_backend_pheromone_ops() {
        let backend: Box<dyn PalaceBackend> = Box::new(InMemoryBackend::new());
        let wid = backend.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        backend.deposit_node_pheromones(&wid, 1.0, 0.5);
        assert!(backend.total_pheromone_mass() > 0.0);
    }

    #[test]
    fn test_palace_backend_failure_deposits() {
        let backend: Box<dyn PalaceBackend> = Box::new(InMemoryBackend::new());
        let wid = backend.create_wing("W", WingType::Domain, "", zero_emb()).unwrap();
        backend.deposit_node_pheromones(&wid, 1.0, 0.0);
        backend.deposit_failure_node_pheromones(&wid, 0.5);
        // Exploitation should be reduced
        let wing = backend.get_wing(&wid).unwrap();
        assert!((wing.pheromones.exploitation - 0.5).abs() < 1e-12);
    }
}
