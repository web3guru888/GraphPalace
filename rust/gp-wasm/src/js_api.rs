//! JavaScript-facing API for GraphPalace WASM.
//!
//! Provides a `GraphPalaceWasm` struct with methods for all core palace
//! operations. In WASM builds, these methods are exposed via wasm-bindgen.
//! In native builds, they serve as the integration layer for testing.
//!
//! The actual wasm-bindgen annotations are gated behind `#[cfg(target_arch = "wasm32")]`
//! since wasm-bindgen can't compile for native targets.

use serde::{Deserialize, Serialize};

use crate::palace::InMemoryPalace;
use gp_core::types::{DrawerSource, HallType, WingType};

/// Search result returned to JavaScript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsSearchResult {
    /// Drawer ID.
    pub id: String,
    /// Drawer content.
    pub content: String,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f64,
    /// Source of the drawer.
    pub source: String,
}

/// Path result returned to JavaScript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsPathResult {
    /// List of node IDs in the path.
    pub path: Vec<String>,
    /// Total cost of the path.
    pub total_cost: f64,
    /// Number of hops.
    pub hops: usize,
}

/// Pheromone status for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsPheromoneStatus {
    /// Node ID.
    pub node_id: String,
    /// Exploitation pheromone level.
    pub exploitation: f64,
    /// Exploration pheromone level.
    pub exploration: f64,
}

/// Palace overview statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsPalaceOverview {
    pub name: String,
    pub wing_count: usize,
    pub room_count: usize,
    pub closet_count: usize,
    pub drawer_count: usize,
    pub entity_count: usize,
    pub agent_count: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Wing info for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsWingInfo {
    pub id: String,
    pub name: String,
    pub wing_type: String,
    pub description: String,
    pub room_count: usize,
}

/// Room info for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsRoomInfo {
    pub id: String,
    pub name: String,
    pub hall_type: String,
}

/// The main WASM API entry point.
///
/// In browser builds, this struct is exported via wasm-bindgen.
/// Methods accept and return JSON strings for cross-boundary compatibility.
pub struct GraphPalaceWasm {
    palace: InMemoryPalace,
}

impl GraphPalaceWasm {
    /// Create a new palace with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            palace: InMemoryPalace::new(name),
        }
    }

    /// Get palace overview as JSON.
    pub fn get_overview(&self) -> String {
        let overview = JsPalaceOverview {
            name: self.palace.palace.name.clone(),
            wing_count: self.palace.wings.len(),
            room_count: self.palace.rooms.len(),
            closet_count: self.palace.closets.len(),
            drawer_count: self.palace.drawers.len(),
            entity_count: self.palace.entities.len(),
            agent_count: self.palace.agents.len(),
            total_nodes: self.palace.node_count(),
            total_edges: self.palace.edge_count(),
        };
        serde_json::to_string(&overview).unwrap_or_default()
    }

    /// Add a wing. Returns the wing ID.
    pub fn add_wing(&mut self, name: &str, wing_type: &str, description: &str) -> String {
        let wt = match wing_type {
            "person" => WingType::Person,
            "project" => WingType::Project,
            "domain" => WingType::Domain,
            "topic" => WingType::Topic,
            _ => WingType::Topic,
        };
        self.palace.add_wing(name, wt, description)
    }

    /// List all wings as JSON.
    pub fn list_wings(&self) -> String {
        let wings: Vec<JsWingInfo> = self
            .palace
            .wings
            .values()
            .map(|w| {
                let room_count = self.palace.list_rooms(&w.id).len();
                JsWingInfo {
                    id: w.id.clone(),
                    name: w.name.clone(),
                    wing_type: w.wing_type.to_string(),
                    description: w.description.clone(),
                    room_count,
                }
            })
            .collect();
        serde_json::to_string(&wings).unwrap_or_default()
    }

    /// Add a room to a wing. Returns room ID or empty string on failure.
    pub fn add_room(&mut self, wing_id: &str, name: &str, hall_type: &str) -> String {
        let ht = match hall_type {
            "facts" => HallType::Facts,
            "events" => HallType::Events,
            "discoveries" => HallType::Discoveries,
            "preferences" => HallType::Preferences,
            "advice" => HallType::Advice,
            _ => HallType::Facts,
        };
        self.palace
            .add_room(wing_id, name, ht)
            .unwrap_or_default()
    }

    /// List rooms in a wing as JSON.
    pub fn list_rooms(&self, wing_id: &str) -> String {
        let rooms: Vec<JsRoomInfo> = self
            .palace
            .list_rooms(wing_id)
            .iter()
            .map(|r| JsRoomInfo {
                id: r.id.clone(),
                name: r.name.clone(),
                hall_type: r.hall_type.to_string(),
            })
            .collect();
        serde_json::to_string(&rooms).unwrap_or_default()
    }

    /// Add a closet to a room. Returns closet ID or empty string on failure.
    pub fn add_closet(&mut self, room_id: &str, name: &str, summary: &str) -> String {
        self.palace
            .add_closet(room_id, name, summary)
            .unwrap_or_default()
    }

    /// Add a drawer. Returns drawer ID or empty string on failure.
    pub fn add_drawer(&mut self, closet_id: &str, content: &str, source: &str) -> String {
        let src = match source {
            "conversation" => DrawerSource::Conversation,
            "file" => DrawerSource::File,
            "api" => DrawerSource::Api,
            "agent" => DrawerSource::Agent,
            _ => DrawerSource::Api,
        };
        self.palace
            .add_drawer(closet_id, content, src)
            .unwrap_or_default()
    }

    /// Delete a drawer. Returns true if deleted.
    pub fn delete_drawer(&mut self, drawer_id: &str) -> bool {
        self.palace.delete_drawer(drawer_id)
    }

    /// Get a drawer's content as JSON.
    pub fn get_drawer(&self, drawer_id: &str) -> String {
        match self.palace.get_drawer(drawer_id) {
            Some(d) => serde_json::to_string(d).unwrap_or_default(),
            None => "null".to_string(),
        }
    }

    /// Deposit success pheromones along a path (JSON array of node IDs).
    pub fn deposit_pheromones(&mut self, path_json: &str, base_reward: f64) {
        if let Ok(path) = serde_json::from_str::<Vec<String>>(path_json) {
            self.palace.deposit_success_on_path(&path, base_reward);
        }
    }

    /// Get pheromone status for a node as JSON.
    pub fn get_pheromone_status(&self, node_id: &str) -> String {
        let ph = self
            .palace
            .wings
            .get(node_id)
            .map(|w| &w.pheromones)
            .or_else(|| self.palace.rooms.get(node_id).map(|r| &r.pheromones))
            .or_else(|| self.palace.closets.get(node_id).map(|c| &c.pheromones))
            .or_else(|| self.palace.drawers.get(node_id).map(|d| &d.pheromones));

        match ph {
            Some(p) => {
                let status = JsPheromoneStatus {
                    node_id: node_id.to_string(),
                    exploitation: p.exploitation,
                    exploration: p.exploration,
                };
                serde_json::to_string(&status).unwrap_or_default()
            }
            None => "null".to_string(),
        }
    }

    /// Force a pheromone decay cycle.
    pub fn decay_pheromones(&mut self) {
        self.palace.decay_all();
    }

    /// Recompute all edge costs.
    pub fn recompute_costs(&mut self) {
        self.palace.recompute_all_costs();
    }

    /// Export the entire palace as JSON.
    pub fn export_json(&self) -> String {
        self.palace.export_json().unwrap_or_default()
    }

    /// Import a palace from JSON. Returns true on success.
    pub fn import_json(&mut self, json: &str) -> bool {
        match InMemoryPalace::import_json(json) {
            Ok(palace) => {
                self.palace = palace;
                true
            }
            Err(_) => false,
        }
    }

    /// Get the node count.
    pub fn node_count(&self) -> usize {
        self.palace.node_count()
    }

    /// Get the edge count.
    pub fn edge_count(&self) -> usize {
        self.palace.edge_count()
    }

    /// Get access to the underlying palace (for advanced usage).
    pub fn palace(&self) -> &InMemoryPalace {
        &self.palace
    }

    /// Get mutable access to the underlying palace (for advanced usage).
    pub fn palace_mut(&mut self) -> &mut InMemoryPalace {
        &mut self.palace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_create_palace() {
        let palace = GraphPalaceWasm::new("Test Palace");
        assert_eq!(palace.node_count(), 1);
        assert_eq!(palace.edge_count(), 0);
    }

    #[test]
    fn wasm_overview() {
        let palace = GraphPalaceWasm::new("Test");
        let json = palace.get_overview();
        let overview: JsPalaceOverview = serde_json::from_str(&json).unwrap();
        assert_eq!(overview.name, "Test");
        assert_eq!(overview.total_nodes, 1);
    }

    #[test]
    fn wasm_add_wing() {
        let mut palace = GraphPalaceWasm::new("Test");
        let id = palace.add_wing("Research", "domain", "Research wing");
        assert!(!id.is_empty());
        assert_eq!(palace.node_count(), 2);
    }

    #[test]
    fn wasm_list_wings() {
        let mut palace = GraphPalaceWasm::new("Test");
        palace.add_wing("A", "domain", "");
        palace.add_wing("B", "project", "");
        let json = palace.list_wings();
        let wings: Vec<JsWingInfo> = serde_json::from_str(&json).unwrap();
        assert_eq!(wings.len(), 2);
    }

    #[test]
    fn wasm_full_hierarchy() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        let room = palace.add_room(&wing, "R", "facts");
        assert!(!room.is_empty());
        let closet = palace.add_closet(&room, "C", "Summary");
        assert!(!closet.is_empty());
        let drawer = palace.add_drawer(&closet, "Content here", "conversation");
        assert!(!drawer.is_empty());
        assert_eq!(palace.node_count(), 5); // palace + wing + room + closet + drawer
    }

    #[test]
    fn wasm_add_room_to_invalid_wing() {
        let mut palace = GraphPalaceWasm::new("Test");
        let room = palace.add_room("nonexistent", "R", "facts");
        assert!(room.is_empty());
    }

    #[test]
    fn wasm_delete_drawer() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        let room = palace.add_room(&wing, "R", "facts");
        let closet = palace.add_closet(&room, "C", "");
        let drawer = palace.add_drawer(&closet, "To delete", "api");
        assert!(palace.delete_drawer(&drawer));
        assert_eq!(palace.get_drawer(&drawer), "null");
    }

    #[test]
    fn wasm_get_drawer() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        let room = palace.add_room(&wing, "R", "facts");
        let closet = palace.add_closet(&room, "C", "");
        let drawer = palace.add_drawer(&closet, "Test content", "file");
        let json = palace.get_drawer(&drawer);
        assert!(json.contains("Test content"));
    }

    #[test]
    fn wasm_pheromones() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        let room = palace.add_room(&wing, "R", "facts");

        // Deposit pheromones
        let path = serde_json::to_string(&vec![&wing, &room]).unwrap();
        palace.deposit_pheromones(&path, 1.0);

        // Check status
        let status = palace.get_pheromone_status(&wing);
        assert!(!status.contains("null"));
        let ps: JsPheromoneStatus = serde_json::from_str(&status).unwrap();
        assert!(ps.exploitation > 0.0);
    }

    #[test]
    fn wasm_decay() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        let room = palace.add_room(&wing, "R", "facts");
        let path = serde_json::to_string(&vec![&wing, &room]).unwrap();
        palace.deposit_pheromones(&path, 1.0);

        let before = palace.get_pheromone_status(&wing);
        palace.decay_pheromones();
        let after = palace.get_pheromone_status(&wing);

        let b: JsPheromoneStatus = serde_json::from_str(&before).unwrap();
        let a: JsPheromoneStatus = serde_json::from_str(&after).unwrap();
        assert!(a.exploitation <= b.exploitation);
    }

    #[test]
    fn wasm_export_import() {
        let mut palace = GraphPalaceWasm::new("Export Test");
        palace.add_wing("W", "domain", "Test");

        let json = palace.export_json();
        assert!(!json.is_empty());

        let mut palace2 = GraphPalaceWasm::new("Empty");
        assert!(palace2.import_json(&json));
        assert_eq!(palace2.node_count(), 2); // palace + wing
    }

    #[test]
    fn wasm_import_invalid_json() {
        let mut palace = GraphPalaceWasm::new("Test");
        assert!(!palace.import_json("not json"));
    }

    #[test]
    fn wasm_list_rooms() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        palace.add_room(&wing, "R1", "facts");
        palace.add_room(&wing, "R2", "events");
        let json = palace.list_rooms(&wing);
        let rooms: Vec<JsRoomInfo> = serde_json::from_str(&json).unwrap();
        assert_eq!(rooms.len(), 2);
    }

    #[test]
    fn wasm_nonexistent_pheromone_status() {
        let palace = GraphPalaceWasm::new("Test");
        assert_eq!(palace.get_pheromone_status("nonexistent"), "null");
    }

    #[test]
    fn wasm_deposit_invalid_json() {
        let mut palace = GraphPalaceWasm::new("Test");
        palace.deposit_pheromones("not json", 1.0); // Should not panic
    }

    #[test]
    fn wasm_recompute_costs() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        let room = palace.add_room(&wing, "R", "facts");
        let path = serde_json::to_string(&vec![&wing, &room]).unwrap();
        palace.deposit_pheromones(&path, 1.0);
        palace.recompute_costs(); // Should not panic
    }

    #[test]
    fn wasm_wing_type_default() {
        let mut palace = GraphPalaceWasm::new("Test");
        let id = palace.add_wing("W", "unknown_type", "");
        assert!(!id.is_empty()); // Defaults to Topic
    }

    #[test]
    fn wasm_hall_type_default() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        let room = palace.add_room(&wing, "R", "unknown_type");
        assert!(!room.is_empty()); // Defaults to Facts
    }

    #[test]
    fn wasm_drawer_source_default() {
        let mut palace = GraphPalaceWasm::new("Test");
        let wing = palace.add_wing("W", "domain", "");
        let room = palace.add_room(&wing, "R", "facts");
        let closet = palace.add_closet(&room, "C", "");
        let drawer = palace.add_drawer(&closet, "Content", "unknown_source");
        assert!(!drawer.is_empty()); // Defaults to Api
    }

    #[test]
    fn wasm_palace_accessor() {
        let palace = GraphPalaceWasm::new("Test");
        assert_eq!(palace.palace().palace.name, "Test");
    }

    #[test]
    fn wasm_overview_after_population() {
        let mut palace = GraphPalaceWasm::new("Full");
        let w = palace.add_wing("W", "domain", "");
        let r = palace.add_room(&w, "R", "facts");
        let c = palace.add_closet(&r, "C", "");
        palace.add_drawer(&c, "D1", "api");
        palace.add_drawer(&c, "D2", "api");

        let json = palace.get_overview();
        let ov: JsPalaceOverview = serde_json::from_str(&json).unwrap();
        assert_eq!(ov.wing_count, 1);
        assert_eq!(ov.room_count, 1);
        assert_eq!(ov.closet_count, 1);
        assert_eq!(ov.drawer_count, 2);
        assert_eq!(ov.total_nodes, 6);
        assert_eq!(ov.total_edges, 5);
    }
}
