//! Path provenance tracking for Semantic A* results.
//!
//! Every pathfinding result carries full provenance: per-step costs,
//! heuristic values, and iteration counts — enabling interpretable
//! routing decisions.

use serde::{Deserialize, Serialize};

/// A single step in the path provenance trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceStep {
    /// ID of the node at this step.
    pub node_id: String,
    /// Relation type of the edge leading to this node (empty for start node).
    pub edge_type: String,
    /// Accumulated cost from start to this node.
    pub g_cost: f64,
    /// Heuristic estimate from this node to goal.
    pub h_cost: f64,
    /// Total estimated cost: g + h.
    pub f_cost: f64,
}

/// Complete result of a pathfinding query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResult {
    /// Ordered node IDs from start to goal.
    pub path: Vec<String>,
    /// Edge relation types along the path (length = path.len() - 1).
    pub edges: Vec<String>,
    /// Total accumulated cost from start to goal.
    pub total_cost: f64,
    /// Number of main-loop iterations.
    pub iterations: usize,
    /// Number of nodes expanded (popped from open set).
    pub nodes_expanded: usize,
    /// Per-step provenance trace.
    pub provenance: Vec<ProvenanceStep>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_step_serialization() {
        let step = ProvenanceStep {
            node_id: "room_1".into(),
            edge_type: "HALL".into(),
            g_cost: 0.5,
            h_cost: 0.3,
            f_cost: 0.8,
        };
        let json = serde_json::to_string(&step).unwrap();
        let deser: ProvenanceStep = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.node_id, "room_1");
        assert!((deser.f_cost - 0.8).abs() < 1e-10);
    }

    #[test]
    fn path_result_serialization() {
        let result = PathResult {
            path: vec!["A".into(), "B".into(), "C".into()],
            edges: vec!["HALL".into(), "TUNNEL".into()],
            total_cost: 1.2,
            iterations: 5,
            nodes_expanded: 3,
            provenance: vec![
                ProvenanceStep {
                    node_id: "A".into(),
                    edge_type: String::new(),
                    g_cost: 0.0,
                    h_cost: 1.0,
                    f_cost: 1.0,
                },
                ProvenanceStep {
                    node_id: "B".into(),
                    edge_type: "HALL".into(),
                    g_cost: 0.5,
                    h_cost: 0.5,
                    f_cost: 1.0,
                },
                ProvenanceStep {
                    node_id: "C".into(),
                    edge_type: "TUNNEL".into(),
                    g_cost: 1.2,
                    h_cost: 0.0,
                    f_cost: 1.2,
                },
            ],
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: PathResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.path.len(), 3);
        assert_eq!(deser.edges.len(), 2);
        assert_eq!(deser.nodes_expanded, 3);
    }

    #[test]
    fn path_result_edges_length_invariant() {
        let result = PathResult {
            path: vec!["X".into(), "Y".into()],
            edges: vec!["HAS_ROOM".into()],
            total_cost: 0.3,
            iterations: 1,
            nodes_expanded: 1,
            provenance: vec![],
        };
        assert_eq!(result.edges.len(), result.path.len() - 1);
    }
}
