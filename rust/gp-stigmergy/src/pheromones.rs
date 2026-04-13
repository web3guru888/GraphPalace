//! Pheromone type definitions and accessors for node/edge pheromone fields.
//!
//! Five pheromone types from spec §4.1:
//! - **Exploitation** (node): "This location is valuable — come here"
//! - **Exploration** (node): "This location has been searched — try elsewhere"
//! - **Success** (edge): "This connection led to good outcomes"
//! - **Traversal** (edge): "This path is frequently used"
//! - **Recency** (edge): "This connection was used recently"

use gp_core::types::{EdgePheromones, NodePheromones};
use serde::{Deserialize, Serialize};

/// The five pheromone types used in GraphPalace stigmergic search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PheromoneType {
    /// Node pheromone: "This location is valuable — come here". Decay rate 0.02.
    Exploitation,
    /// Node pheromone: "This location has been searched — try elsewhere". Decay rate 0.05.
    Exploration,
    /// Edge pheromone: "This connection led to good outcomes". Decay rate 0.01.
    Success,
    /// Edge pheromone: "This path is frequently used". Decay rate 0.03.
    Traversal,
    /// Edge pheromone: "This connection was used recently". Decay rate 0.10.
    Recency,
}

impl PheromoneType {
    /// Returns the default decay rate for this pheromone type (from spec §4.1).
    pub fn default_decay_rate(&self) -> f64 {
        default_decay_rate(self)
    }

    /// Returns `true` if this pheromone type is stored on nodes.
    pub fn is_node_pheromone(&self) -> bool {
        matches!(self, PheromoneType::Exploitation | PheromoneType::Exploration)
    }

    /// Returns `true` if this pheromone type is stored on edges.
    pub fn is_edge_pheromone(&self) -> bool {
        matches!(
            self,
            PheromoneType::Success | PheromoneType::Traversal | PheromoneType::Recency
        )
    }

    /// All five pheromone types.
    pub fn all() -> &'static [PheromoneType; 5] {
        &[
            PheromoneType::Exploitation,
            PheromoneType::Exploration,
            PheromoneType::Success,
            PheromoneType::Traversal,
            PheromoneType::Recency,
        ]
    }
}

impl std::fmt::Display for PheromoneType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PheromoneType::Exploitation => write!(f, "exploitation"),
            PheromoneType::Exploration => write!(f, "exploration"),
            PheromoneType::Success => write!(f, "success"),
            PheromoneType::Traversal => write!(f, "traversal"),
            PheromoneType::Recency => write!(f, "recency"),
        }
    }
}

/// Returns the default decay rate for a pheromone type (from spec §4.1).
///
/// | Type          | Decay Rate | Half-life (cycles) |
/// |---------------|------------|--------------------|
/// | Exploitation  | 0.02       | ~35                |
/// | Exploration   | 0.05       | ~14                |
/// | Success       | 0.01       | ~69                |
/// | Traversal     | 0.03       | ~23                |
/// | Recency       | 0.10       | ~7                 |
pub fn default_decay_rate(ptype: &PheromoneType) -> f64 {
    match ptype {
        PheromoneType::Exploitation => 0.02,
        PheromoneType::Exploration => 0.05,
        PheromoneType::Success => 0.01,
        PheromoneType::Traversal => 0.03,
        PheromoneType::Recency => 0.10,
    }
}

// ─── Node pheromone accessors ─────────────────────────────────────────────

/// Get the value of a node pheromone by type.
///
/// # Panics
/// Panics if `ptype` is an edge pheromone type.
pub fn get_node_pheromone(pheromones: &NodePheromones, ptype: &PheromoneType) -> f64 {
    match ptype {
        PheromoneType::Exploitation => pheromones.exploitation,
        PheromoneType::Exploration => pheromones.exploration,
        _ => panic!(
            "Cannot read node pheromone for edge type {:?}",
            ptype
        ),
    }
}

/// Set the value of a node pheromone by type.
///
/// # Panics
/// Panics if `ptype` is an edge pheromone type.
pub fn set_node_pheromone(pheromones: &mut NodePheromones, ptype: &PheromoneType, value: f64) {
    match ptype {
        PheromoneType::Exploitation => pheromones.exploitation = value,
        PheromoneType::Exploration => pheromones.exploration = value,
        _ => panic!(
            "Cannot set node pheromone for edge type {:?}",
            ptype
        ),
    }
}

// ─── Edge pheromone accessors ─────────────────────────────────────────────

/// Get the value of an edge pheromone by type.
///
/// # Panics
/// Panics if `ptype` is a node pheromone type.
pub fn get_edge_pheromone(pheromones: &EdgePheromones, ptype: &PheromoneType) -> f64 {
    match ptype {
        PheromoneType::Success => pheromones.success,
        PheromoneType::Traversal => pheromones.traversal,
        PheromoneType::Recency => pheromones.recency,
        _ => panic!(
            "Cannot read edge pheromone for node type {:?}",
            ptype
        ),
    }
}

/// Set the value of an edge pheromone by type.
///
/// # Panics
/// Panics if `ptype` is a node pheromone type.
pub fn set_edge_pheromone(pheromones: &mut EdgePheromones, ptype: &PheromoneType, value: f64) {
    match ptype {
        PheromoneType::Success => pheromones.success = value,
        PheromoneType::Traversal => pheromones.traversal = value,
        PheromoneType::Recency => pheromones.recency = value,
        _ => panic!(
            "Cannot set edge pheromone for node type {:?}",
            ptype
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_rates_match_spec() {
        assert!((default_decay_rate(&PheromoneType::Exploitation) - 0.02).abs() < f64::EPSILON);
        assert!((default_decay_rate(&PheromoneType::Exploration) - 0.05).abs() < f64::EPSILON);
        assert!((default_decay_rate(&PheromoneType::Success) - 0.01).abs() < f64::EPSILON);
        assert!((default_decay_rate(&PheromoneType::Traversal) - 0.03).abs() < f64::EPSILON);
        assert!((default_decay_rate(&PheromoneType::Recency) - 0.10).abs() < f64::EPSILON);
    }

    #[test]
    fn test_method_decay_rate_matches_fn() {
        for ptype in PheromoneType::all() {
            assert_eq!(ptype.default_decay_rate(), default_decay_rate(ptype));
        }
    }

    #[test]
    fn test_is_node_pheromone() {
        assert!(PheromoneType::Exploitation.is_node_pheromone());
        assert!(PheromoneType::Exploration.is_node_pheromone());
        assert!(!PheromoneType::Success.is_node_pheromone());
        assert!(!PheromoneType::Traversal.is_node_pheromone());
        assert!(!PheromoneType::Recency.is_node_pheromone());
    }

    #[test]
    fn test_is_edge_pheromone() {
        assert!(!PheromoneType::Exploitation.is_edge_pheromone());
        assert!(!PheromoneType::Exploration.is_edge_pheromone());
        assert!(PheromoneType::Success.is_edge_pheromone());
        assert!(PheromoneType::Traversal.is_edge_pheromone());
        assert!(PheromoneType::Recency.is_edge_pheromone());
    }

    #[test]
    fn test_all_types_count() {
        assert_eq!(PheromoneType::all().len(), 5);
    }

    #[test]
    fn test_get_set_node_exploitation() {
        let mut p = NodePheromones::default();
        assert_eq!(get_node_pheromone(&p, &PheromoneType::Exploitation), 0.0);
        set_node_pheromone(&mut p, &PheromoneType::Exploitation, 0.75);
        assert!((get_node_pheromone(&p, &PheromoneType::Exploitation) - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_set_node_exploration() {
        let mut p = NodePheromones::default();
        set_node_pheromone(&mut p, &PheromoneType::Exploration, 0.42);
        assert!((p.exploration - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_set_edge_success() {
        let mut p = EdgePheromones::default();
        set_edge_pheromone(&mut p, &PheromoneType::Success, 0.9);
        assert!((get_edge_pheromone(&p, &PheromoneType::Success) - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_set_edge_traversal() {
        let mut p = EdgePheromones::default();
        set_edge_pheromone(&mut p, &PheromoneType::Traversal, 0.33);
        assert!((p.traversal - 0.33).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_set_edge_recency() {
        let mut p = EdgePheromones::default();
        set_edge_pheromone(&mut p, &PheromoneType::Recency, 1.0);
        assert!((get_edge_pheromone(&p, &PheromoneType::Recency) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "Cannot read node pheromone for edge type")]
    fn test_get_node_pheromone_with_edge_type_panics() {
        let p = NodePheromones::default();
        get_node_pheromone(&p, &PheromoneType::Success);
    }

    #[test]
    #[should_panic(expected = "Cannot set node pheromone for edge type")]
    fn test_set_node_pheromone_with_edge_type_panics() {
        let mut p = NodePheromones::default();
        set_node_pheromone(&mut p, &PheromoneType::Recency, 1.0);
    }

    #[test]
    #[should_panic(expected = "Cannot read edge pheromone for node type")]
    fn test_get_edge_pheromone_with_node_type_panics() {
        let p = EdgePheromones::default();
        get_edge_pheromone(&p, &PheromoneType::Exploitation);
    }

    #[test]
    #[should_panic(expected = "Cannot set edge pheromone for node type")]
    fn test_set_edge_pheromone_with_node_type_panics() {
        let mut p = EdgePheromones::default();
        set_edge_pheromone(&mut p, &PheromoneType::Exploration, 1.0);
    }

    #[test]
    fn test_display() {
        assert_eq!(PheromoneType::Exploitation.to_string(), "exploitation");
        assert_eq!(PheromoneType::Exploration.to_string(), "exploration");
        assert_eq!(PheromoneType::Success.to_string(), "success");
        assert_eq!(PheromoneType::Traversal.to_string(), "traversal");
        assert_eq!(PheromoneType::Recency.to_string(), "recency");
    }
}
