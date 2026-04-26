//! Palace status, hot-path, and cold-spot types.

use chrono::{DateTime, Utc};
use gp_core::types::StatementType;
use serde::{Deserialize, Serialize};

/// Snapshot of the palace's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceStatus {
    /// Name of the palace.
    pub name: String,
    /// Number of wings.
    pub wing_count: usize,
    /// Number of rooms.
    pub room_count: usize,
    /// Number of closets.
    pub closet_count: usize,
    /// Number of drawers (atomic memories).
    pub drawer_count: usize,
    /// Number of knowledge-graph entities.
    pub entity_count: usize,
    /// Number of knowledge-graph relationships.
    pub relationship_count: usize,
    /// Total pheromone mass across all nodes and edges.
    pub total_pheromone_mass: f64,
    /// When pheromones were last decayed (None if never).
    pub last_decay_time: Option<DateTime<Utc>>,
}

/// An edge with high success pheromone — a frequently rewarded path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPath {
    /// Source node ID.
    pub from_id: String,
    /// Target node ID.
    pub to_id: String,
    /// Accumulated success pheromone.
    pub success_pheromone: f64,
}

/// A node with low total pheromone — under-explored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdSpot {
    /// Node ID.
    pub node_id: String,
    /// Human-readable name or content snippet.
    pub name: String,
    /// Combined exploitation + exploration pheromone.
    pub total_pheromone: f64,
}

/// A relationship in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgRelationship {
    /// Subject entity name or ID.
    pub subject: String,
    /// Predicate (relationship label).
    pub predicate: String,
    /// Object entity name or ID.
    pub object: String,
    /// Confidence score in [0, 1].
    pub confidence: f64,
    /// Start of the assertion validity window.
    #[serde(default)]
    pub valid_from: Option<String>,
    /// End of the assertion validity window.
    #[serde(default)]
    pub valid_to: Option<String>,
    /// When the triple was first recorded.
    #[serde(default)]
    pub observed_at: Option<String>,
    /// When the triple was retracted/invalidated in the system.
    #[serde(default)]
    pub invalidated_at: Option<String>,
    /// Classification: fact, observation, inference, hypothesis.
    #[serde(default)]
    pub statement_type: StatementType,
}

// ---------------------------------------------------------------------------
// Hyperstructure lifecycle (NE/E tracking)
// ---------------------------------------------------------------------------

/// Phase classification for a hyperstructure node.
///
/// Based on Norris (2011): NE hyperstructures are dynamically maintained
/// by active processes; E hyperstructures are stable self-assembled states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HyperstructurePhase {
    /// Actively maintained by ongoing pheromone deposits. Dissolves if activity ceases.
    NonEquilibrium,
    /// Stable, self-assembled. Low activity but persistent structure.
    Equilibrium,
    /// Between states — moderate activity, not yet classified.
    Transitioning,
}

impl std::fmt::Display for HyperstructurePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonEquilibrium => write!(f, "non-equilibrium"),
            Self::Equilibrium => write!(f, "equilibrium"),
            Self::Transitioning => write!(f, "transitioning"),
        }
    }
}

/// Lifecycle metrics for a palace node (wing, room, closet, or drawer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperstructureMetrics {
    /// Node ID.
    pub node_id: String,
    /// Human-readable label.
    pub label: String,
    /// Node type: "wing", "room", "closet", "drawer", "entity".
    pub node_type: String,
    /// Non-equilibrium score: sum of exploitation + exploration pheromones.
    /// High values indicate active maintenance.
    pub ne_score: f64,
    /// Equilibrium score: structural connectivity (child count / edge count).
    /// High values indicate stable structure.
    pub e_score: f64,
    /// NE/E ratio. > 1.0 = active, < 1.0 = stable, ≈ 1.0 = transitioning.
    pub ne_e_ratio: f64,
    /// Classified phase based on NE/E ratio thresholds.
    pub phase: HyperstructurePhase,
}

/// NE/E ratio threshold for NonEquilibrium classification.
pub const NE_THRESHOLD: f64 = 1.5;
/// NE/E ratio threshold for Equilibrium classification.
pub const E_THRESHOLD: f64 = 0.5;

/// Classify a node's hyperstructure phase from its NE/E ratio.
pub fn classify_phase(ne_e_ratio: f64) -> HyperstructurePhase {
    if ne_e_ratio >= NE_THRESHOLD {
        HyperstructurePhase::NonEquilibrium
    } else if ne_e_ratio <= E_THRESHOLD {
        HyperstructurePhase::Equilibrium
    } else {
        HyperstructurePhase::Transitioning
    }
}

/// Summary of hyperstructure phase distribution across the palace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleSummary {
    /// Total NE nodes.
    pub ne_count: usize,
    /// Total E nodes.
    pub e_count: usize,
    /// Total transitioning nodes.
    pub transitioning_count: usize,
    /// Palace-wide NE/E ratio (ne_count / max(e_count, 1)).
    pub global_ne_e_ratio: f64,
    /// Per-node metrics (all nodes).
    pub nodes: Vec<HyperstructureMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palace_status_serialization() {
        let status = PalaceStatus {
            name: "Test Palace".into(),
            wing_count: 3,
            room_count: 10,
            closet_count: 20,
            drawer_count: 50,
            entity_count: 15,
            relationship_count: 8,
            total_pheromone_mass: 42.5,
            last_decay_time: Some(Utc::now()),
        };
        let json = serde_json::to_string(&status).unwrap();
        let deser: PalaceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.name, "Test Palace");
        assert_eq!(deser.wing_count, 3);
        assert_eq!(deser.drawer_count, 50);
    }

    #[test]
    fn palace_status_no_decay_time() {
        let status = PalaceStatus {
            name: "New Palace".into(),
            wing_count: 0,
            room_count: 0,
            closet_count: 0,
            drawer_count: 0,
            entity_count: 0,
            relationship_count: 0,
            total_pheromone_mass: 0.0,
            last_decay_time: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("null"));
    }

    #[test]
    fn hot_path_serialization() {
        let hp = HotPath {
            from_id: "room_1".into(),
            to_id: "room_2".into(),
            success_pheromone: 3.15,
        };
        let json = serde_json::to_string(&hp).unwrap();
        let deser: HotPath = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.from_id, "room_1");
        assert!((deser.success_pheromone - 3.15).abs() < 1e-10);
    }

    #[test]
    fn cold_spot_serialization() {
        let cs = ColdSpot {
            node_id: "drawer_5".into(),
            name: "some memory".into(),
            total_pheromone: 0.001,
        };
        let json = serde_json::to_string(&cs).unwrap();
        let deser: ColdSpot = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.node_id, "drawer_5");
    }

    #[test]
    fn kg_relationship_serialization() {
        let rel = KgRelationship {
            subject: "Einstein".into(),
            predicate: "discovered".into(),
            object: "Relativity".into(),
            confidence: 0.99,
            valid_from: None,
            valid_to: None,
            observed_at: Some("2026-01-01T00:00:00+00:00".into()),
            invalidated_at: None,
            statement_type: StatementType::Fact,
        };
        let json = serde_json::to_string(&rel).unwrap();
        let deser: KgRelationship = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.predicate, "discovered");
        assert!((deser.confidence - 0.99).abs() < 1e-10);
        assert_eq!(deser.statement_type, StatementType::Fact);
        assert!(deser.invalidated_at.is_none());
    }

    #[test]
    fn kg_relationship_backward_compat_deserialization() {
        // Old JSON without temporal fields should still deserialize (serde defaults).
        let json = r#"{"subject":"A","predicate":"knows","object":"B","confidence":0.5}"#;
        let deser: KgRelationship = serde_json::from_str(json).unwrap();
        assert_eq!(deser.subject, "A");
        assert_eq!(deser.statement_type, StatementType::Fact);
        assert!(deser.valid_from.is_none());
        assert!(deser.invalidated_at.is_none());
    }

    #[test]
    fn kg_relationship_hypothesis_round_trip() {
        let rel = KgRelationship {
            subject: "X".into(),
            predicate: "causes".into(),
            object: "Y".into(),
            confidence: 0.3,
            valid_from: Some("2026-01-01T00:00:00+00:00".into()),
            valid_to: Some("2026-12-31T23:59:59+00:00".into()),
            observed_at: Some("2026-04-14T00:00:00+00:00".into()),
            invalidated_at: None,
            statement_type: StatementType::Hypothesis,
        };
        let json = serde_json::to_string(&rel).unwrap();
        assert!(json.contains("hypothesis"));
        let deser: KgRelationship = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.statement_type, StatementType::Hypothesis);
        assert!(deser.valid_from.is_some());
    }

    // ─── Hyperstructure lifecycle ─────────────────────────────────────────

    #[test]
    fn test_classify_phase_ne() {
        assert_eq!(classify_phase(2.0), HyperstructurePhase::NonEquilibrium);
        assert_eq!(classify_phase(1.5), HyperstructurePhase::NonEquilibrium);
    }

    #[test]
    fn test_classify_phase_e() {
        assert_eq!(classify_phase(0.3), HyperstructurePhase::Equilibrium);
        assert_eq!(classify_phase(0.0), HyperstructurePhase::Equilibrium);
        assert_eq!(classify_phase(0.5), HyperstructurePhase::Equilibrium);
    }

    #[test]
    fn test_classify_phase_transitioning() {
        assert_eq!(classify_phase(1.0), HyperstructurePhase::Transitioning);
        assert_eq!(classify_phase(0.8), HyperstructurePhase::Transitioning);
        assert_eq!(classify_phase(1.4), HyperstructurePhase::Transitioning);
    }

    #[test]
    fn test_hyperstructure_metrics_serialization() {
        let m = HyperstructureMetrics {
            node_id: "wing_1".into(),
            label: "Science".into(),
            node_type: "wing".into(),
            ne_score: 2.5,
            e_score: 3.0,
            ne_e_ratio: 0.833,
            phase: HyperstructurePhase::Transitioning,
        };
        let json = serde_json::to_string(&m).unwrap();
        let deser: HyperstructureMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.node_id, "wing_1");
        assert_eq!(deser.phase, HyperstructurePhase::Transitioning);
    }

    #[test]
    fn test_lifecycle_summary_serialization() {
        let s = LifecycleSummary {
            ne_count: 5,
            e_count: 10,
            transitioning_count: 3,
            global_ne_e_ratio: 0.5,
            nodes: vec![],
        };
        let json = serde_json::to_string(&s).unwrap();
        let deser: LifecycleSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.ne_count, 5);
    }

    #[test]
    fn test_hyperstructure_phase_display() {
        assert_eq!(HyperstructurePhase::NonEquilibrium.to_string(), "non-equilibrium");
        assert_eq!(HyperstructurePhase::Equilibrium.to_string(), "equilibrium");
        assert_eq!(HyperstructurePhase::Transitioning.to_string(), "transitioning");
    }
}
