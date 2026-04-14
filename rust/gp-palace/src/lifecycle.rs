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
}
