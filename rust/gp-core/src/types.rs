//! Core node and edge types for the GraphPalace schema.
//!
//! Maps directly to the Cypher DDL in §3 of the spec.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::config::EMBEDDING_DIM;

/// 384-dimensional embedding vector (all-MiniLM-L6-v2).
pub type Embedding = [f32; EMBEDDING_DIM];

/// Create a zero embedding.
pub fn zero_embedding() -> Embedding {
    [0.0f32; EMBEDDING_DIM]
}

// ─── Pheromone Fields ──────────────────────────────────────────────────────

/// Node-level pheromone fields (exploitation + exploration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePheromones {
    /// "This location is valuable — come here"
    pub exploitation: f64,
    /// "This location has been searched — try elsewhere"
    pub exploration: f64,
}

impl Default for NodePheromones {
    fn default() -> Self {
        Self {
            exploitation: 0.0,
            exploration: 0.0,
        }
    }
}

/// Edge-level pheromone fields (success + traversal + recency).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgePheromones {
    /// "This connection led to good outcomes"
    pub success: f64,
    /// "This path is frequently used"
    pub traversal: f64,
    /// "This connection was used recently"
    pub recency: f64,
}

impl Default for EdgePheromones {
    fn default() -> Self {
        Self {
            success: 0.0,
            traversal: 0.0,
            recency: 0.0,
        }
    }
}

/// Common edge cost fields shared by all relationship types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCost {
    /// Base cost for this edge type (from relation weight table).
    pub base_cost: f64,
    /// Current cost after pheromone modulation.
    pub current_cost: f64,
}

impl EdgeCost {
    pub fn new(base_cost: f64) -> Self {
        Self {
            base_cost,
            current_cost: base_cost,
        }
    }
}

// ─── Node Types ────────────────────────────────────────────────────────────

/// The top-level palace node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Palace {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
}

/// Wing type categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WingType {
    Person,
    Project,
    Domain,
    Topic,
}

impl std::fmt::Display for WingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Person => write!(f, "person"),
            Self::Project => write!(f, "project"),
            Self::Domain => write!(f, "domain"),
            Self::Topic => write!(f, "topic"),
        }
    }
}

/// A wing of the palace (top-level domain grouping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wing {
    pub id: String,
    pub name: String,
    pub wing_type: WingType,
    pub description: String,
    #[serde(with = "BigArray")]
    pub embedding: Embedding,
    pub pheromones: NodePheromones,
    pub created_at: DateTime<Utc>,
}

/// Room hall type categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HallType {
    Facts,
    Events,
    Discoveries,
    Preferences,
    Advice,
}

impl std::fmt::Display for HallType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Facts => write!(f, "facts"),
            Self::Events => write!(f, "events"),
            Self::Discoveries => write!(f, "discoveries"),
            Self::Preferences => write!(f, "preferences"),
            Self::Advice => write!(f, "advice"),
        }
    }
}

/// A room within a wing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    pub id: String,
    pub name: String,
    pub hall_type: HallType,
    pub description: String,
    #[serde(with = "BigArray")]
    pub embedding: Embedding,
    pub pheromones: NodePheromones,
    pub created_at: DateTime<Utc>,
}

/// A closet within a room (summary container for drawers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Closet {
    pub id: String,
    pub name: String,
    pub summary: String,
    #[serde(with = "BigArray")]
    pub embedding: Embedding,
    pub pheromones: NodePheromones,
    pub drawer_count: u64,
    pub created_at: DateTime<Utc>,
}

/// Source type for a drawer's content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DrawerSource {
    Conversation,
    File,
    Api,
    Agent,
}

impl std::fmt::Display for DrawerSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conversation => write!(f, "conversation"),
            Self::File => write!(f, "file"),
            Self::Api => write!(f, "api"),
            Self::Agent => write!(f, "agent"),
        }
    }
}

/// A drawer — the atomic unit of memory. Contains verbatim text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drawer {
    pub id: String,
    pub content: String,
    #[serde(with = "BigArray")]
    pub embedding: Embedding,
    pub source: DrawerSource,
    pub source_file: Option<String>,
    pub importance: f64,
    pub pheromones: NodePheromones,
    pub created_at: DateTime<Utc>,
    pub accessed_at: DateTime<Utc>,
}

/// Entity type categories for the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityType {
    Person,
    Concept,
    Event,
    Place,
    Organization,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Person => write!(f, "person"),
            Self::Concept => write!(f, "concept"),
            Self::Event => write!(f, "event"),
            Self::Place => write!(f, "place"),
            Self::Organization => write!(f, "organization"),
        }
    }
}

/// An entity in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: EntityType,
    pub description: String,
    #[serde(with = "BigArray")]
    pub embedding: Embedding,
    pub pheromones: NodePheromones,
    pub created_at: DateTime<Utc>,
}

/// A specialist agent that manages a wing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub focus: String,
    pub diary: String,
    #[serde(with = "BigArray")]
    pub goal_embedding: Embedding,
    pub temperature: f64,
    pub created_at: DateTime<Utc>,
}

// ─── Edge Types ────────────────────────────────────────────────────────────

/// Palace → Wing relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contains {
    pub from: String,
    pub to: String,
}

/// Wing → Room relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasRoom {
    pub from: String,
    pub to: String,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
}

impl HasRoom {
    pub fn new(from: String, to: String) -> Self {
        Self {
            from,
            to,
            cost: EdgeCost::new(0.3),
            pheromones: EdgePheromones::default(),
        }
    }
}

/// Room → Closet relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasCloset {
    pub from: String,
    pub to: String,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
}

impl HasCloset {
    pub fn new(from: String, to: String) -> Self {
        Self {
            from,
            to,
            cost: EdgeCost::new(0.3),
            pheromones: EdgePheromones::default(),
        }
    }
}

/// Closet → Drawer relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasDrawer {
    pub from: String,
    pub to: String,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
}

impl HasDrawer {
    pub fn new(from: String, to: String) -> Self {
        Self {
            from,
            to,
            cost: EdgeCost::new(0.3),
            pheromones: EdgePheromones::default(),
        }
    }
}

/// Room → Room (same wing) relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hall {
    pub from: String,
    pub to: String,
    pub hall_type: String,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
}

impl Hall {
    pub fn new(from: String, to: String, hall_type: String) -> Self {
        Self {
            from,
            to,
            hall_type,
            cost: EdgeCost::new(0.5),
            pheromones: EdgePheromones::default(),
        }
    }
}

/// Room → Room (across wings) relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tunnel {
    pub from: String,
    pub to: String,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
}

impl Tunnel {
    pub fn new(from: String, to: String) -> Self {
        Self {
            from,
            to,
            cost: EdgeCost::new(0.7),
            pheromones: EdgePheromones::default(),
        }
    }
}

/// Classification of a knowledge graph statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StatementType {
    /// An established, verified fact.
    #[default]
    Fact,
    /// A direct observation (may be noisy or context-dependent).
    Observation,
    /// A conclusion derived from other statements.
    Inference,
    /// A tentative, unverified proposition.
    Hypothesis,
}


impl std::fmt::Display for StatementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fact => write!(f, "fact"),
            Self::Observation => write!(f, "observation"),
            Self::Inference => write!(f, "inference"),
            Self::Hypothesis => write!(f, "hypothesis"),
        }
    }
}

/// Entity → Entity knowledge graph relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatesTo {
    pub from: String,
    pub to: String,
    pub predicate: String,
    pub confidence: f64,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub observed_at: DateTime<Utc>,
    /// Timestamp when this triple was retracted/invalidated in the system.
    #[serde(default)]
    pub invalidated_at: Option<DateTime<Utc>>,
    /// Classification of this statement (fact, observation, inference, hypothesis).
    #[serde(default)]
    pub statement_type: StatementType,
}

impl RelatesTo {
    pub fn new(from: String, to: String, predicate: String) -> Self {
        Self {
            from,
            to,
            predicate,
            confidence: 0.5,
            cost: EdgeCost::new(1.0),
            pheromones: EdgePheromones::default(),
            valid_from: None,
            valid_to: None,
            observed_at: Utc::now(),
            invalidated_at: None,
            statement_type: StatementType::default(),
        }
    }
}

/// Drawer → Entity reference relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct References {
    pub from: String,
    pub to: String,
    pub relevance: f64,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
}

impl References {
    pub fn new(from: String, to: String) -> Self {
        Self {
            from,
            to,
            relevance: 1.0,
            cost: EdgeCost::new(0.5),
            pheromones: EdgePheromones::default(),
        }
    }
}

/// Drawer → Drawer semantic similarity relationship (auto-computed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarTo {
    pub from: String,
    pub to: String,
    pub similarity: f64,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
}

impl SimilarTo {
    pub fn new(from: String, to: String, similarity: f64) -> Self {
        let base_cost = 1.0 - similarity;
        Self {
            from,
            to,
            similarity,
            cost: EdgeCost::new(base_cost),
            pheromones: EdgePheromones::default(),
        }
    }
}

/// Agent → Wing management relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manages {
    pub from: String,
    pub to: String,
}

/// Agent → Drawer investigation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InvestigationResult {
    Useful,
    Irrelevant,
    Contradicts,
}

/// Agent → Drawer investigation relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Investigated {
    pub from: String,
    pub to: String,
    pub result: InvestigationResult,
    pub investigated_at: DateTime<Utc>,
}

// ─── Generic graph node/edge for algorithms ────────────────────────────────

/// A generic node reference used by pathfinding and agent algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    #[serde(with = "BigArray")]
    pub embedding: Embedding,
    pub pheromones: NodePheromones,
    /// Number of edges connected to this node.
    pub degree: usize,
}

/// A generic edge reference used by pathfinding and agent algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub relation_type: String,
    pub cost: EdgeCost,
    pub pheromones: EdgePheromones,
}

/// Relation type weight table (Appendix A).
pub fn structural_cost(relation_type: &str) -> f64 {
    match relation_type {
        "CONTAINS" => 0.2,
        "HAS_ROOM" | "HAS_CLOSET" | "HAS_DRAWER" => 0.3,
        "HALL" => 0.5,
        "TUNNEL" => 0.7,
        "REFERENCES" => 0.5,
        "SIMILAR_TO" => 0.4,
        "RELATES_TO" => 0.8,
        // Knowledge graph predicates
        "instance_of" | "subclass_of" => 0.3,
        "part_of" | "has_part" => 0.5,
        "causes" | "inhibits" => 0.6,
        "correlates_with" | "located_in" => 0.7,
        "created_by" | "used_for" => 0.8,
        "occurred_at" => 1.0,
        "identified_by" => 1.5,
        _ => 1.0, // DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_embedding() {
        let emb = zero_embedding();
        assert_eq!(emb.len(), EMBEDDING_DIM);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_node_pheromones_default() {
        let p = NodePheromones::default();
        assert_eq!(p.exploitation, 0.0);
        assert_eq!(p.exploration, 0.0);
    }

    #[test]
    fn test_edge_cost_new() {
        let cost = EdgeCost::new(0.5);
        assert_eq!(cost.base_cost, 0.5);
        assert_eq!(cost.current_cost, 0.5);
    }

    #[test]
    fn test_structural_cost_table() {
        assert_eq!(structural_cost("CONTAINS"), 0.2);
        assert_eq!(structural_cost("HAS_ROOM"), 0.3);
        assert_eq!(structural_cost("HALL"), 0.5);
        assert_eq!(structural_cost("TUNNEL"), 0.7);
        assert_eq!(structural_cost("RELATES_TO"), 0.8);
        assert_eq!(structural_cost("causes"), 0.6);
        assert_eq!(structural_cost("unknown_type"), 1.0);
    }

    #[test]
    fn test_similar_to_cost() {
        let sim = SimilarTo::new("a".into(), "b".into(), 0.8);
        assert!((sim.cost.base_cost - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_wing_type_display() {
        assert_eq!(WingType::Person.to_string(), "person");
        assert_eq!(WingType::Project.to_string(), "project");
    }

    #[test]
    fn test_drawer_source_display() {
        assert_eq!(DrawerSource::Conversation.to_string(), "conversation");
        assert_eq!(DrawerSource::Agent.to_string(), "agent");
    }

    #[test]
    fn test_relates_to_defaults() {
        let rel = RelatesTo::new("e1".into(), "e2".into(), "causes".into());
        assert_eq!(rel.confidence, 0.5);
        assert_eq!(rel.cost.base_cost, 1.0);
        assert!(rel.valid_to.is_none());
    }

    #[test]
    fn test_graph_edge_create() {
        let edge = GraphEdge {
            from: "a".into(),
            to: "b".into(),
            relation_type: "HALL".into(),
            cost: EdgeCost::new(0.5),
            pheromones: EdgePheromones::default(),
        };
        assert_eq!(edge.relation_type, "HALL");
    }
}
