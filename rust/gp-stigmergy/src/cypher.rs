//! Cypher query generation for bulk pheromone operations (spec §4.2).
//!
//! Generates parameterized Cypher query strings for executing pheromone
//! decay, deposit, and cost recomputation operations against a Kuzu
//! database. Since we don't have Kuzu FFI bindings yet, these produce
//! query strings + parameter maps that can be passed to the DB when available.

use std::collections::HashMap;

use gp_core::config::PheromoneConfig;
use serde::{Deserialize, Serialize};

/// A value that can be bound to a Cypher query parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CypherValue {
    /// A floating-point number.
    Float(f64),
    /// An integer.
    Int(i64),
    /// A text string.
    String(String),
    /// A list of strings (for IN clauses).
    StringList(Vec<String>),
}

impl From<f64> for CypherValue {
    fn from(v: f64) -> Self { CypherValue::Float(v) }
}

impl From<i64> for CypherValue {
    fn from(v: i64) -> Self { CypherValue::Int(v) }
}

impl From<String> for CypherValue {
    fn from(v: String) -> Self { CypherValue::String(v) }
}

impl From<&str> for CypherValue {
    fn from(v: &str) -> Self { CypherValue::String(v.to_string()) }
}

/// A Cypher query with named parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypherQuery {
    /// The Cypher query string with `$param_name` placeholders.
    pub query: String,
    /// Named parameter bindings.
    pub params: HashMap<String, CypherValue>,
}

impl CypherQuery {
    /// Create a new query with no parameters.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            params: HashMap::new(),
        }
    }

    /// Create a new query with parameters.
    pub fn with_params(query: impl Into<String>, params: HashMap<String, CypherValue>) -> Self {
        Self {
            query: query.into(),
            params,
        }
    }

    /// Add a parameter binding.
    pub fn param(mut self, name: impl Into<String>, value: impl Into<CypherValue>) -> Self {
        self.params.insert(name.into(), value.into());
        self
    }
}

// ─── Decay Queries ────────────────────────────────────────────────────────

/// Generate a Cypher query for bulk node pheromone decay.
///
/// Decays exploitation and exploration pheromones on all nodes where
/// at least one pheromone is above the threshold (0.001).
///
/// Parameters bound: `$exploitation_rate`, `$exploration_rate`.
pub fn decay_node_pheromones_cypher(config: &PheromoneConfig) -> CypherQuery {
    let query = r#"MATCH (n)
WHERE n.exploitation_pheromone > 0.001 OR n.exploration_pheromone > 0.001
SET n.exploitation_pheromone = n.exploitation_pheromone * (1.0 - $exploitation_rate),
    n.exploration_pheromone = n.exploration_pheromone * (1.0 - $exploration_rate)"#;

    CypherQuery::new(query)
        .param("exploitation_rate", CypherValue::Float(config.exploitation_decay))
        .param("exploration_rate", CypherValue::Float(config.exploration_decay))
}

/// Generate a Cypher query for bulk edge pheromone decay.
///
/// Decays success, traversal, and recency pheromones on all edges where
/// at least one pheromone is above the threshold (0.001).
///
/// Parameters bound: `$success_rate`, `$traversal_rate`, `$recency_rate`.
pub fn decay_edge_pheromones_cypher(config: &PheromoneConfig) -> CypherQuery {
    let query = r#"MATCH ()-[e]->()
WHERE e.success_pheromone > 0.001 OR e.traversal_pheromone > 0.001 OR e.recency_pheromone > 0.001
SET e.success_pheromone = e.success_pheromone * (1.0 - $success_rate),
    e.traversal_pheromone = e.traversal_pheromone * (1.0 - $traversal_rate),
    e.recency_pheromone = e.recency_pheromone * (1.0 - $recency_rate)"#;

    CypherQuery::new(query)
        .param("success_rate", CypherValue::Float(config.success_decay))
        .param("traversal_rate", CypherValue::Float(config.traversal_decay))
        .param("recency_rate", CypherValue::Float(config.recency_decay))
}

/// Generate Cypher queries for both node and edge decay in one batch.
pub fn decay_all_pheromones_cypher(config: &PheromoneConfig) -> Vec<CypherQuery> {
    vec![
        decay_node_pheromones_cypher(config),
        decay_edge_pheromones_cypher(config),
    ]
}

// ─── Deposit Queries ──────────────────────────────────────────────────────

/// Generate Cypher queries for depositing pheromones along a successful path.
///
/// For each edge at position `i` in a path of length `n`:
/// - `success_pheromone += base_reward × (1.0 - i/n)` (position-weighted)
/// - `traversal_pheromone += 0.1`
/// - `recency_pheromone = 1.0` (reset to max)
///
/// For each node on the path:
/// - `exploitation_pheromone += 0.2`
///
/// Returns a vector of CypherQuery statements (one per edge + one for all nodes).
pub fn deposit_success_cypher(
    path_node_ids: &[&str],
    path_edge_indices: &[usize],
    base_reward: f64,
) -> Vec<CypherQuery> {
    let mut queries = Vec::new();
    let n = path_edge_indices.len() as f64;

    // Edge deposits: one query per edge with position-weighted reward
    for (i, _edge_idx) in path_edge_indices.iter().enumerate() {
        let position_weight = if n > 0.0 { 1.0 - (i as f64 / n) } else { 1.0 };
        let reward = base_reward * position_weight;

        let query = "MATCH ()-[e]->() WHERE id(e) = $edge_id \
             SET e.success_pheromone = e.success_pheromone + $reward, \
             e.traversal_pheromone = e.traversal_pheromone + 0.1, \
             e.recency_pheromone = 1.0".to_string();
        queries.push(
            CypherQuery::new(query)
                .param("edge_id", CypherValue::Int(*_edge_idx as i64))
                .param("reward", CypherValue::Float(reward)),
        );
    }

    // Node deposits: exploitation += 0.2 for all nodes on the path
    if !path_node_ids.is_empty() {
        let node_list: Vec<String> = path_node_ids.iter().map(|s| s.to_string()).collect();
        let query = "MATCH (n) WHERE n.id IN $node_ids \
                     SET n.exploitation_pheromone = n.exploitation_pheromone + 0.2";
        queries.push(
            CypherQuery::new(query)
                .param("node_ids", CypherValue::StringList(node_list)),
        );
    }

    queries
}

/// Generate a Cypher query for depositing exploration pheromone on a node.
///
/// `exploration_pheromone += 0.3`
pub fn deposit_exploration_cypher(node_id: &str) -> CypherQuery {
    let query = "MATCH (n) WHERE n.id = $node_id \
                 SET n.exploration_pheromone = n.exploration_pheromone + 0.3";
    CypherQuery::new(query)
        .param("node_id", CypherValue::String(node_id.to_string()))
}

// ─── Cost Recomputation Queries ───────────────────────────────────────────

/// Generate a Cypher query for bulk edge cost recomputation.
///
/// Applies the formula from spec §4.4:
/// ```text
/// pheromone_factor = 0.5 × min(success, 1) + 0.3 × min(recency, 1) + 0.2 × min(traversal, 1)
/// current_cost = base_cost × (1.0 - pheromone_factor × 0.5)
/// ```
///
/// Clamped to [0.0, 10.0].
pub fn recompute_edge_costs_cypher() -> CypherQuery {
    let query = r#"MATCH ()-[e]->()
WHERE e.base_cost IS NOT NULL
SET e.current_cost = CASE
    WHEN e.base_cost * (1.0 - (0.5 * LEAST(e.success_pheromone, 1.0) + 0.3 * LEAST(e.recency_pheromone, 1.0) + 0.2 * LEAST(e.traversal_pheromone, 1.0)) * 0.5) < 0.0 THEN 0.0
    WHEN e.base_cost * (1.0 - (0.5 * LEAST(e.success_pheromone, 1.0) + 0.3 * LEAST(e.recency_pheromone, 1.0) + 0.2 * LEAST(e.traversal_pheromone, 1.0)) * 0.5) > 10.0 THEN 10.0
    ELSE e.base_cost * (1.0 - (0.5 * LEAST(e.success_pheromone, 1.0) + 0.3 * LEAST(e.recency_pheromone, 1.0) + 0.2 * LEAST(e.traversal_pheromone, 1.0)) * 0.5)
END"#;
    CypherQuery::new(query)
}

// ─── Status / Query Queries ───────────────────────────────────────────────

/// Generate a Cypher query to get pheromone status for a specific node.
pub fn pheromone_status_node_cypher(node_id: &str) -> CypherQuery {
    let query = "MATCH (n) WHERE n.id = $node_id \
                 RETURN n.id, n.exploitation_pheromone, n.exploration_pheromone";
    CypherQuery::new(query)
        .param("node_id", CypherValue::String(node_id.to_string()))
}

/// Generate a Cypher query to get pheromone status for edges from a node.
pub fn pheromone_status_edges_cypher(node_id: &str) -> CypherQuery {
    let query = "MATCH (n)-[e]->(m) WHERE n.id = $node_id \
                 RETURN n.id, m.id, e.success_pheromone, e.traversal_pheromone, e.recency_pheromone, e.current_cost";
    CypherQuery::new(query)
        .param("node_id", CypherValue::String(node_id.to_string()))
}

/// Generate a Cypher query to find the k most-traversed (hot) paths.
///
/// Orders edges by combined traversal + success pheromone strength.
pub fn hot_paths_cypher(k: usize, wing_id: Option<&str>) -> CypherQuery {
    let query = match wing_id {
        Some(_) => format!(
            "MATCH (w:Wing)-[*]->(n)-[e]->(m) WHERE w.id = $wing_id \
             RETURN n.id, m.id, e.success_pheromone, e.traversal_pheromone, \
             (e.success_pheromone + e.traversal_pheromone) AS heat \
             ORDER BY heat DESC LIMIT {k}"
        ),
        None => format!(
            "MATCH ()-[e]->() \
             RETURN startNode(e).id AS from_id, endNode(e).id AS to_id, \
             e.success_pheromone, e.traversal_pheromone, \
             (e.success_pheromone + e.traversal_pheromone) AS heat \
             ORDER BY heat DESC LIMIT {k}"
        ),
    };
    let mut cq = CypherQuery::new(query);
    if let Some(wid) = wing_id {
        cq = cq.param("wing_id", CypherValue::String(wid.to_string()));
    }
    cq
}

/// Generate a Cypher query to find the k least-explored (cold) spots.
///
/// Orders nodes by low exploitation and low exploration pheromones.
pub fn cold_spots_cypher(k: usize, wing_id: Option<&str>) -> CypherQuery {
    let query = match wing_id {
        Some(_) => format!(
            "MATCH (w:Wing)-[*]->(n) WHERE w.id = $wing_id \
             RETURN n.id, n.exploitation_pheromone, n.exploration_pheromone, \
             (n.exploitation_pheromone + n.exploration_pheromone) AS warmth \
             ORDER BY warmth ASC LIMIT {k}"
        ),
        None => format!(
            "MATCH (n) \
             WHERE n.exploitation_pheromone IS NOT NULL \
             RETURN n.id, n.exploitation_pheromone, n.exploration_pheromone, \
             (n.exploitation_pheromone + n.exploration_pheromone) AS warmth \
             ORDER BY warmth ASC LIMIT {k}"
        ),
    };
    let mut cq = CypherQuery::new(query);
    if let Some(wid) = wing_id {
        cq = cq.param("wing_id", CypherValue::String(wid.to_string()));
    }
    cq
}

/// Generate a Cypher query to force an immediate decay cycle.
///
/// This is equivalent to calling both node and edge decay queries.
pub fn force_decay_cypher(config: &PheromoneConfig) -> Vec<CypherQuery> {
    decay_all_pheromones_cypher(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::config::PheromoneConfig;

    fn default_config() -> PheromoneConfig {
        PheromoneConfig::default()
    }

    // ─── Decay queries ───────────────────────────────────────────────────

    #[test]
    fn test_decay_node_query_contains_set() {
        let q = decay_node_pheromones_cypher(&default_config());
        assert!(q.query.contains("SET n.exploitation_pheromone"));
        assert!(q.query.contains("SET") || q.query.contains("set"));
    }

    #[test]
    fn test_decay_node_query_has_params() {
        let q = decay_node_pheromones_cypher(&default_config());
        assert_eq!(q.params.len(), 2);
        assert_eq!(q.params["exploitation_rate"], CypherValue::Float(0.02));
        assert_eq!(q.params["exploration_rate"], CypherValue::Float(0.05));
    }

    #[test]
    fn test_decay_edge_query_contains_match() {
        let q = decay_edge_pheromones_cypher(&default_config());
        assert!(q.query.contains("MATCH"));
        assert!(q.query.contains("success_pheromone"));
        assert!(q.query.contains("traversal_pheromone"));
        assert!(q.query.contains("recency_pheromone"));
    }

    #[test]
    fn test_decay_edge_query_has_params() {
        let q = decay_edge_pheromones_cypher(&default_config());
        assert_eq!(q.params.len(), 3);
        assert_eq!(q.params["success_rate"], CypherValue::Float(0.01));
        assert_eq!(q.params["traversal_rate"], CypherValue::Float(0.03));
        assert_eq!(q.params["recency_rate"], CypherValue::Float(0.10));
    }

    #[test]
    fn test_decay_all_returns_two_queries() {
        let qs = decay_all_pheromones_cypher(&default_config());
        assert_eq!(qs.len(), 2);
    }

    #[test]
    fn test_decay_custom_config() {
        let config = PheromoneConfig {
            exploitation_decay: 0.05,
            exploration_decay: 0.10,
            success_decay: 0.02,
            traversal_decay: 0.04,
            recency_decay: 0.20,
            decay_interval_cycles: 5,
        };
        let q = decay_node_pheromones_cypher(&config);
        assert_eq!(q.params["exploitation_rate"], CypherValue::Float(0.05));
        assert_eq!(q.params["exploration_rate"], CypherValue::Float(0.10));
    }

    #[test]
    fn test_decay_node_query_has_threshold() {
        let q = decay_node_pheromones_cypher(&default_config());
        assert!(q.query.contains("0.001"), "Should filter by threshold 0.001");
    }

    // ─── Deposit queries ─────────────────────────────────────────────────

    #[test]
    fn test_deposit_success_generates_queries() {
        let nodes = vec!["A", "B", "C"];
        let edges = vec![0, 1];
        let qs = deposit_success_cypher(&nodes, &edges, 1.0);
        // 2 edge queries + 1 node query
        assert_eq!(qs.len(), 3);
    }

    #[test]
    fn test_deposit_success_position_weighting() {
        let nodes = vec!["A", "B", "C", "D"];
        let edges = vec![0, 1, 2];
        let qs = deposit_success_cypher(&nodes, &edges, 2.0);
        
        // Edge 0: reward = 2.0 × (1 - 0/3) = 2.0
        assert_eq!(qs[0].params["reward"], CypherValue::Float(2.0));
        // Edge 1: reward = 2.0 × (1 - 1/3) ≈ 1.333...
        if let CypherValue::Float(v) = qs[1].params["reward"] {
            assert!((v - 2.0 * (1.0 - 1.0/3.0)).abs() < 1e-10);
        }
        // Edge 2: reward = 2.0 × (1 - 2/3) ≈ 0.666...
        if let CypherValue::Float(v) = qs[2].params["reward"] {
            assert!((v - 2.0 * (1.0 - 2.0/3.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_deposit_success_empty_path() {
        let qs = deposit_success_cypher(&[], &[], 1.0);
        assert!(qs.is_empty());
    }

    #[test]
    fn test_deposit_success_node_ids_in_params() {
        let nodes = vec!["n1", "n2"];
        let edges = vec![0];
        let qs = deposit_success_cypher(&nodes, &edges, 1.0);
        // Last query should have node_ids param
        let node_q = &qs[qs.len() - 1];
        match &node_q.params["node_ids"] {
            CypherValue::StringList(ids) => {
                assert_eq!(ids.len(), 2);
                assert_eq!(ids[0], "n1");
                assert_eq!(ids[1], "n2");
            }
            _ => panic!("Expected StringList"),
        }
    }

    #[test]
    fn test_deposit_exploration_query() {
        let q = deposit_exploration_cypher("my_node");
        assert!(q.query.contains("exploration_pheromone"));
        assert!(q.query.contains("0.3"));
        assert_eq!(q.params["node_id"], CypherValue::String("my_node".into()));
    }

    // ─── Cost recomputation ──────────────────────────────────────────────

    #[test]
    fn test_recompute_costs_query() {
        let q = recompute_edge_costs_cypher();
        assert!(q.query.contains("current_cost"));
        assert!(q.query.contains("base_cost"));
        assert!(q.query.contains("success_pheromone"));
        assert!(q.query.contains("recency_pheromone"));
        assert!(q.query.contains("traversal_pheromone"));
        assert!(q.params.is_empty()); // No params needed
    }

    #[test]
    fn test_recompute_costs_has_clamp() {
        let q = recompute_edge_costs_cypher();
        assert!(q.query.contains("0.0"), "Should clamp to minimum 0.0");
        assert!(q.query.contains("10.0"), "Should clamp to maximum 10.0");
    }

    // ─── Status queries ──────────────────────────────────────────────────

    #[test]
    fn test_pheromone_status_node() {
        let q = pheromone_status_node_cypher("wing_1");
        assert!(q.query.contains("exploitation_pheromone"));
        assert!(q.query.contains("exploration_pheromone"));
        assert_eq!(q.params["node_id"], CypherValue::String("wing_1".into()));
    }

    #[test]
    fn test_pheromone_status_edges() {
        let q = pheromone_status_edges_cypher("room_1");
        assert!(q.query.contains("success_pheromone"));
        assert!(q.query.contains("current_cost"));
    }

    // ─── Hot paths / cold spots ──────────────────────────────────────────

    #[test]
    fn test_hot_paths_global() {
        let q = hot_paths_cypher(10, None);
        assert!(q.query.contains("ORDER BY heat DESC"));
        assert!(q.query.contains("LIMIT 10"));
        assert!(q.params.is_empty());
    }

    #[test]
    fn test_hot_paths_scoped_to_wing() {
        let q = hot_paths_cypher(5, Some("wing_research"));
        assert!(q.query.contains("$wing_id"));
        assert!(q.query.contains("LIMIT 5"));
        assert_eq!(q.params["wing_id"], CypherValue::String("wing_research".into()));
    }

    #[test]
    fn test_cold_spots_global() {
        let q = cold_spots_cypher(20, None);
        assert!(q.query.contains("ORDER BY warmth ASC"));
        assert!(q.query.contains("LIMIT 20"));
    }

    #[test]
    fn test_cold_spots_scoped() {
        let q = cold_spots_cypher(3, Some("wing_archive"));
        assert!(q.query.contains("$wing_id"));
        assert_eq!(q.params["wing_id"], CypherValue::String("wing_archive".into()));
    }

    // ─── Force decay ─────────────────────────────────────────────────────

    #[test]
    fn test_force_decay_returns_two() {
        let qs = force_decay_cypher(&default_config());
        assert_eq!(qs.len(), 2);
    }

    // ─── CypherQuery builder ─────────────────────────────────────────────

    #[test]
    fn test_cypher_query_builder() {
        let q = CypherQuery::new("MATCH (n) RETURN n")
            .param("limit", CypherValue::Int(10))
            .param("name", CypherValue::String("test".into()));
        assert_eq!(q.params.len(), 2);
    }

    #[test]
    fn test_cypher_value_from_f64() {
        let v: CypherValue = 3.14.into();
        assert_eq!(v, CypherValue::Float(3.14));
    }

    #[test]
    fn test_cypher_value_from_str() {
        let v: CypherValue = "hello".into();
        assert_eq!(v, CypherValue::String("hello".into()));
    }

    #[test]
    fn test_cypher_value_from_i64() {
        let v: CypherValue = 42i64.into();
        assert_eq!(v, CypherValue::Int(42));
    }

    #[test]
    fn test_cypher_query_serialization() {
        let q = CypherQuery::new("MATCH (n) RETURN n")
            .param("x", CypherValue::Float(1.0));
        let json = serde_json::to_string(&q).unwrap();
        assert!(json.contains("MATCH"));
        let q2: CypherQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.query, q.query);
        assert_eq!(q2.params["x"], CypherValue::Float(1.0));
    }
}
