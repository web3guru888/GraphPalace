//! Semantic A* pathfinding algorithm from spec §5.4.
//!
//! Combines composite edge costs with an adaptive semantic heuristic
//! for efficient navigation through the GraphPalace.

use std::collections::{BinaryHeap, HashMap};

use ordered_float::OrderedFloat;

use gp_core::config::{AStarConfig, CostWeights};
use gp_core::types::{GraphEdge, GraphNode};

use crate::edge_cost::composite_edge_cost;
use crate::heuristic::semantic_heuristic;
use crate::provenance::{PathResult, ProvenanceStep};

/// Trait abstracting graph storage for the A* algorithm.
///
/// Implementations provide node lookups and neighbor enumeration
/// over whatever backend stores the GraphPalace data.
pub trait GraphAccess {
    /// Look up a node by its ID.
    fn get_node(&self, id: &str) -> Option<GraphNode>;
    /// Return all outgoing edges from `id` together with their target nodes.
    fn get_neighbors(&self, id: &str) -> Vec<(GraphEdge, GraphNode)>;
}

/// Internal A* node for the priority queue (min-heap via reversed Ord).
#[derive(Debug)]
struct AStarNode {
    node_id: String,
    f_cost: OrderedFloat<f64>,
    g_cost: f64,
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
    }
}

impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed: BinaryHeap is a max-heap, so we flip for min-heap behavior.
        other.f_cost.cmp(&self.f_cost)
    }
}

/// Semantic A* pathfinder.
///
/// Uses [`composite_edge_cost`] for g-cost increments and
/// [`semantic_heuristic`] for the h-cost estimate.
pub struct SemanticAStar {
    pub config: AStarConfig,
    pub cost_weights: CostWeights,
}

impl SemanticAStar {
    /// Create a new pathfinder with the given configuration.
    pub fn new(config: AStarConfig, cost_weights: CostWeights) -> Self {
        Self {
            config,
            cost_weights,
        }
    }

    /// Find the shortest semantic path from `start` to `goal`.
    ///
    /// Returns `None` if no path exists or `max_iterations` is exceeded.
    pub fn find_path(
        &self,
        graph: &dyn GraphAccess,
        start: &str,
        goal: &str,
    ) -> Option<PathResult> {
        let start_node = graph.get_node(start)?;
        let goal_node = graph.get_node(goal)?;

        // h for start
        let h_start = semantic_heuristic(
            &start_node.embedding,
            &goal_node.embedding,
            start_node.degree,
            self.config.cross_domain_threshold,
        );

        let mut open = BinaryHeap::new();
        open.push(AStarNode {
            node_id: start.to_string(),
            f_cost: OrderedFloat(h_start),
            g_cost: 0.0,
        });

        // Best known g-cost to each node.
        let mut g_scores: HashMap<String, f64> = HashMap::new();
        g_scores.insert(start.to_string(), 0.0);

        // Backtracking: node_id → (parent_id, edge_relation_type, g_cost, h_cost).
        let mut came_from: HashMap<String, (String, String, f64, f64)> = HashMap::new();

        let mut iterations = 0usize;
        let mut nodes_expanded = 0usize;

        while let Some(current) = open.pop() {
            iterations += 1;
            if iterations > self.config.max_iterations {
                return None;
            }

            // Skip if we've already found a better path to this node.
            if let Some(&best_g) = g_scores.get(&current.node_id)
                && current.g_cost > best_g + 1e-10 {
                    continue;
                }

            nodes_expanded += 1;

            // Goal reached!
            if current.node_id == goal {
                return Some(self.reconstruct_path(
                    &came_from,
                    &start_node,
                    &goal_node,
                    current.g_cost,
                    iterations,
                    nodes_expanded,
                ));
            }

            // Expand neighbors.
            let neighbors = graph.get_neighbors(&current.node_id);
            for (edge, neighbor_node) in &neighbors {
                let edge_cost = composite_edge_cost(
                    edge,
                    &neighbor_node.embedding,
                    &goal_node.embedding,
                    &self.cost_weights,
                );
                let tentative_g = current.g_cost + edge_cost;

                let is_better = match g_scores.get(&neighbor_node.id) {
                    Some(&existing) => tentative_g < existing - 1e-10,
                    None => true,
                };

                if is_better {
                    let h = semantic_heuristic(
                        &neighbor_node.embedding,
                        &goal_node.embedding,
                        neighbor_node.degree,
                        self.config.cross_domain_threshold,
                    );

                    g_scores.insert(neighbor_node.id.clone(), tentative_g);
                    came_from.insert(
                        neighbor_node.id.clone(),
                        (
                            current.node_id.clone(),
                            edge.relation_type.clone(),
                            tentative_g,
                            h,
                        ),
                    );

                    open.push(AStarNode {
                        node_id: neighbor_node.id.clone(),
                        f_cost: OrderedFloat(tentative_g + h),
                        g_cost: tentative_g,
                    });
                }
            }
        }

        // Open set exhausted, no path found.
        None
    }

    /// Reconstruct the path from start to goal using the came_from map.
    fn reconstruct_path(
        &self,
        came_from: &HashMap<String, (String, String, f64, f64)>,
        start_node: &GraphNode,
        goal_node: &GraphNode,
        total_cost: f64,
        iterations: usize,
        nodes_expanded: usize,
    ) -> PathResult {
        let mut path = Vec::new();
        let mut edges = Vec::new();
        let mut provenance = Vec::new();

        // Walk backward from goal to start.
        let mut current_id = goal_node.id.clone();
        loop {
            if let Some((parent, edge_type, g, h)) = came_from.get(&current_id) {
                provenance.push(ProvenanceStep {
                    node_id: current_id.clone(),
                    edge_type: edge_type.clone(),
                    g_cost: *g,
                    h_cost: *h,
                    f_cost: g + h,
                });
                path.push(current_id.clone());
                edges.push(edge_type.clone());
                current_id = parent.clone();
            } else {
                // This is the start node.
                let h_start = semantic_heuristic(
                    &start_node.embedding,
                    &goal_node.embedding,
                    start_node.degree,
                    self.config.cross_domain_threshold,
                );
                provenance.push(ProvenanceStep {
                    node_id: current_id.clone(),
                    edge_type: String::new(),
                    g_cost: 0.0,
                    h_cost: h_start,
                    f_cost: h_start,
                });
                path.push(current_id);
                break;
            }
        }

        // Reverse to get start → goal order.
        path.reverse();
        edges.reverse();
        provenance.reverse();

        PathResult {
            path,
            edges,
            total_cost,
            iterations,
            nodes_expanded,
            provenance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::types::{EdgeCost, EdgePheromones, GraphEdge, GraphNode, NodePheromones};

    /// A simple in-memory graph for testing.
    struct MockGraph {
        nodes: HashMap<String, GraphNode>,
        edges: HashMap<String, Vec<(GraphEdge, String)>>, // from_id → [(edge, to_id)]
    }

    impl MockGraph {
        fn new() -> Self {
            Self {
                nodes: HashMap::new(),
                edges: HashMap::new(),
            }
        }

        fn add_node(&mut self, id: &str, embedding: &[f32]) {
            let mut emb = [0.0f32; 384];
            for (i, &v) in embedding.iter().enumerate() {
                if i < 384 {
                    emb[i] = v;
                }
            }
            self.nodes.insert(
                id.to_string(),
                GraphNode {
                    id: id.to_string(),
                    label: id.to_string(),
                    embedding: emb,
                    pheromones: NodePheromones::default(),
                    degree: 0,
                },
            );
        }

        fn add_edge(&mut self, from: &str, to: &str, relation: &str) {
            let edge = GraphEdge {
                from: from.to_string(),
                to: to.to_string(),
                relation_type: relation.to_string(),
                cost: EdgeCost::new(gp_core::types::structural_cost(relation)),
                pheromones: EdgePheromones::default(),
            };
            self.edges
                .entry(from.to_string())
                .or_default()
                .push((edge, to.to_string()));

            // Update degrees.
            if let Some(n) = self.nodes.get_mut(from) {
                n.degree += 1;
            }
            if let Some(n) = self.nodes.get_mut(to) {
                n.degree += 1;
            }
        }
    }

    impl GraphAccess for MockGraph {
        fn get_node(&self, id: &str) -> Option<GraphNode> {
            self.nodes.get(id).cloned()
        }

        fn get_neighbors(&self, id: &str) -> Vec<(GraphEdge, GraphNode)> {
            match self.edges.get(id) {
                Some(neighbors) => neighbors
                    .iter()
                    .filter_map(|(edge, to_id)| {
                        self.nodes.get(to_id).map(|node| (edge.clone(), node.clone()))
                    })
                    .collect(),
                None => vec![],
            }
        }
    }

    fn default_astar() -> SemanticAStar {
        SemanticAStar::new(AStarConfig::default(), CostWeights::default())
    }

    #[test]
    fn find_path_linear_a_b_c() {
        let mut graph = MockGraph::new();
        // A, B, C all have similar embeddings so semantic cost is low.
        graph.add_node("A", &[1.0, 0.1, 0.0]);
        graph.add_node("B", &[0.9, 0.2, 0.0]);
        graph.add_node("C", &[0.8, 0.3, 0.0]);
        graph.add_edge("A", "B", "HALL");
        graph.add_edge("B", "C", "HALL");

        let astar = default_astar();
        let result = astar.find_path(&graph, "A", "C");
        assert!(result.is_some(), "should find path A→B→C");

        let result = result.unwrap();
        assert_eq!(result.path, vec!["A", "B", "C"]);
        assert_eq!(result.edges, vec!["HALL", "HALL"]);
        assert!(result.total_cost > 0.0);
        assert!(result.iterations > 0);
        assert!(result.nodes_expanded >= 2); // at least A and B expanded
        assert_eq!(result.provenance.len(), 3);
        assert_eq!(result.provenance[0].node_id, "A");
        assert_eq!(result.provenance[0].edge_type, "");
        assert_eq!(result.provenance[2].node_id, "C");
    }

    #[test]
    fn find_path_no_path_returns_none() {
        let mut graph = MockGraph::new();
        graph.add_node("A", &[1.0, 0.0]);
        graph.add_node("Z", &[0.0, 1.0]);
        // No edges → no path.

        let astar = default_astar();
        assert!(astar.find_path(&graph, "A", "Z").is_none());
    }

    #[test]
    fn find_path_nonexistent_start_returns_none() {
        let mut graph = MockGraph::new();
        graph.add_node("A", &[1.0]);
        let astar = default_astar();
        assert!(astar.find_path(&graph, "MISSING", "A").is_none());
    }

    #[test]
    fn find_path_nonexistent_goal_returns_none() {
        let mut graph = MockGraph::new();
        graph.add_node("A", &[1.0]);
        let astar = default_astar();
        assert!(astar.find_path(&graph, "A", "MISSING").is_none());
    }

    #[test]
    fn find_path_start_equals_goal() {
        let mut graph = MockGraph::new();
        graph.add_node("A", &[1.0, 0.0]);

        let astar = default_astar();
        let result = astar.find_path(&graph, "A", "A");
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.path, vec!["A"]);
        assert!(result.edges.is_empty());
        assert!((result.total_cost - 0.0).abs() < 1e-10);
    }

    #[test]
    fn max_iterations_limit() {
        let mut graph = MockGraph::new();
        // Create a long chain that exceeds max_iterations=3.
        graph.add_node("A", &[1.0, 0.0]);
        graph.add_node("B", &[0.9, 0.1]);
        graph.add_node("C", &[0.8, 0.2]);
        graph.add_node("D", &[0.7, 0.3]);
        graph.add_node("E", &[0.6, 0.4]);
        graph.add_edge("A", "B", "HALL");
        graph.add_edge("B", "C", "HALL");
        graph.add_edge("C", "D", "HALL");
        graph.add_edge("D", "E", "HALL");

        let config = AStarConfig {
            max_iterations: 3,
            cross_domain_threshold: 0.3,
        };
        let astar = SemanticAStar::new(config, CostWeights::default());
        let result = astar.find_path(&graph, "A", "E");
        assert!(result.is_none(), "should fail due to max_iterations");
    }

    #[test]
    fn prefers_shorter_cost_path() {
        // Two paths: A→B→D (cheap) and A→C→D (expensive).
        let mut graph = MockGraph::new();
        // Embeddings: all similar to D so semantic cost is low.
        graph.add_node("A", &[1.0, 0.0, 0.0]);
        graph.add_node("B", &[0.95, 0.05, 0.0]);
        graph.add_node("C", &[0.0, 1.0, 0.0]); // very different → high semantic cost
        graph.add_node("D", &[0.9, 0.1, 0.0]);

        // B is semantically close to D, C is far.
        graph.add_edge("A", "B", "HAS_ROOM"); // struct cost 0.3
        graph.add_edge("A", "C", "HAS_ROOM"); // struct cost 0.3
        graph.add_edge("B", "D", "HAS_ROOM"); // struct cost 0.3
        graph.add_edge("C", "D", "HAS_ROOM"); // struct cost 0.3

        let astar = default_astar();
        let result = astar.find_path(&graph, "A", "D").unwrap();

        // Should prefer A→B→D because B is semantically close to D.
        assert_eq!(result.path, vec!["A", "B", "D"]);
    }

    #[test]
    fn provenance_costs_are_monotonically_nondecreasing() {
        let mut graph = MockGraph::new();
        graph.add_node("A", &[1.0, 0.0]);
        graph.add_node("B", &[0.9, 0.1]);
        graph.add_node("C", &[0.8, 0.2]);
        graph.add_edge("A", "B", "HALL");
        graph.add_edge("B", "C", "HALL");

        let astar = default_astar();
        let result = astar.find_path(&graph, "A", "C").unwrap();

        for window in result.provenance.windows(2) {
            assert!(
                window[1].g_cost >= window[0].g_cost - 1e-10,
                "g_cost should be non-decreasing"
            );
        }
    }
}
