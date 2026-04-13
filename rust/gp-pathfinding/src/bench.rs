//! Benchmark infrastructure for pathfinding performance testing.
//!
//! Uses `#[cfg(test)]` since we're not adding criterion as a dependency.
//! Generates test graphs of configurable size and measures A* pathfinding.

use std::collections::HashMap;
use std::time::Instant;

use gp_core::config::{AStarConfig, CostWeights};
use gp_core::types::{EdgeCost, EdgePheromones, GraphEdge, GraphNode, NodePheromones};

use crate::astar::{GraphAccess, SemanticAStar};

/// Results from a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the benchmark.
    pub name: String,
    /// Number of nodes in the test graph.
    pub num_nodes: usize,
    /// Number of edges in the test graph.
    pub num_edges: usize,
    /// Time taken for the pathfinding operation.
    pub elapsed_ms: f64,
    /// Whether a path was found.
    pub path_found: bool,
    /// Length of the found path (nodes).
    pub path_length: usize,
    /// Nodes expanded during search.
    pub nodes_expanded: usize,
    /// Number of iterations.
    pub iterations: usize,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] nodes={}, edges={}, time={:.3}ms, found={}, path_len={}, expanded={}",
            self.name, self.num_nodes, self.num_edges, self.elapsed_ms,
            self.path_found, self.path_length, self.nodes_expanded
        )
    }
}

/// Generate a palace hierarchy graph: Palace → Wings → Rooms → Closets → Drawers.
///
/// Structure:
/// - 1 Palace node
/// - `num_wings` Wing nodes (connected to Palace via CONTAINS)
/// - `rooms_per_wing` Room nodes per wing (connected via HAS_ROOM)
/// - `closets_per_room` Closet nodes per room (connected via HAS_CLOSET)
/// - `drawers_per_closet` Drawer nodes per closet (connected via HAS_DRAWER)
///
/// Uses seeded pseudo-random embeddings for deterministic results.
pub struct PalaceGraph {
    nodes: HashMap<String, GraphNode>,
    adjacency: HashMap<String, Vec<(GraphEdge, String)>>,
    pub node_count: usize,
    pub edge_count: usize,
}

impl PalaceGraph {
    /// Build a palace hierarchy graph with the given dimensions.
    pub fn build(
        num_wings: usize,
        rooms_per_wing: usize,
        closets_per_room: usize,
        drawers_per_closet: usize,
    ) -> Self {
        let mut nodes = HashMap::new();
        let mut adjacency: HashMap<String, Vec<(GraphEdge, String)>> = HashMap::new();
        let mut edge_count = 0;

        // Helper: create a deterministic embedding from a seed
        let make_embedding = |seed: usize| -> [f32; 384] {
            let mut emb = [0.0f32; 384];
            // Simple hash-based embedding for determinism
            let mut h = seed as u64;
            for slot in emb.iter_mut() {
                h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *slot = ((h >> 33) as f32) / (u32::MAX as f32) - 0.5;
            }
            // L2 normalize
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for slot in emb.iter_mut() {
                    *slot /= norm;
                }
            }
            emb
        };

        let add_node = |nodes: &mut HashMap<String, GraphNode>, id: &str, seed: usize| {
            nodes.insert(id.to_string(), GraphNode {
                id: id.to_string(),
                label: id.to_string(),
                embedding: make_embedding(seed),
                pheromones: NodePheromones::default(),
                degree: 0,
            });
        };

        let add_edge = |adj: &mut HashMap<String, Vec<(GraphEdge, String)>>,
                        nodes: &mut HashMap<String, GraphNode>,
                        from: &str, to: &str, rel: &str, base_cost: f64| {
            let edge = GraphEdge {
                from: from.to_string(),
                to: to.to_string(),
                relation_type: rel.to_string(),
                cost: EdgeCost::new(base_cost),
                pheromones: EdgePheromones::default(),
            };
            adj.entry(from.to_string()).or_default().push((edge, to.to_string()));
            if let Some(n) = nodes.get_mut(from) { n.degree += 1; }
            if let Some(n) = nodes.get_mut(to) { n.degree += 1; }
        };

        // Palace root
        add_node(&mut nodes, "palace", 0);
        let mut seed = 1;

        for w in 0..num_wings {
            let wing_id = format!("wing_{w}");
            add_node(&mut nodes, &wing_id, seed);
            seed += 1;
            add_edge(&mut adjacency, &mut nodes, "palace", &wing_id, "CONTAINS", 0.2);
            edge_count += 1;

            for r in 0..rooms_per_wing {
                let room_id = format!("room_{w}_{r}");
                add_node(&mut nodes, &room_id, seed);
                seed += 1;
                add_edge(&mut adjacency, &mut nodes, &wing_id, &room_id, "HAS_ROOM", 0.3);
                edge_count += 1;

                // Add halls between rooms in same wing
                if r > 0 {
                    let prev_room = format!("room_{w}_{}", r - 1);
                    add_edge(&mut adjacency, &mut nodes, &prev_room, &room_id, "HALL", 0.5);
                    add_edge(&mut adjacency, &mut nodes, &room_id, &prev_room, "HALL", 0.5);
                    edge_count += 2;
                }

                for c in 0..closets_per_room {
                    let closet_id = format!("closet_{w}_{r}_{c}");
                    add_node(&mut nodes, &closet_id, seed);
                    seed += 1;
                    add_edge(&mut adjacency, &mut nodes, &room_id, &closet_id, "HAS_CLOSET", 0.3);
                    edge_count += 1;

                    for d in 0..drawers_per_closet {
                        let drawer_id = format!("drawer_{w}_{r}_{c}_{d}");
                        add_node(&mut nodes, &drawer_id, seed);
                        seed += 1;
                        add_edge(&mut adjacency, &mut nodes, &closet_id, &drawer_id, "HAS_DRAWER", 0.3);
                        edge_count += 1;
                    }
                }
            }
        }

        // Add tunnels between first rooms of different wings
        for w1 in 0..num_wings {
            for w2 in (w1 + 1)..num_wings {
                let r1 = format!("room_{w1}_0");
                let r2 = format!("room_{w2}_0");
                if nodes.contains_key(&r1) && nodes.contains_key(&r2) {
                    add_edge(&mut adjacency, &mut nodes, &r1, &r2, "TUNNEL", 0.7);
                    add_edge(&mut adjacency, &mut nodes, &r2, &r1, "TUNNEL", 0.7);
                    edge_count += 2;
                }
            }
        }

        let node_count = nodes.len();
        Self { nodes, adjacency, node_count, edge_count }
    }

    /// Set pheromone levels on all edges (for testing pheromone-guided search).
    pub fn set_edge_pheromones(&mut self, success: f64, traversal: f64, recency: f64) {
        for edges in self.adjacency.values_mut() {
            for (edge, _) in edges.iter_mut() {
                edge.pheromones.success = success;
                edge.pheromones.traversal = traversal;
                edge.pheromones.recency = recency;
            }
        }
    }
}

impl GraphAccess for PalaceGraph {
    fn get_node(&self, id: &str) -> Option<GraphNode> {
        self.nodes.get(id).cloned()
    }

    fn get_neighbors(&self, id: &str) -> Vec<(GraphEdge, GraphNode)> {
        match self.adjacency.get(id) {
            Some(neighbors) => neighbors
                .iter()
                .filter_map(|(edge, to_id)| {
                    self.nodes.get(to_id).map(|n| (edge.clone(), n.clone()))
                })
                .collect(),
            None => vec![],
        }
    }
}

/// Run a single benchmark: find path between two nodes and measure time.
#[allow(clippy::too_many_arguments)]
pub fn run_benchmark(
    name: &str,
    graph: &dyn GraphAccess,
    start: &str,
    goal: &str,
    config: AStarConfig,
    weights: CostWeights,
    num_nodes: usize,
    num_edges: usize,
) -> BenchmarkResult {
    let astar = SemanticAStar::new(config, weights);

    let start_time = Instant::now();
    let result = astar.find_path(graph, start, goal);
    let elapsed = start_time.elapsed();

    match result {
        Some(path) => BenchmarkResult {
            name: name.to_string(),
            num_nodes,
            num_edges,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            path_found: true,
            path_length: path.path.len(),
            nodes_expanded: path.nodes_expanded,
            iterations: path.iterations,
        },
        None => BenchmarkResult {
            name: name.to_string(),
            num_nodes,
            num_edges,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            path_found: false,
            path_length: 0,
            nodes_expanded: 0,
            iterations: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_tiny_palace() {
        let graph = PalaceGraph::build(2, 2, 1, 2);
        let result = run_benchmark(
            "tiny_palace",
            &graph,
            "palace",
            "drawer_0_0_0_0",
            AStarConfig::default(),
            CostWeights::default(),
            graph.node_count,
            graph.edge_count,
        );
        assert!(result.path_found, "Should find path in tiny palace");
        assert!(result.path_length >= 4); // palace → wing → room → closet → drawer
        eprintln!("{result}");
    }

    #[test]
    fn bench_small_palace() {
        let graph = PalaceGraph::build(3, 3, 2, 3);
        let result = run_benchmark(
            "small_palace",
            &graph,
            "palace",
            "drawer_2_2_1_2",
            AStarConfig::default(),
            CostWeights::default(),
            graph.node_count,
            graph.edge_count,
        );
        assert!(result.path_found, "Should find path in small palace");
        eprintln!("{result}");
    }

    #[test]
    fn bench_medium_palace() {
        let graph = PalaceGraph::build(5, 5, 3, 4);
        let result = run_benchmark(
            "medium_palace",
            &graph,
            "drawer_0_0_0_0",
            "drawer_4_4_2_3",
            AStarConfig::default(),
            CostWeights::default(),
            graph.node_count,
            graph.edge_count,
        );
        // Might not find path since hierarchy is directed downward
        // and we'd need upward edges or tunnel connections
        eprintln!("{result}");
    }

    #[test]
    fn bench_palace_graph_node_counts() {
        // 2 wings × 2 rooms × 1 closet × 2 drawers = 8 drawers
        // + 1 palace + 2 wings + 4 rooms + 4 closets = 19 nodes
        let graph = PalaceGraph::build(2, 2, 1, 2);
        assert_eq!(graph.node_count, 19);
    }

    #[test]
    fn bench_cross_wing_via_tunnel() {
        let graph = PalaceGraph::build(3, 2, 1, 1);
        // room_0_0 should have tunnel to room_1_0 and room_2_0
        let result = run_benchmark(
            "cross_wing_tunnel",
            &graph,
            "room_0_0",
            "room_2_0",
            AStarConfig::default(),
            CostWeights::default(),
            graph.node_count,
            graph.edge_count,
        );
        assert!(result.path_found, "Should find path via tunnel");
        eprintln!("{result}");
    }

    #[test]
    fn bench_same_wing_via_hall() {
        let graph = PalaceGraph::build(1, 5, 1, 1);
        let result = run_benchmark(
            "same_wing_halls",
            &graph,
            "room_0_0",
            "room_0_4",
            AStarConfig::default(),
            CostWeights::default(),
            graph.node_count,
            graph.edge_count,
        );
        assert!(result.path_found, "Should find path via halls");
        // Path should go room_0_0 → room_0_1 → ... → room_0_4
        assert!(result.path_length >= 5, "Should traverse through halls: len={}", result.path_length);
        eprintln!("{result}");
    }

    #[test]
    fn bench_with_pheromones() {
        let mut graph = PalaceGraph::build(2, 3, 1, 2);
        graph.set_edge_pheromones(0.8, 0.5, 0.9);
        let result = run_benchmark(
            "with_pheromones",
            &graph,
            "palace",
            "drawer_0_1_0_0",
            AStarConfig::default(),
            CostWeights::default(),
            graph.node_count,
            graph.edge_count,
        );
        assert!(result.path_found);
        eprintln!("{result}");
    }

    #[test]
    fn bench_hypothesis_testing_weights() {
        let graph = PalaceGraph::build(2, 3, 1, 1);
        let result = run_benchmark(
            "hypothesis_testing",
            &graph,
            "room_0_0",
            "room_1_2",
            AStarConfig::default(),
            CostWeights::hypothesis_testing(),
            graph.node_count,
            graph.edge_count,
        );
        eprintln!("{result}");
    }

    #[test]
    fn bench_exploratory_research_weights() {
        let graph = PalaceGraph::build(2, 3, 1, 1);
        let result = run_benchmark(
            "exploratory_research",
            &graph,
            "room_0_0",
            "room_1_2",
            AStarConfig::default(),
            CostWeights::exploratory_research(),
            graph.node_count,
            graph.edge_count,
        );
        eprintln!("{result}");
    }

    #[test]
    fn bench_evidence_gathering_weights() {
        let graph = PalaceGraph::build(2, 3, 1, 1);
        let result = run_benchmark(
            "evidence_gathering",
            &graph,
            "room_0_0",
            "room_1_2",
            AStarConfig::default(),
            CostWeights::evidence_gathering(),
            graph.node_count,
            graph.edge_count,
        );
        eprintln!("{result}");
    }

    #[test]
    fn bench_memory_recall_weights() {
        let graph = PalaceGraph::build(2, 3, 1, 1);
        let result = run_benchmark(
            "memory_recall",
            &graph,
            "room_0_0",
            "room_1_2",
            AStarConfig::default(),
            CostWeights::memory_recall(),
            graph.node_count,
            graph.edge_count,
        );
        eprintln!("{result}");
    }

    #[test]
    fn palace_graph_has_tunnels() {
        let graph = PalaceGraph::build(3, 1, 0, 0);
        // 3 wings → 3 tunnels (each bidirectional = 6 directed edges)
        // room_0_0 ↔ room_1_0, room_0_0 ↔ room_2_0, room_1_0 ↔ room_2_0
        let neighbors = graph.get_neighbors("room_0_0");
        let tunnels: Vec<_> = neighbors.iter().filter(|(e, _)| e.relation_type == "TUNNEL").collect();
        assert_eq!(tunnels.len(), 2, "room_0_0 should have tunnels to rooms in other wings");
    }

    #[test]
    fn palace_graph_has_halls() {
        let graph = PalaceGraph::build(1, 3, 0, 0);
        // room_0_1 should have halls to room_0_0 (backward) and room_0_2 (forward)
        let neighbors = graph.get_neighbors("room_0_1");
        let halls: Vec<_> = neighbors.iter().filter(|(e, _)| e.relation_type == "HALL").collect();
        assert_eq!(halls.len(), 2, "room_0_1 should have halls to room_0_0 and room_0_2");
        // Also check room_0_0 has a hall to room_0_1
        let n0 = graph.get_neighbors("room_0_0");
        let halls0: Vec<_> = n0.iter().filter(|(e, _)| e.relation_type == "HALL").collect();
        assert!(!halls0.is_empty(), "room_0_0 should have hall edges");
    }

    #[test]
    fn benchmark_result_display() {
        let result = BenchmarkResult {
            name: "test".to_string(),
            num_nodes: 100,
            num_edges: 200,
            elapsed_ms: 1.234,
            path_found: true,
            path_length: 5,
            nodes_expanded: 20,
            iterations: 25,
        };
        let s = format!("{result}");
        assert!(s.contains("test"));
        assert!(s.contains("100"));
        assert!(s.contains("1.234"));
    }

    #[test]
    fn palace_graph_deterministic() {
        // Building the same graph twice should produce identical embeddings
        let g1 = PalaceGraph::build(2, 2, 1, 1);
        let g2 = PalaceGraph::build(2, 2, 1, 1);
        let n1 = g1.get_node("wing_0").unwrap();
        let n2 = g2.get_node("wing_0").unwrap();
        assert_eq!(n1.embedding, n2.embedding);
    }
}
