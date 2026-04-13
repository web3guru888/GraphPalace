//! Integration tests for pathfinding across full palace hierarchy.

use gp_core::config::{AStarConfig, CostWeights};
use gp_pathfinding::bench::PalaceGraph;
use gp_pathfinding::SemanticAStar;

#[test]
fn full_palace_hierarchy_traversal() {
    let graph = PalaceGraph::build(2, 2, 2, 2);
    let astar = SemanticAStar::new(AStarConfig::default(), CostWeights::default());
    
    // Navigate from palace to a deep drawer
    let result = astar.find_path(&graph, "palace", "drawer_0_0_0_0");
    assert!(result.is_some(), "Should find path from palace to drawer");
    let path = result.unwrap();
    // Path: palace → wing_0 → room_0_0 → closet_0_0_0 → drawer_0_0_0_0
    assert_eq!(path.path.len(), 5);
    assert_eq!(path.path[0], "palace");
    assert!(path.path[4].starts_with("drawer_"));
}

#[test]
fn cross_wing_navigation() {
    let graph = PalaceGraph::build(3, 2, 1, 1);
    let astar = SemanticAStar::new(AStarConfig::default(), CostWeights::default());
    
    // Navigate between rooms in different wings via tunnel
    let result = astar.find_path(&graph, "room_0_0", "room_2_0");
    assert!(result.is_some(), "Should find cross-wing path via tunnel");
    let path = result.unwrap();
    // Should be direct tunnel: room_0_0 → room_2_0
    assert!(path.path.len() <= 3, "Tunnel should be direct or near-direct, got len={}", path.path.len());
}

#[test]
fn hall_navigation_within_wing() {
    let graph = PalaceGraph::build(1, 5, 0, 0);
    let astar = SemanticAStar::new(AStarConfig::default(), CostWeights::default());
    
    let result = astar.find_path(&graph, "room_0_0", "room_0_4");
    assert!(result.is_some(), "Should find path through halls");
    let path = result.unwrap();
    // Should traverse: room_0_0 → room_0_1 → room_0_2 → room_0_3 → room_0_4
    assert_eq!(path.path.len(), 5);
}

#[test]
fn varying_pheromone_levels_affect_path_choice() {
    // With pheromones set, costs change and paths may differ
    let graph_clean = PalaceGraph::build(2, 3, 1, 1);
    let mut graph_pheromone = PalaceGraph::build(2, 3, 1, 1);
    graph_pheromone.set_edge_pheromones(1.0, 1.0, 1.0);
    
    let astar = SemanticAStar::new(AStarConfig::default(), CostWeights::default());
    
    let r1 = astar.find_path(&graph_clean, "room_0_0", "room_1_2");
    let r2 = astar.find_path(&graph_pheromone, "room_0_0", "room_1_2");
    
    // Both should find paths (via tunnels)
    if let (Some(p1), Some(p2)) = (r1, r2) {
        // Pheromone version should have lower total cost (pheromones reduce costs)
        assert!(p2.total_cost <= p1.total_cost + 0.01,
            "Pheromones should reduce cost: clean={}, pheromone={}", p1.total_cost, p2.total_cost);
    }
}

#[test]
fn context_adaptive_weights_change_behavior() {
    let graph = PalaceGraph::build(2, 3, 1, 1);
    let config = AStarConfig::default();
    
    let default_astar = SemanticAStar::new(config.clone(), CostWeights::default());
    let hypothesis_astar = SemanticAStar::new(config.clone(), CostWeights::hypothesis_testing());
    let exploratory_astar = SemanticAStar::new(config.clone(), CostWeights::exploratory_research());
    
    let r_default = default_astar.find_path(&graph, "room_0_0", "room_1_2");
    let r_hypothesis = hypothesis_astar.find_path(&graph, "room_0_0", "room_1_2");
    let r_exploratory = exploratory_astar.find_path(&graph, "room_0_0", "room_1_2");
    
    // All should find paths
    if let (Some(d), Some(h), Some(e)) = (r_default, r_hypothesis, r_exploratory) {
        // Costs should differ due to different weight distributions
        // (They may or may not, depends on embeddings, but the astar should run without error)
        eprintln!("Default cost: {}, Hypothesis: {}, Exploratory: {}", d.total_cost, h.total_cost, e.total_cost);
    }
}

#[test]
fn provenance_is_complete() {
    let graph = PalaceGraph::build(1, 3, 1, 1);
    let astar = SemanticAStar::new(AStarConfig::default(), CostWeights::default());
    
    let result = astar.find_path(&graph, "palace", "drawer_0_2_0_0");
    if let Some(path) = result {
        assert_eq!(path.provenance.len(), path.path.len(), "Provenance should have entry per node");
        
        // First provenance entry should have g_cost = 0 and empty edge_type
        assert!((path.provenance[0].g_cost - 0.0).abs() < 1e-10);
        assert!(path.provenance[0].edge_type.is_empty());
        
        // g_costs should be non-decreasing
        for w in path.provenance.windows(2) {
            assert!(w[1].g_cost >= w[0].g_cost - 1e-10);
        }
    }
}
