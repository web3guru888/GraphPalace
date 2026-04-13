//! Pheromone Navigation
//!
//! Demonstrates depositing pheromones along a path and navigating
//! with Semantic A* using the composite cost model.
//!
//! Note: This example references GraphPalace crate APIs. It compiles
//! against the types but requires a live Kuzu backend to execute.

use gp_core::config::PheromoneConfig;
use gp_pathfinding::{CostWeights, SemanticAStar};
use gp_stigmergy::{DecayEngine, PheromoneManager, RewardCalculator};

fn main() {
    // Configure pheromone system
    let pheromone_config = PheromoneConfig::default();
    println!("Pheromone decay rates:");
    println!("  Exploitation: {} (half-life ~35 cycles)", pheromone_config.exploitation_decay);
    println!("  Exploration:  {} (half-life ~14 cycles)", pheromone_config.exploration_decay);
    println!("  Success:      {} (half-life ~69 cycles)", pheromone_config.success_decay);
    println!("  Traversal:    {} (half-life ~23 cycles)", pheromone_config.traversal_decay);
    println!("  Recency:      {} (half-life ~7 cycles)", pheromone_config.recency_decay);

    // Create managers
    let mut manager = PheromoneManager::new(pheromone_config.clone());
    let decay_engine = DecayEngine::new(pheromone_config);
    let reward_calc = RewardCalculator::new(1.0); // base reward

    // Simulate a successful 3-edge path
    let path_edges = vec!["e-01", "e-02", "e-03"];
    println!("\nDepositing pheromones on successful path:");
    for (i, edge_id) in path_edges.iter().enumerate() {
        let position_weight = 1.0 - (i as f64 / path_edges.len() as f64);
        let reward = reward_calc.compute_reward(position_weight);
        println!("  Edge {}: position_weight={:.2}, reward={:.2}", edge_id, position_weight, reward);
    }

    // Show how pheromones decay over time
    println!("\nPheromone decay simulation (success, starting at 1.0):");
    let mut value = 1.0;
    for cycle in 0..10 {
        println!("  Cycle {}: {:.4}", cycle * 10, value);
        for _ in 0..10 {
            value = decay_engine.decay(value, 0.01); // success rate
        }
    }

    // Configure Semantic A*
    let weights = CostWeights::default(); // 0.4 semantic, 0.3 pheromone, 0.3 structural
    let astar = SemanticAStar::new(10_000, weights.clone());
    println!("\nSemantic A* configured:");
    println!("  Max iterations: {}", astar.max_iterations());
    println!("  Weights: semantic={}, pheromone={}, structural={}",
        weights.semantic, weights.pheromone, weights.structural);

    // Show context-adaptive weights
    let hypothesis_weights = CostWeights::hypothesis_testing();
    println!("\nHypothesis testing mode:");
    println!("  Weights: semantic={}, pheromone={}, structural={}",
        hypothesis_weights.semantic, hypothesis_weights.pheromone, hypothesis_weights.structural);

    println!("\nReady to navigate the palace!");
}
