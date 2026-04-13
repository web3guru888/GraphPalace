//! Integration tests for the full stigmergy cycle: deposit → decay → recompute.

use gp_core::config::PheromoneConfig;
use gp_core::types::{EdgeCost, EdgePheromones, NodePheromones};
use gp_stigmergy::{
    bulk_decay_edges, bulk_decay_nodes, decay, deposit_exploration, deposit_path_success,
    pheromone_factor, recompute_edge_cost,
};

fn default_config() -> PheromoneConfig {
    PheromoneConfig::default()
}

// ─── Full cycle: deposit → decay → recompute ────────────────────────────

#[test]
fn full_cycle_deposit_decay_recompute() {
    let config = default_config();
    let mut edges = vec![EdgePheromones::default(); 3];
    let mut nodes = vec![NodePheromones::default(); 4];
    let mut costs = vec![EdgeCost::new(0.5); 3];

    // Step 1: Deposit success on the path
    deposit_path_success(&mut edges, &mut nodes, 1.0);

    // Verify deposits happened
    assert!(edges[0].success > 0.0);
    assert!(edges[1].success > 0.0);
    assert!(edges[2].success > 0.0);
    assert!(nodes[0].exploitation > 0.0);

    // Step 2: Recompute costs — should decrease from base
    for (cost, edge) in costs.iter_mut().zip(edges.iter()) {
        recompute_edge_cost(cost, edge);
    }
    for cost in &costs {
        assert!(cost.current_cost < cost.base_cost, "Cost should decrease after deposit");
    }

    // Step 3: Decay
    bulk_decay_edges(&mut edges, &config);
    bulk_decay_nodes(&mut nodes, &config);

    // Step 4: Recompute again — costs should increase slightly toward base
    let costs_before: Vec<f64> = costs.iter().map(|c| c.current_cost).collect();
    for (cost, edge) in costs.iter_mut().zip(edges.iter()) {
        recompute_edge_cost(cost, edge);
    }
    for (i, cost) in costs.iter().enumerate() {
        assert!(
            cost.current_cost >= costs_before[i] - 1e-10,
            "Cost should increase after decay"
        );
    }
}

#[test]
fn multiple_decay_cycles_converge_toward_zero() {
    let config = default_config();
    let mut edges = vec![EdgePheromones {
        success: 1.0,
        traversal: 1.0,
        recency: 1.0,
    }];
    let mut nodes = vec![NodePheromones {
        exploitation: 1.0,
        exploration: 1.0,
    }];

    // Run 2100 decay cycles (success decay ρ=0.01 is slowest: 0.99^2100 ≈ 7e-10 < floor 1e-9)
    for _ in 0..2100 {
        bulk_decay_edges(&mut edges, &config);
        bulk_decay_nodes(&mut nodes, &config);
    }

    // All pheromones should be exactly zero (floored by PHEROMONE_FLOOR)
    assert_eq!(edges[0].success, 0.0);
    assert_eq!(edges[0].traversal, 0.0);
    assert_eq!(edges[0].recency, 0.0);
    assert_eq!(nodes[0].exploitation, 0.0);
    assert_eq!(nodes[0].exploration, 0.0);
}

#[test]
fn deposit_multi_edge_path_preserves_ordering() {
    let mut edges = vec![EdgePheromones::default(); 5];
    let mut nodes = vec![NodePheromones::default(); 6];

    deposit_path_success(&mut edges, &mut nodes, 1.0);

    // Position weighting: first edge should have highest success
    for i in 1..edges.len() {
        assert!(
            edges[i - 1].success > edges[i].success,
            "Edge {} should have higher success than edge {}: {} vs {}",
            i - 1, i, edges[i - 1].success, edges[i].success
        );
    }

    // After decay, ordering should be preserved
    let config = default_config();
    bulk_decay_edges(&mut edges, &config);

    for i in 1..edges.len() {
        assert!(
            edges[i - 1].success >= edges[i].success,
            "Ordering preserved after decay"
        );
    }
}

#[test]
fn exploration_deposit_does_not_affect_edges() {
    let edges = vec![EdgePheromones::default(); 2];
    let mut node = NodePheromones::default();

    // Deposit exploration on the node
    deposit_exploration(&mut node);

    // Edges should be unchanged
    for edge in &edges {
        assert_eq!(edge.success, 0.0);
        assert_eq!(edge.traversal, 0.0);
        assert_eq!(edge.recency, 0.0);
    }

    // Node should have exploration but not exploitation
    assert!(node.exploration > 0.0);
    assert_eq!(node.exploitation, 0.0);
}

#[test]
fn ten_decay_cycles_after_deposit_costs_near_base() {
    let config = default_config();
    let mut edges = vec![EdgePheromones::default()];
    let mut nodes = vec![NodePheromones::default(); 2];
    let mut costs = vec![EdgeCost::new(0.5)];

    // Deposit
    deposit_path_success(&mut edges, &mut nodes, 0.5);

    // 50 decay cycles
    for _ in 0..50 {
        bulk_decay_edges(&mut edges, &config);
    }

    // Recompute
    recompute_edge_cost(&mut costs[0], &edges[0]);

    // After 50 cycles of success decay (ρ=0.01), pheromone = 0.5 × 0.99^50 ≈ 0.303
    // After 50 cycles of recency decay (ρ=0.10), pheromone ≈ 0 (decayed away)
    // After 50 cycles of traversal decay (ρ=0.03), pheromone = 0.1 × 0.97^50 ≈ 0.022
    // Cost should be closer to base than right after deposit
    assert!(costs[0].current_cost > 0.4, "Cost should be near base: {}", costs[0].current_cost);
}

#[test]
fn heavy_deposit_overcomes_decay() {
    let config = default_config();
    let mut edges = vec![EdgePheromones::default()];
    let mut nodes = vec![NodePheromones::default(); 2];

    // Deposit heavily, then decay once
    for _ in 0..10 {
        deposit_path_success(&mut edges, &mut nodes, 2.0);
    }
    let success_before_decay = edges[0].success;
    assert!(success_before_decay > 10.0);

    bulk_decay_edges(&mut edges, &config);

    // Still very high
    assert!(edges[0].success > 9.0, "Heavy deposit should survive one decay: {}", edges[0].success);
}

#[test]
fn round_trip_deposit_decay_deposit_accumulates() {
    let config = default_config();
    let mut edges = vec![EdgePheromones::default()];
    let mut nodes = vec![NodePheromones::default(); 2];
    let mut cost = EdgeCost::new(1.0);

    // First deposit
    deposit_path_success(&mut edges, &mut nodes, 1.0);
    let first_success = edges[0].success;

    // Decay 5 times
    for _ in 0..5 {
        bulk_decay_edges(&mut edges, &config);
    }
    let decayed_success = edges[0].success;
    assert!(decayed_success < first_success);

    // Second deposit
    deposit_path_success(&mut edges, &mut nodes, 1.0);
    let accumulated_success = edges[0].success;

    // Should be higher than just one deposit (decayed + fresh)
    assert!(accumulated_success > 1.0, "Should accumulate: {}", accumulated_success);

    // Recompute cost
    recompute_edge_cost(&mut cost, &edges[0]);
    assert!(cost.current_cost < cost.base_cost);
}

#[test]
fn empty_path_deposit_is_noop() {
    let mut edges: Vec<EdgePheromones> = vec![];
    let mut nodes: Vec<NodePheromones> = vec![];
    deposit_path_success(&mut edges, &mut nodes, 1.0);
    // No panic, no state change
}

#[test]
fn large_path_deposit_and_decay() {
    let config = default_config();
    let n = 100;
    let mut edges = vec![EdgePheromones::default(); n];
    let mut nodes = vec![NodePheromones::default(); n + 1];
    let mut costs: Vec<EdgeCost> = (0..n).map(|_| EdgeCost::new(0.5)).collect();

    // Deposit
    deposit_path_success(&mut edges, &mut nodes, 1.0);

    // Verify position weighting
    assert!(edges[0].success > edges[99].success);

    // First edge: reward = 1.0 × (1 - 0/100) = 1.0
    assert!((edges[0].success - 1.0).abs() < 1e-10);
    // Last edge: reward = 1.0 × (1 - 99/100) = 0.01
    assert!((edges[99].success - 0.01).abs() < 1e-10);

    // Decay 10 cycles
    for _ in 0..10 {
        bulk_decay_edges(&mut edges, &config);
    }

    // Recompute all costs
    for (cost, edge) in costs.iter_mut().zip(edges.iter()) {
        recompute_edge_cost(cost, edge);
    }

    // First edge should still have lower cost (more pheromone remaining)
    assert!(costs[0].current_cost <= costs[99].current_cost);
}

#[test]
fn decay_floor_prevents_denormalized_accumulation() {
    // Verify that after enough decay cycles, values hit exact zero (not tiny denormalized floats)
    let mut val = 0.001; // Just above the initial threshold
    for _ in 0..10000 {
        val = decay(val, 0.10);
    }
    assert_eq!(val, 0.0, "Should be exactly zero after many decay steps");
}

#[test]
fn pheromone_factor_tracks_deposit() {
    let mut edge = EdgePheromones::default();
    assert_eq!(pheromone_factor(&edge), 0.0);

    // Deposit some pheromones
    edge.success = 0.5;
    edge.recency = 1.0;
    edge.traversal = 0.1;

    let factor = pheromone_factor(&edge);
    // 0.5 × 0.5 + 0.3 × 1.0 + 0.2 × 0.1 = 0.25 + 0.30 + 0.02 = 0.57
    assert!((factor - 0.57).abs() < 1e-10);
}

#[test]
fn recompute_after_deposit_matches_formula() {
    let mut edge_p = EdgePheromones::default();
    let mut cost = EdgeCost::new(1.0);
    let mut nodes = vec![NodePheromones::default(); 2];

    // Deposit: success = 1.0, traversal = 0.1, recency = 1.0
    deposit_path_success(std::slice::from_mut(&mut edge_p), &mut nodes, 1.0);

    recompute_edge_cost(&mut cost, &edge_p);

    // factor = 0.5×min(1.0,1) + 0.3×min(1.0,1) + 0.2×min(0.1,1) = 0.5 + 0.3 + 0.02 = 0.82
    // current = 1.0 × (1 - 0.82 × 0.5) = 1.0 × 0.59 = 0.59
    assert!((cost.current_cost - 0.59).abs() < 1e-10, "Got {}", cost.current_cost);
}
