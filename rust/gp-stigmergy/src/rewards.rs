//! Pheromone reward deposits (spec §4.3).
//!
//! After a successful search, pheromones are deposited along the path
//! to reinforce good connections and mark valuable locations.
//!
//! Position weighting ensures earlier edges in the path receive more
//! reward, biasing future searches toward the beginning of successful routes.

use gp_core::types::{EdgePheromones, NodePheromones};

/// Traversal increment per edge in a successful path.
pub const TRAVERSAL_INCREMENT: f64 = 0.1;

/// Recency value deposited on each edge (always set to maximum).
pub const RECENCY_VALUE: f64 = 1.0;

/// Exploitation increment for each node on a successful path.
pub const EXPLOITATION_INCREMENT: f64 = 0.2;

/// Exploration increment when a node is explored (visited during search).
pub const EXPLORATION_INCREMENT: f64 = 0.3;

/// Deposit pheromones along a successful search path.
///
/// For a path of length `n`:
/// - Each **edge** at position `i` receives:
///   - `success += base_reward × (1.0 - i / n)` — position-weighted reward
///   - `traversal += 0.1` — frequency signal
///   - `recency = 1.0` — freshness signal (reset, not additive)
/// - Each **node** receives:
///   - `exploitation += 0.2` — value signal
///
/// The `edges` and `nodes` slices must correspond to the same path.
/// A path with N nodes typically has N-1 edges, but this function
/// operates on whatever slices are provided.
///
/// # Arguments
/// - `edges`: mutable slice of edge pheromones along the path (in path order)
/// - `nodes`: mutable slice of node pheromones along the path
/// - `base_reward`: base reward amount for the success pheromone
pub fn deposit_path_success(
    edges: &mut [EdgePheromones],
    nodes: &mut [NodePheromones],
    base_reward: f64,
) {
    let path_len = edges.len() as f64;

    // Deposit on edges with position-weighted success reward
    if path_len > 0.0 {
        for (i, edge) in edges.iter_mut().enumerate() {
            let position_weight = 1.0 - (i as f64 / path_len);
            edge.success += base_reward * position_weight;
            edge.traversal += TRAVERSAL_INCREMENT;
            edge.recency = RECENCY_VALUE;
        }
    }

    // Deposit on nodes
    for node in nodes.iter_mut() {
        node.exploitation += EXPLOITATION_INCREMENT;
    }
}

/// Deposit exploration pheromone on a node that has been visited during search.
///
/// This signals "already searched — try elsewhere" to future agents,
/// promoting diversity in the search swarm.
pub fn deposit_exploration(pheromones: &mut NodePheromones) {
    pheromones.exploration += EXPLORATION_INCREMENT;
}

/// Clamp a value to a ceiling. Returns the ceiling if value exceeds it.
#[inline]
pub fn clamp_to_ceiling(value: f64, ceiling: f64) -> f64 {
    if value > ceiling { ceiling } else { value }
}

/// Clamp edge pheromones to their saturation ceilings.
pub fn apply_edge_saturation(edge: &mut EdgePheromones, config: &gp_core::config::PheromoneConfig) {
    edge.success = clamp_to_ceiling(edge.success, config.success_max);
    edge.traversal = clamp_to_ceiling(edge.traversal, config.traversal_max);
    edge.recency = clamp_to_ceiling(edge.recency, config.recency_max);
}

/// Clamp node pheromones to their saturation ceilings.
pub fn apply_node_saturation(node: &mut NodePheromones, config: &gp_core::config::PheromoneConfig) {
    node.exploitation = clamp_to_ceiling(node.exploitation, config.exploitation_max);
    node.exploration = clamp_to_ceiling(node.exploration, config.exploration_max);
}

/// Deposit pheromones along a successful search path with saturation clamping.
///
/// Same as [`deposit_path_success`] but enforces τ_max ceilings from config.
pub fn deposit_path_success_clamped(
    edges: &mut [EdgePheromones],
    nodes: &mut [NodePheromones],
    base_reward: f64,
    config: &gp_core::config::PheromoneConfig,
) {
    deposit_path_success(edges, nodes, base_reward);
    for edge in edges.iter_mut() {
        apply_edge_saturation(edge, config);
    }
    for node in nodes.iter_mut() {
        apply_node_saturation(node, config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Path deposit ─────────────────────────────────────────────────────

    #[test]
    fn test_deposit_path_success_single_edge() {
        let mut edges = vec![EdgePheromones::default()];
        let mut nodes = vec![NodePheromones::default(), NodePheromones::default()];
        let base_reward = 1.0;

        deposit_path_success(&mut edges, &mut nodes, base_reward);

        // Single edge at position 0: weight = 1.0 - 0/1 = 1.0
        assert!((edges[0].success - 1.0).abs() < 1e-12);
        assert!((edges[0].traversal - TRAVERSAL_INCREMENT).abs() < 1e-12);
        assert!((edges[0].recency - RECENCY_VALUE).abs() < 1e-12);

        // Both nodes get exploitation
        assert!((nodes[0].exploitation - EXPLOITATION_INCREMENT).abs() < 1e-12);
        assert!((nodes[1].exploitation - EXPLOITATION_INCREMENT).abs() < 1e-12);
    }

    #[test]
    fn test_deposit_path_success_position_weighting() {
        let mut edges = vec![
            EdgePheromones::default(),
            EdgePheromones::default(),
            EdgePheromones::default(),
            EdgePheromones::default(),
        ];
        let mut nodes = vec![NodePheromones::default(); 5];
        let base_reward = 2.0;

        deposit_path_success(&mut edges, &mut nodes, base_reward);

        // Path len = 4
        // Edge 0: weight = 1.0 - 0/4 = 1.00, success = 2.0 × 1.00 = 2.0
        // Edge 1: weight = 1.0 - 1/4 = 0.75, success = 2.0 × 0.75 = 1.5
        // Edge 2: weight = 1.0 - 2/4 = 0.50, success = 2.0 × 0.50 = 1.0
        // Edge 3: weight = 1.0 - 3/4 = 0.25, success = 2.0 × 0.25 = 0.5
        assert!((edges[0].success - 2.0).abs() < 1e-12);
        assert!((edges[1].success - 1.5).abs() < 1e-12);
        assert!((edges[2].success - 1.0).abs() < 1e-12);
        assert!((edges[3].success - 0.5).abs() < 1e-12);

        // All edges get same traversal and recency
        for edge in &edges {
            assert!((edge.traversal - TRAVERSAL_INCREMENT).abs() < 1e-12);
            assert!((edge.recency - RECENCY_VALUE).abs() < 1e-12);
        }
    }

    #[test]
    fn test_deposit_is_additive() {
        let mut edges = vec![EdgePheromones {
            success: 0.5,
            traversal: 0.3,
            recency: 0.2,
        }];
        let mut nodes = vec![NodePheromones {
            exploitation: 0.1,
            exploration: 0.0,
        }];

        deposit_path_success(&mut edges, &mut nodes, 1.0);

        // success is additive: 0.5 + 1.0 × 1.0 = 1.5
        assert!((edges[0].success - 1.5).abs() < 1e-12);
        // traversal is additive: 0.3 + 0.1 = 0.4
        assert!((edges[0].traversal - 0.4).abs() < 1e-12);
        // recency is set (not additive): = 1.0
        assert!((edges[0].recency - 1.0).abs() < 1e-12);
        // exploitation is additive: 0.1 + 0.2 = 0.3
        assert!((nodes[0].exploitation - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_deposit_path_success_empty_edges() {
        let mut edges: Vec<EdgePheromones> = vec![];
        let mut nodes = vec![NodePheromones::default()];

        deposit_path_success(&mut edges, &mut nodes, 1.0);

        // Node still gets exploitation even with no edges
        assert!((nodes[0].exploitation - EXPLOITATION_INCREMENT).abs() < 1e-12);
    }

    #[test]
    fn test_deposit_path_success_empty_both() {
        let mut edges: Vec<EdgePheromones> = vec![];
        let mut nodes: Vec<NodePheromones> = vec![];

        // Should not panic
        deposit_path_success(&mut edges, &mut nodes, 1.0);
    }

    #[test]
    fn test_deposit_path_success_zero_reward() {
        let mut edges = vec![EdgePheromones::default()];
        let mut nodes = vec![NodePheromones::default()];

        deposit_path_success(&mut edges, &mut nodes, 0.0);

        // Success should be 0 (0 × weight), but traversal/recency still applied
        assert_eq!(edges[0].success, 0.0);
        assert!((edges[0].traversal - TRAVERSAL_INCREMENT).abs() < 1e-12);
        assert!((edges[0].recency - RECENCY_VALUE).abs() < 1e-12);
    }

    #[test]
    fn test_position_weight_sum() {
        // Total reward deposited should be base_reward × Σ(1 - i/n) for i in 0..n
        // = base_reward × n × (n+1)/(2n) = base_reward × (n+1)/2
        let n = 10;
        let base_reward = 1.0;
        let mut edges = vec![EdgePheromones::default(); n];
        let mut nodes = vec![NodePheromones::default(); n + 1];

        deposit_path_success(&mut edges, &mut nodes, base_reward);

        let total_success: f64 = edges.iter().map(|e| e.success).sum();
        // Σ(1 - i/10) for i=0..10 = 10 - (0+1+...+9)/10 = 10 - 45/10 = 10 - 4.5 = 5.5
        assert!((total_success - 5.5).abs() < 1e-10);
    }

    // ─── Exploration deposit ──────────────────────────────────────────────

    #[test]
    fn test_deposit_exploration() {
        let mut p = NodePheromones::default();
        deposit_exploration(&mut p);
        assert!((p.exploration - EXPLORATION_INCREMENT).abs() < 1e-12);
    }

    #[test]
    fn test_deposit_exploration_additive() {
        let mut p = NodePheromones {
            exploitation: 0.0,
            exploration: 0.5,
        };
        deposit_exploration(&mut p);
        assert!((p.exploration - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_deposit_exploration_does_not_affect_exploitation() {
        let mut p = NodePheromones {
            exploitation: 0.7,
            exploration: 0.0,
        };
        deposit_exploration(&mut p);
        assert!((p.exploitation - 0.7).abs() < 1e-12);
        assert!((p.exploration - EXPLORATION_INCREMENT).abs() < 1e-12);
    }

    // ─── Saturation ceiling (τ_max) ──────────────────────────────────────

    #[test]
    fn test_clamp_to_ceiling() {
        assert_eq!(clamp_to_ceiling(3.0, 5.0), 3.0);
        assert_eq!(clamp_to_ceiling(7.0, 5.0), 5.0);
        assert_eq!(clamp_to_ceiling(5.0, 5.0), 5.0);
        assert_eq!(clamp_to_ceiling(0.0, 5.0), 0.0);
    }

    #[test]
    fn test_apply_edge_saturation() {
        let config = gp_core::config::PheromoneConfig::default();
        let mut edge = EdgePheromones { success: 10.0, traversal: 5.0, recency: 3.0 };
        apply_edge_saturation(&mut edge, &config);
        assert_eq!(edge.success, 5.0);   // clamped from 10.0
        assert_eq!(edge.traversal, 2.0); // clamped from 5.0
        assert_eq!(edge.recency, 1.0);   // clamped from 3.0
    }

    #[test]
    fn test_apply_node_saturation() {
        let config = gp_core::config::PheromoneConfig::default();
        let mut node = NodePheromones { exploitation: 8.0, exploration: 6.0 };
        apply_node_saturation(&mut node, &config);
        assert_eq!(node.exploitation, 5.0); // clamped from 8.0
        assert_eq!(node.exploration, 3.0);  // clamped from 6.0
    }

    #[test]
    fn test_deposit_path_success_clamped() {
        let config = gp_core::config::PheromoneConfig::default();
        // Start with values near ceiling
        let mut edges = vec![EdgePheromones { success: 4.5, traversal: 1.9, recency: 0.5 }];
        let mut nodes = vec![NodePheromones { exploitation: 4.9, exploration: 0.0 }];
        deposit_path_success_clamped(&mut edges, &mut nodes, 2.0, &config);
        // success would be 4.5 + 2.0 = 6.5, clamped to 5.0
        assert!((edges[0].success - 5.0).abs() < 1e-12);
        // exploitation would be 4.9 + 0.2 = 5.1, clamped to 5.0
        assert!((nodes[0].exploitation - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_saturation_below_ceiling_unchanged() {
        let config = gp_core::config::PheromoneConfig::default();
        let mut edges = vec![EdgePheromones { success: 0.5, traversal: 0.1, recency: 0.1 }];
        let mut nodes = vec![NodePheromones { exploitation: 0.1, exploration: 0.0 }];
        deposit_path_success_clamped(&mut edges, &mut nodes, 1.0, &config);
        // All below ceiling, should be same as unclamped
        assert!((edges[0].success - 1.5).abs() < 1e-12);
    }
}
