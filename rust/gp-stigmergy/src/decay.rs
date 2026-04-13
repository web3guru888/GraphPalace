//! Pheromone decay functions (spec §4.2).
//!
//! Exponential decay model: `τ(t+1) = τ(t) × (1 - ρ)`
//!
//! Each pheromone type has a different decay rate `ρ`, configured via
//! [`PheromoneConfig`]. Faster-decaying pheromones (e.g., recency at ρ=0.10)
//! track recent activity, while slower-decaying ones (e.g., success at ρ=0.01)
//! accumulate long-term knowledge.

use gp_core::config::PheromoneConfig;
use gp_core::types::{EdgePheromones, NodePheromones};

/// Minimum pheromone threshold. Values below this are clamped to zero
/// to avoid denormalized float accumulation.
pub const PHEROMONE_FLOOR: f64 = 1e-9;

/// Apply one step of exponential decay.
///
/// `τ(t+1) = τ(t) × (1 - ρ)`
///
/// Returns 0.0 if the result falls below [`PHEROMONE_FLOOR`].
///
/// # Arguments
/// - `current`: current pheromone strength τ(t)
/// - `rate`: decay rate ρ ∈ [0, 1]
#[inline]
pub fn decay(current: f64, rate: f64) -> f64 {
    let next = current * (1.0 - rate);
    if next < PHEROMONE_FLOOR {
        0.0
    } else {
        next
    }
}

/// Decay all pheromone fields on a node (exploitation + exploration).
pub fn decay_node_pheromones(pheromones: &mut NodePheromones, config: &PheromoneConfig) {
    pheromones.exploitation = decay(pheromones.exploitation, config.exploitation_decay);
    pheromones.exploration = decay(pheromones.exploration, config.exploration_decay);
}

/// Decay all pheromone fields on an edge (success + traversal + recency).
pub fn decay_edge_pheromones(pheromones: &mut EdgePheromones, config: &PheromoneConfig) {
    pheromones.success = decay(pheromones.success, config.success_decay);
    pheromones.traversal = decay(pheromones.traversal, config.traversal_decay);
    pheromones.recency = decay(pheromones.recency, config.recency_decay);
}

/// Apply one decay step to a slice of node pheromones.
pub fn bulk_decay_nodes(nodes: &mut [NodePheromones], config: &PheromoneConfig) {
    for node in nodes.iter_mut() {
        decay_node_pheromones(node, config);
    }
}

/// Apply one decay step to a slice of edge pheromones.
pub fn bulk_decay_edges(edges: &mut [EdgePheromones], config: &PheromoneConfig) {
    for edge in edges.iter_mut() {
        decay_edge_pheromones(edge, config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PheromoneConfig {
        PheromoneConfig::default()
    }

    // ─── Single decay step ────────────────────────────────────────────────

    #[test]
    fn test_decay_single_step() {
        // τ × (1 - ρ) = 1.0 × (1 - 0.02) = 0.98
        let result = decay(1.0, 0.02);
        assert!((result - 0.98).abs() < 1e-12);
    }

    #[test]
    fn test_decay_with_zero_rate() {
        // No decay: τ × (1 - 0) = τ
        let result = decay(0.5, 0.0);
        assert!((result - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_decay_with_full_rate() {
        // Complete decay: τ × (1 - 1) = 0
        let result = decay(0.5, 1.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_decay_from_zero() {
        let result = decay(0.0, 0.05);
        assert_eq!(result, 0.0);
    }

    // ─── Multiple decay steps ─────────────────────────────────────────────

    #[test]
    fn test_multiple_decay_steps() {
        // After n steps: τ(n) = τ(0) × (1 - ρ)^n
        let mut val = 1.0;
        let rate = 0.10;
        for _ in 0..10 {
            val = decay(val, rate);
        }
        let expected = 1.0_f64 * (1.0 - rate).powi(10);
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_half_life_recency() {
        // Recency decay 0.10 → half-life ~7 cycles
        // (1-0.10)^7 = 0.9^7 ≈ 0.4783
        let mut val = 1.0;
        for _ in 0..7 {
            val = decay(val, 0.10);
        }
        assert!(val < 0.5, "After 7 cycles of ρ=0.10, should be below 0.5, got {val}");
    }

    #[test]
    fn test_half_life_exploitation() {
        // Exploitation decay 0.02 → half-life ~35 cycles
        // (1-0.02)^35 = 0.98^35 ≈ 0.4935
        let mut val = 1.0;
        for _ in 0..35 {
            val = decay(val, 0.02);
        }
        assert!(val < 0.5, "After 35 cycles of ρ=0.02, should be below 0.5, got {val}");
    }

    // ─── Near-zero threshold ──────────────────────────────────────────────

    #[test]
    fn test_near_zero_clamped() {
        // A very small value should be clamped to 0.0
        let result = decay(1e-10, 0.5);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_floor_threshold_exact() {
        // Just above the floor should survive
        let result = decay(1e-8, 0.01);
        assert!(result > 0.0);

        // Just below the floor should clamp
        let result2 = decay(1e-10, 0.01);
        assert_eq!(result2, 0.0);
    }

    #[test]
    fn test_decay_converges_to_zero() {
        let mut val = 1.0;
        for _ in 0..1000 {
            val = decay(val, 0.10);
        }
        assert_eq!(val, 0.0, "After 1000 steps of ρ=0.10, should be exactly 0.0");
    }

    // ─── Node pheromone decay ─────────────────────────────────────────────

    #[test]
    fn test_decay_node_pheromones() {
        let config = default_config();
        let mut p = NodePheromones {
            exploitation: 1.0,
            exploration: 1.0,
        };
        decay_node_pheromones(&mut p, &config);
        assert!((p.exploitation - 0.98).abs() < 1e-12); // 1.0 × (1-0.02)
        assert!((p.exploration - 0.95).abs() < 1e-12); // 1.0 × (1-0.05)
    }

    #[test]
    fn test_decay_node_pheromones_preserves_zero() {
        let config = default_config();
        let mut p = NodePheromones::default();
        decay_node_pheromones(&mut p, &config);
        assert_eq!(p.exploitation, 0.0);
        assert_eq!(p.exploration, 0.0);
    }

    // ─── Edge pheromone decay ─────────────────────────────────────────────

    #[test]
    fn test_decay_edge_pheromones() {
        let config = default_config();
        let mut p = EdgePheromones {
            success: 1.0,
            traversal: 1.0,
            recency: 1.0,
        };
        decay_edge_pheromones(&mut p, &config);
        assert!((p.success - 0.99).abs() < 1e-12); // 1.0 × (1-0.01)
        assert!((p.traversal - 0.97).abs() < 1e-12); // 1.0 × (1-0.03)
        assert!((p.recency - 0.90).abs() < 1e-12); // 1.0 × (1-0.10)
    }

    #[test]
    fn test_decay_edge_pheromones_preserves_zero() {
        let config = default_config();
        let mut p = EdgePheromones::default();
        decay_edge_pheromones(&mut p, &config);
        assert_eq!(p.success, 0.0);
        assert_eq!(p.traversal, 0.0);
        assert_eq!(p.recency, 0.0);
    }

    // ─── Bulk decay ───────────────────────────────────────────────────────

    #[test]
    fn test_bulk_decay_nodes() {
        let config = default_config();
        let mut nodes = vec![
            NodePheromones { exploitation: 1.0, exploration: 0.5 },
            NodePheromones { exploitation: 0.8, exploration: 0.3 },
            NodePheromones { exploitation: 0.0, exploration: 0.0 },
        ];
        bulk_decay_nodes(&mut nodes, &config);

        assert!((nodes[0].exploitation - 0.98).abs() < 1e-12);
        assert!((nodes[0].exploration - 0.475).abs() < 1e-12);
        assert!((nodes[1].exploitation - 0.784).abs() < 1e-12);
        assert!((nodes[1].exploration - 0.285).abs() < 1e-12);
        assert_eq!(nodes[2].exploitation, 0.0);
        assert_eq!(nodes[2].exploration, 0.0);
    }

    #[test]
    fn test_bulk_decay_edges() {
        let config = default_config();
        let mut edges = vec![
            EdgePheromones { success: 1.0, traversal: 1.0, recency: 1.0 },
            EdgePheromones { success: 0.5, traversal: 0.0, recency: 0.2 },
        ];
        bulk_decay_edges(&mut edges, &config);

        assert!((edges[0].success - 0.99).abs() < 1e-12);
        assert!((edges[0].traversal - 0.97).abs() < 1e-12);
        assert!((edges[0].recency - 0.90).abs() < 1e-12);
        assert!((edges[1].success - 0.495).abs() < 1e-12);
        assert_eq!(edges[1].traversal, 0.0);
        assert!((edges[1].recency - 0.18).abs() < 1e-12);
    }

    #[test]
    fn test_bulk_decay_empty_slices() {
        let config = default_config();
        let mut nodes: Vec<NodePheromones> = vec![];
        let mut edges: Vec<EdgePheromones> = vec![];
        bulk_decay_nodes(&mut nodes, &config);
        bulk_decay_edges(&mut edges, &config);
        // No panic, no-op.
    }
}
