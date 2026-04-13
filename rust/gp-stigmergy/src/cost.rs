//! Edge cost recomputation from pheromone fields (spec §4.4).
//!
//! Pheromone levels modulate the effective traversal cost of an edge.
//! Higher pheromone levels reduce cost, biasing search toward
//! well-reinforced paths.
//!
//! Formula:
//! ```text
//! pheromone_factor = 0.5 × min(success, 1) + 0.3 × min(recency, 1) + 0.2 × min(traversal, 1)
//! current_cost = clamp(base_cost × (1 - pheromone_factor × 0.5), 0.0, 10.0)
//! ```

use gp_core::types::{EdgeCost, EdgePheromones};

/// Weight of the success pheromone in the composite pheromone factor.
pub const SUCCESS_WEIGHT: f64 = 0.5;

/// Weight of the recency pheromone in the composite pheromone factor.
pub const RECENCY_WEIGHT: f64 = 0.3;

/// Weight of the traversal pheromone in the composite pheromone factor.
pub const TRAVERSAL_WEIGHT: f64 = 0.2;

/// Maximum scaling effect of pheromones on cost (50% reduction at maximum).
pub const MAX_PHEROMONE_DISCOUNT: f64 = 0.5;

/// Minimum allowable edge cost.
pub const MIN_COST: f64 = 0.0;

/// Maximum allowable edge cost.
pub const MAX_COST: f64 = 10.0;

/// Compute the composite pheromone factor from edge pheromones.
///
/// Each pheromone contribution is clamped to [0, 1] before weighting:
/// `factor = 0.5 × min(success, 1) + 0.3 × min(recency, 1) + 0.2 × min(traversal, 1)`
///
/// Returns a value in [0, 1].
pub fn pheromone_factor(pheromones: &EdgePheromones) -> f64 {
    SUCCESS_WEIGHT * pheromones.success.min(1.0)
        + RECENCY_WEIGHT * pheromones.recency.min(1.0)
        + TRAVERSAL_WEIGHT * pheromones.traversal.min(1.0)
}

/// Recompute the current cost of an edge based on its pheromone levels.
///
/// `current_cost = clamp(base_cost × (1 - pheromone_factor × 0.5), 0.0, 10.0)`
///
/// At maximum pheromones (all fields ≥ 1.0), the cost is reduced by 50%.
/// At zero pheromones, `current_cost == base_cost`.
pub fn recompute_edge_cost(edge_cost: &mut EdgeCost, pheromones: &EdgePheromones) {
    let factor = pheromone_factor(pheromones);
    let discount = factor * MAX_PHEROMONE_DISCOUNT;
    edge_cost.current_cost = (edge_cost.base_cost * (1.0 - discount)).clamp(MIN_COST, MAX_COST);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Pheromone factor ─────────────────────────────────────────────────

    #[test]
    fn test_pheromone_factor_zero() {
        let p = EdgePheromones::default();
        assert_eq!(pheromone_factor(&p), 0.0);
    }

    #[test]
    fn test_pheromone_factor_all_ones() {
        let p = EdgePheromones {
            success: 1.0,
            traversal: 1.0,
            recency: 1.0,
        };
        let factor = pheromone_factor(&p);
        // 0.5 × 1 + 0.3 × 1 + 0.2 × 1 = 1.0
        assert!((factor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pheromone_factor_clamped_above_one() {
        let p = EdgePheromones {
            success: 5.0,   // clamped to 1.0
            traversal: 3.0, // clamped to 1.0
            recency: 2.0,   // clamped to 1.0
        };
        let factor = pheromone_factor(&p);
        // Still 0.5 + 0.3 + 0.2 = 1.0
        assert!((factor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pheromone_factor_partial() {
        let p = EdgePheromones {
            success: 0.8,
            traversal: 0.0,
            recency: 0.5,
        };
        let factor = pheromone_factor(&p);
        // 0.5 × 0.8 + 0.3 × 0.5 + 0.2 × 0.0 = 0.40 + 0.15 + 0.0 = 0.55
        assert!((factor - 0.55).abs() < 1e-12);
    }

    #[test]
    fn test_pheromone_factor_weights_sum_to_one() {
        assert!((SUCCESS_WEIGHT + RECENCY_WEIGHT + TRAVERSAL_WEIGHT - 1.0).abs() < 1e-12);
    }

    // ─── Edge cost recomputation ──────────────────────────────────────────

    #[test]
    fn test_recompute_zero_pheromones() {
        let mut cost = EdgeCost::new(0.5);
        let p = EdgePheromones::default();
        recompute_edge_cost(&mut cost, &p);
        // No pheromones → current_cost == base_cost
        assert!((cost.current_cost - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_recompute_max_pheromones() {
        let mut cost = EdgeCost::new(1.0);
        let p = EdgePheromones {
            success: 1.0,
            traversal: 1.0,
            recency: 1.0,
        };
        recompute_edge_cost(&mut cost, &p);
        // factor = 1.0, discount = 0.5 → current = 1.0 × 0.5 = 0.5
        assert!((cost.current_cost - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_recompute_partial_pheromones() {
        let mut cost = EdgeCost::new(2.0);
        let p = EdgePheromones {
            success: 0.5,
            traversal: 0.5,
            recency: 0.5,
        };
        recompute_edge_cost(&mut cost, &p);
        // factor = 0.5×0.5 + 0.3×0.5 + 0.2×0.5 = 0.25 + 0.15 + 0.10 = 0.50
        // discount = 0.50 × 0.5 = 0.25
        // current = 2.0 × 0.75 = 1.5
        assert!((cost.current_cost - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_recompute_clamp_minimum() {
        let mut cost = EdgeCost::new(0.0);
        let p = EdgePheromones {
            success: 1.0,
            traversal: 1.0,
            recency: 1.0,
        };
        recompute_edge_cost(&mut cost, &p);
        assert_eq!(cost.current_cost, 0.0);
    }

    #[test]
    fn test_recompute_clamp_maximum() {
        let mut cost = EdgeCost::new(25.0);
        let p = EdgePheromones::default();
        recompute_edge_cost(&mut cost, &p);
        // 25.0 × 1.0 = 25.0 → clamped to 10.0
        assert!((cost.current_cost - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_recompute_preserves_base_cost() {
        let mut cost = EdgeCost::new(0.7);
        let p = EdgePheromones {
            success: 0.5,
            traversal: 0.5,
            recency: 0.5,
        };
        recompute_edge_cost(&mut cost, &p);
        // base_cost should be unchanged
        assert!((cost.base_cost - 0.7).abs() < 1e-12);
        // current_cost should be different
        assert!(cost.current_cost < cost.base_cost);
    }

    #[test]
    fn test_recompute_saturated_pheromones() {
        // Even with pheromones >> 1.0, cost can't go below base × 0.5
        let mut cost = EdgeCost::new(4.0);
        let p = EdgePheromones {
            success: 100.0,
            traversal: 50.0,
            recency: 200.0,
        };
        recompute_edge_cost(&mut cost, &p);
        // factor = 1.0 (all clamped), discount = 0.5 → current = 4.0 × 0.5 = 2.0
        assert!((cost.current_cost - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_recompute_only_success() {
        let mut cost = EdgeCost::new(1.0);
        let p = EdgePheromones {
            success: 1.0,
            traversal: 0.0,
            recency: 0.0,
        };
        recompute_edge_cost(&mut cost, &p);
        // factor = 0.5, discount = 0.25 → current = 0.75
        assert!((cost.current_cost - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_recompute_only_recency() {
        let mut cost = EdgeCost::new(1.0);
        let p = EdgePheromones {
            success: 0.0,
            traversal: 0.0,
            recency: 1.0,
        };
        recompute_edge_cost(&mut cost, &p);
        // factor = 0.3, discount = 0.15 → current = 0.85
        assert!((cost.current_cost - 0.85).abs() < 1e-12);
    }

    #[test]
    fn test_recompute_only_traversal() {
        let mut cost = EdgeCost::new(1.0);
        let p = EdgePheromones {
            success: 0.0,
            traversal: 1.0,
            recency: 0.0,
        };
        recompute_edge_cost(&mut cost, &p);
        // factor = 0.2, discount = 0.10 → current = 0.90
        assert!((cost.current_cost - 0.90).abs() < 1e-12);
    }
}
