//! Interest score computation for graph frontier nodes (spec §10.2).
//!
//! The interest score determines which nodes are most worth investigating
//! by combining structural properties, pheromone signals, and stochastic noise.
//!
//! ```text
//! interest(node) = structural + pheromone + noise
//!   structural = 1.0 / (1.0 + degree)
//!   pheromone  = 0.6 × (exploitation − exploration) + 0.4 × exploitation
//!   noise      = Normal(0, 0.1)
//! ```

use gp_core::types::GraphNode;
use rand::Rng;

/// Compute the interest score for a single node.
///
/// Higher scores indicate nodes that are more worth investigating:
/// - Low-degree nodes (structural novelty)
/// - High exploitation + low exploration pheromones (valuable but not exhausted)
/// - Random noise for diversity
///
/// Uses the provided RNG for deterministic testing.
pub fn compute_interest_score<R: Rng>(node: &GraphNode, rng: &mut R) -> f64 {
    let structural = compute_structural_score(node.degree);
    let pheromone = compute_pheromone_score(
        node.pheromones.exploitation,
        node.pheromones.exploration,
    );
    let noise = sample_noise(rng);
    structural + pheromone + noise
}

/// Compute interest scores for a batch of nodes.
pub fn compute_interest_scores<R: Rng>(nodes: &[GraphNode], rng: &mut R) -> Vec<(String, f64)> {
    nodes
        .iter()
        .map(|node| {
            let score = compute_interest_score(node, rng);
            (node.id.clone(), score)
        })
        .collect()
}

/// Rank nodes by interest score (highest first).
pub fn rank_by_interest<R: Rng>(nodes: &[GraphNode], rng: &mut R) -> Vec<(String, f64)> {
    let mut scores = compute_interest_scores(nodes, rng);
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

/// Structural component: 1.0 / (1.0 + degree).
///
/// Low-degree nodes score higher (they're novel, less connected).
#[inline]
pub fn compute_structural_score(degree: usize) -> f64 {
    1.0 / (1.0 + degree as f64)
}

/// Pheromone component: 0.6 × (exploitation − exploration) + 0.4 × exploitation.
///
/// Prefers nodes with high exploitation (valuable) and low exploration (not exhausted).
#[inline]
pub fn compute_pheromone_score(exploitation: f64, exploration: f64) -> f64 {
    0.6 * (exploitation - exploration) + 0.4 * exploitation
}

/// Sample noise from Normal(0, 0.1) using Box-Muller transform.
#[inline]
fn sample_noise<R: Rng>(rng: &mut R) -> f64 {
    // Simple Box-Muller transform for normal distribution
    // Use r#gen because `gen` is a reserved keyword in Rust 2024
    let u1: f64 = loop {
        let v: f64 = rng.r#gen::<f64>();
        if v > 0.0001 {
            break v;
        }
    };
    let u2: f64 = rng.r#gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    z * 0.1 // scale to stddev 0.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::types::{NodePheromones, GraphNode};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_node(id: &str, degree: usize, exploitation: f64, exploration: f64) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            label: id.to_string(),
            embedding: [0.0f32; 384],
            pheromones: NodePheromones { exploitation, exploration },
            degree,
        }
    }

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn structural_score_zero_degree() {
        assert!((compute_structural_score(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn structural_score_high_degree() {
        let score = compute_structural_score(99);
        assert!((score - 0.01).abs() < 1e-10);
    }

    #[test]
    fn structural_score_monotonically_decreasing() {
        for d in 0..100 {
            assert!(compute_structural_score(d) >= compute_structural_score(d + 1));
        }
    }

    #[test]
    fn pheromone_score_zero_both() {
        assert_eq!(compute_pheromone_score(0.0, 0.0), 0.0);
    }

    #[test]
    fn pheromone_score_high_exploitation() {
        let score = compute_pheromone_score(1.0, 0.0);
        // 0.6 × (1.0 - 0.0) + 0.4 × 1.0 = 0.6 + 0.4 = 1.0
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pheromone_score_high_exploration_reduces() {
        let score = compute_pheromone_score(0.5, 1.0);
        // 0.6 × (0.5 - 1.0) + 0.4 × 0.5 = 0.6 × (-0.5) + 0.2 = -0.3 + 0.2 = -0.1
        assert!((score - (-0.1)).abs() < 1e-10);
    }

    #[test]
    fn pheromone_score_balanced() {
        let score = compute_pheromone_score(0.5, 0.5);
        // 0.6 × 0.0 + 0.4 × 0.5 = 0.2
        assert!((score - 0.2).abs() < 1e-10);
    }

    #[test]
    fn interest_score_deterministic_with_seed() {
        let node = make_node("n1", 5, 0.5, 0.1);
        let mut rng1 = seeded_rng();
        let mut rng2 = seeded_rng();
        let s1 = compute_interest_score(&node, &mut rng1);
        let s2 = compute_interest_score(&node, &mut rng2);
        assert!((s1 - s2).abs() < 1e-15, "Same seed should give same result");
    }

    #[test]
    fn interest_score_contains_all_components() {
        let node = make_node("n1", 0, 0.0, 0.0);
        let mut rng = seeded_rng();
        let score = compute_interest_score(&node, &mut rng);
        // structural = 1.0, pheromone = 0.0, noise ≈ small
        // Score should be near 1.0
        assert!((score - 1.0).abs() < 0.5, "Score should be near 1.0, got {score}");
    }

    #[test]
    fn high_exploitation_increases_interest() {
        let node_low = make_node("low", 5, 0.0, 0.0);
        let node_high = make_node("high", 5, 2.0, 0.0);
        let mut rng1 = seeded_rng();
        let mut rng2 = seeded_rng();
        let s_low = compute_interest_score(&node_low, &mut rng1);
        let s_high = compute_interest_score(&node_high, &mut rng2);
        assert!(s_high > s_low, "High exploitation should increase interest");
    }

    #[test]
    fn high_exploration_decreases_interest() {
        let node_fresh = make_node("fresh", 5, 0.5, 0.0);
        let node_explored = make_node("explored", 5, 0.5, 2.0);
        let mut rng1 = seeded_rng();
        let mut rng2 = seeded_rng();
        let s_fresh = compute_interest_score(&node_fresh, &mut rng1);
        let s_explored = compute_interest_score(&node_explored, &mut rng2);
        assert!(s_fresh > s_explored, "High exploration should decrease interest");
    }

    #[test]
    fn batch_scores_correct_count() {
        let nodes = vec![
            make_node("a", 0, 0.0, 0.0),
            make_node("b", 1, 0.5, 0.1),
            make_node("c", 2, 0.3, 0.7),
        ];
        let mut rng = seeded_rng();
        let scores = compute_interest_scores(&nodes, &mut rng);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn rank_by_interest_sorted_descending() {
        let nodes = vec![
            make_node("low", 10, 0.0, 1.0),    // low interest
            make_node("high", 0, 2.0, 0.0),     // high interest
            make_node("mid", 5, 0.5, 0.5),      // mid interest
        ];
        let mut rng = seeded_rng();
        let ranked = rank_by_interest(&nodes, &mut rng);
        assert_eq!(ranked.len(), 3);
        // High should be first
        assert_eq!(ranked[0].0, "high");
        // Low should be last
        assert_eq!(ranked[2].0, "low");
    }

    #[test]
    fn empty_batch_returns_empty() {
        let mut rng = seeded_rng();
        let scores = compute_interest_scores(&[], &mut rng);
        assert!(scores.is_empty());
    }

    #[test]
    fn noise_is_bounded() {
        // Sample 1000 noise values and check they're reasonable
        let mut rng = seeded_rng();
        for _ in 0..1000 {
            let n = sample_noise(&mut rng);
            assert!(n.abs() < 1.0, "Noise should rarely exceed ±1.0: got {n}");
        }
    }
}
