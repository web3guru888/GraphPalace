//! Adaptive semantic heuristic from spec §5.3.
//!
//! The heuristic adapts its weighting based on whether the current node
//! is in the same semantic domain as the goal (high similarity) or a
//! different domain (low similarity).

/// Semantic A* heuristic that adapts between cross-domain and same-domain.
///
/// When `similarity < threshold` (cross-domain), uses a 50/50 mix of
/// semantic and graph-structural components, encouraging broader exploration.
///
/// When `similarity >= threshold` (same-domain), uses 90/10 semantic-heavy
/// weighting for more focused pursuit.
///
/// # Arguments
/// * `current_embedding` – embedding of the current node
/// * `goal_embedding` – embedding of the goal node
/// * `current_degree` – number of edges connected to the current node
/// * `cross_domain_threshold` – similarity threshold (default 0.3)
pub fn semantic_heuristic(
    current_embedding: &[f32],
    goal_embedding: &[f32],
    current_degree: usize,
    cross_domain_threshold: f64,
) -> f64 {
    let sim = gp_embeddings::cosine_similarity(current_embedding, goal_embedding) as f64;
    let h_semantic = 1.0 - sim;

    // Connectivity factor: well-connected nodes are easier to route through.
    let connectivity = (current_degree as f64 / 20.0).clamp(0.1, 1.0);

    // Graph-structural component penalises low-connectivity nodes.
    let h_graph = (h_semantic / connectivity) * 0.5;

    let similarity = 1.0 - h_semantic; // == sim, but written symmetrically

    if similarity < cross_domain_threshold {
        // Cross-domain: equal weight to exploration via graph structure.
        0.5 * h_semantic + 0.5 * h_graph
    } else {
        // Same-domain: heavy semantic focus.
        0.9 * h_semantic + 0.1 * h_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_domain_low_similarity() {
        // Orthogonal embeddings → similarity 0.0 < 0.3 → cross-domain
        let current = [1.0f32, 0.0, 0.0];
        let goal = [0.0f32, 1.0, 0.0];
        let h = semantic_heuristic(&current, &goal, 5, 0.3);

        // h_semantic = 1.0, connectivity = 5/20 = 0.25
        // h_graph = (1.0 / 0.25) * 0.5 = 2.0
        // cross-domain: 0.5*1.0 + 0.5*2.0 = 1.5
        assert!(
            (h - 1.5).abs() < 1e-5,
            "expected ~1.5 for cross-domain, got {h}"
        );
    }

    #[test]
    fn same_domain_high_similarity() {
        // Very similar embeddings → similarity ≈ 1.0 > 0.3 → same-domain
        let current = [1.0f32, 0.0, 0.0];
        let goal = [1.0f32, 0.0, 0.0];
        let h = semantic_heuristic(&current, &goal, 10, 0.3);

        // h_semantic ≈ 0.0, same-domain: 0.9*0 + 0.1*0 ≈ 0
        assert!(h.abs() < 1e-5, "identical should be ~0, got {h}");
    }

    #[test]
    fn boundary_at_threshold() {
        // Build embeddings so cosine sim ≈ 0.3 exactly
        // cos(θ) = 0.3 → a=[0.3, sqrt(1-0.09)] = [0.3, ~0.9539], b=[1,0]
        let current = [0.3f32, (1.0f32 - 0.09).sqrt()];
        let goal = [1.0f32, 0.0f32];

        let sim = gp_embeddings::cosine_similarity(&current, &goal) as f64;
        // sim ≈ 0.3

        // At threshold → same-domain branch (>= threshold)
        let h = semantic_heuristic(&current, &goal, 10, sim);
        // h_semantic = 1 - sim ≈ 0.7
        // connectivity = 10/20 = 0.5
        // h_graph = (0.7/0.5)*0.5 = 0.7
        // same-domain: 0.9*0.7 + 0.1*0.7 = 0.63 + 0.07 = 0.7
        let expected = 0.9 * (1.0 - sim) + 0.1 * ((1.0 - sim) / 0.5) * 0.5;
        assert!(
            (h - expected).abs() < 1e-5,
            "at boundary should use same-domain, expected {expected}, got {h}"
        );
    }

    #[test]
    fn low_degree_increases_heuristic() {
        // Same embeddings, different degrees
        let current = [1.0f32, 0.0, 0.0];
        let goal = [0.0f32, 0.0, 1.0];

        let h_low = semantic_heuristic(&current, &goal, 1, 0.3);
        let h_high = semantic_heuristic(&current, &goal, 20, 0.3);

        // Low degree → low connectivity → higher h_graph → higher total
        assert!(
            h_low > h_high,
            "low degree should produce higher heuristic: {h_low} vs {h_high}"
        );
    }

    #[test]
    fn degree_zero_uses_minimum_connectivity() {
        let current = [1.0f32, 0.0];
        let goal = [0.0f32, 1.0];
        let h = semantic_heuristic(&current, &goal, 0, 0.3);

        // connectivity clamped to 0.1
        // h_semantic = 1.0, h_graph = (1.0/0.1)*0.5 = 5.0
        // cross-domain: 0.5*1.0 + 0.5*5.0 = 3.0
        assert!(
            (h - 3.0).abs() < 1e-5,
            "zero degree should clamp to min connectivity, got {h}"
        );
    }

    #[test]
    fn high_degree_caps_connectivity() {
        let current = [1.0f32, 0.0];
        let goal = [0.0f32, 1.0];

        // degree 100 → connectivity capped at 1.0
        let h100 = semantic_heuristic(&current, &goal, 100, 0.3);
        let h20 = semantic_heuristic(&current, &goal, 20, 0.3);

        assert!(
            (h100 - h20).abs() < 1e-5,
            "degree beyond 20 should give same result: {h100} vs {h20}"
        );
    }
}
