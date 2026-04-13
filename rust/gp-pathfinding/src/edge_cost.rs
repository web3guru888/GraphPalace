//! Composite edge cost model from spec §5.1–5.2.
//!
//! Combines semantic distance, pheromone guidance, and structural weight
//! into a single traversal cost for each edge.

use gp_core::config::CostWeights;
use gp_core::types::{EdgePheromones, GraphEdge};

/// Compute composite edge cost: α×C_semantic + β×C_pheromone + γ×C_structural.
///
/// # Arguments
/// * `edge` – the graph edge being evaluated
/// * `target_embedding` – embedding of the node at the edge target
/// * `goal_embedding` – embedding of the overall search goal
/// * `weights` – α/β/γ cost weights (should sum to 1.0)
pub fn composite_edge_cost(
    edge: &GraphEdge,
    target_embedding: &[f32],
    goal_embedding: &[f32],
    weights: &CostWeights,
) -> f64 {
    let c_sem = semantic_cost(target_embedding, goal_embedding);
    let c_pher = pheromone_cost(&edge.pheromones);
    let c_struct = structural_cost_for(&edge.relation_type);

    weights.semantic * c_sem + weights.pheromone * c_pher + weights.structural * c_struct
}

/// Semantic cost: `1.0 - cosine_similarity(target, goal)`.
///
/// Returns 0.0 for identical embeddings, ~2.0 for opposite embeddings.
pub fn semantic_cost(target_embedding: &[f32], goal_embedding: &[f32]) -> f64 {
    let sim = gp_embeddings::cosine_similarity(target_embedding, goal_embedding) as f64;
    1.0 - sim
}

/// Pheromone cost: `1.0 - (0.5×success.min(1) + 0.3×recency.min(1) + 0.2×traversal.min(1))`.
///
/// Strong pheromones lower cost (attracting traversal); absent pheromones → cost 1.0.
pub fn pheromone_cost(pheromones: &EdgePheromones) -> f64 {
    let success = pheromones.success.min(1.0);
    let recency = pheromones.recency.min(1.0);
    let traversal = pheromones.traversal.min(1.0);
    1.0 - (0.5 * success + 0.3 * recency + 0.2 * traversal)
}

/// Structural cost for a given relation type.
///
/// Delegates to the canonical weight table in [`gp_core::types::structural_cost`].
pub fn structural_cost_for(relation_type: &str) -> f64 {
    gp_core::types::structural_cost(relation_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::types::{EdgeCost, EdgePheromones, GraphEdge};

    fn make_edge(relation: &str, pheromones: EdgePheromones) -> GraphEdge {
        GraphEdge {
            from: "a".into(),
            to: "b".into(),
            relation_type: relation.into(),
            cost: EdgeCost::new(0.5),
            pheromones,
        }
    }

    // ── semantic_cost tests ────────────────────────────────────────────

    #[test]
    fn semantic_cost_identical_embeddings() {
        let emb = [1.0f32, 2.0, 3.0];
        let cost = semantic_cost(&emb, &emb);
        assert!(cost.abs() < 1e-5, "identical embeddings should have cost ~0.0, got {cost}");
    }

    #[test]
    fn semantic_cost_opposite_embeddings() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [-1.0f32, 0.0, 0.0];
        let cost = semantic_cost(&a, &b);
        assert!(
            (cost - 2.0).abs() < 1e-5,
            "opposite embeddings should have cost ~2.0, got {cost}"
        );
    }

    #[test]
    fn semantic_cost_orthogonal_embeddings() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        let cost = semantic_cost(&a, &b);
        assert!(
            (cost - 1.0).abs() < 1e-5,
            "orthogonal embeddings should have cost ~1.0, got {cost}"
        );
    }

    #[test]
    fn semantic_cost_similar_embeddings() {
        let a = [1.0f32, 0.0];
        let b = [0.9f32, 0.1];
        let cost = semantic_cost(&a, &b);
        // These are quite similar, so cost should be small but > 0
        assert!(cost > 0.0 && cost < 0.1, "similar embeddings should have low cost, got {cost}");
    }

    // ── pheromone_cost tests ───────────────────────────────────────────

    #[test]
    fn pheromone_cost_all_zeros() {
        let p = EdgePheromones::default();
        assert!(
            (pheromone_cost(&p) - 1.0).abs() < 1e-10,
            "zero pheromones → cost 1.0"
        );
    }

    #[test]
    fn pheromone_cost_all_maxed() {
        let p = EdgePheromones {
            success: 1.0,
            traversal: 1.0,
            recency: 1.0,
        };
        let cost = pheromone_cost(&p);
        assert!(cost.abs() < 1e-10, "maxed pheromones → cost 0.0, got {cost}");
    }

    #[test]
    fn pheromone_cost_clamped_above_one() {
        let p = EdgePheromones {
            success: 5.0,
            traversal: 10.0,
            recency: 3.0,
        };
        let cost = pheromone_cost(&p);
        // After clamping all to 1.0, same as all-maxed
        assert!(cost.abs() < 1e-10, "clamped pheromones → cost 0.0, got {cost}");
    }

    #[test]
    fn pheromone_cost_partial() {
        let p = EdgePheromones {
            success: 0.5,
            traversal: 0.0,
            recency: 0.0,
        };
        // 1.0 - (0.5*0.5 + 0.3*0.0 + 0.2*0.0) = 1.0 - 0.25 = 0.75
        let cost = pheromone_cost(&p);
        assert!(
            (cost - 0.75).abs() < 1e-10,
            "partial pheromones should give 0.75, got {cost}"
        );
    }

    // ── structural_cost_for tests ──────────────────────────────────────

    #[test]
    fn structural_cost_known_types() {
        assert!((structural_cost_for("CONTAINS") - 0.2).abs() < 1e-10);
        assert!((structural_cost_for("HAS_ROOM") - 0.3).abs() < 1e-10);
        assert!((structural_cost_for("TUNNEL") - 0.7).abs() < 1e-10);
        assert!((structural_cost_for("causes") - 0.6).abs() < 1e-10);
    }

    #[test]
    fn structural_cost_unknown_defaults_to_one() {
        assert!((structural_cost_for("WORMHOLE") - 1.0).abs() < 1e-10);
    }

    // ── composite_edge_cost tests ──────────────────────────────────────

    #[test]
    fn composite_with_default_weights() {
        let weights = CostWeights::default(); // 0.4 sem, 0.3 pher, 0.3 struct
        let edge = make_edge("HALL", EdgePheromones::default());
        let target = [1.0f32, 0.0, 0.0];
        let goal = [1.0f32, 0.0, 0.0]; // identical → semantic cost 0

        let cost = composite_edge_cost(&edge, &target, &goal, &weights);
        // sem=0.0, pher=1.0 (zero pheromones), struct=0.5 (HALL)
        // 0.4*0.0 + 0.3*1.0 + 0.3*0.5 = 0.45
        assert!(
            (cost - 0.45).abs() < 1e-5,
            "expected 0.45, got {cost}"
        );
    }

    #[test]
    fn composite_with_all_components() {
        let weights = CostWeights {
            semantic: 0.5,
            pheromone: 0.3,
            structural: 0.2,
        };
        let pheromones = EdgePheromones {
            success: 1.0,
            traversal: 1.0,
            recency: 1.0,
        };
        let edge = make_edge("CONTAINS", pheromones);
        let target = [1.0f32, 0.0];
        let goal = [0.0f32, 1.0]; // orthogonal → semantic cost 1.0

        let cost = composite_edge_cost(&edge, &target, &goal, &weights);
        // sem=1.0, pher=0.0 (maxed pheromones), struct=0.2 (CONTAINS)
        // 0.5*1.0 + 0.3*0.0 + 0.2*0.2 = 0.54
        assert!(
            (cost - 0.54).abs() < 1e-5,
            "expected 0.54, got {cost}"
        );
    }
}
