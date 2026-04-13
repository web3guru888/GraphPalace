//! Active Inference agent and Expected Free Energy (spec §6.1–6.2).
//!
//! Agents maintain beliefs about graph nodes and select actions by
//! minimising Expected Free Energy — a composite objective that balances
//! epistemic value (uncertainty reduction), pragmatic value (goal
//! alignment), and edge-quality signals (pheromone exploitation).

use std::collections::HashMap;

use gp_core::types::{Embedding, GraphNode};
use gp_embeddings::cosine_similarity;

use crate::beliefs::BeliefState;

/// An Active Inference agent that navigates the memory palace graph.
///
/// The agent maintains per-node beliefs and selects traversal actions
/// by evaluating Expected Free Energy across candidate neighbours.
#[derive(Debug, Clone)]
pub struct ActiveInferenceAgent {
    /// Unique agent identifier.
    pub id: String,
    /// Human-readable agent name.
    pub name: String,
    /// Per-node Gaussian beliefs (keyed by node id).
    pub beliefs: HashMap<String, BeliefState>,
    /// Goal embedding: the direction in semantic space the agent is drawn to.
    pub goal_embedding: Embedding,
    /// Softmax temperature for action selection (higher = more random).
    pub temperature: f64,
}

impl ActiveInferenceAgent {
    /// Create a new agent with the given goal embedding and temperature.
    pub fn new(id: String, name: String, goal_embedding: Embedding, temperature: f64) -> Self {
        Self {
            id,
            name,
            beliefs: HashMap::new(),
            goal_embedding,
            temperature,
        }
    }

    /// Get the current belief for a node, or the default prior if unseen.
    pub fn get_belief(&self, node_id: &str) -> BeliefState {
        self.beliefs
            .get(node_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Update belief for a node after observing a value with given precision.
    pub fn observe(&mut self, node_id: &str, observation: f64, precision: f64) {
        let belief = self
            .beliefs
            .entry(node_id.to_string())
            .or_default();
        belief.update(observation, precision);
    }
}

/// Compute Expected Free Energy for visiting a node.
///
/// EFE = −(epistemic + pragmatic + edge_quality)
///
/// - **Epistemic value** (1/precision): high uncertainty → more negative EFE (preferred).
/// - **Pragmatic value** (cosine similarity to goal): alignment → more negative EFE.
/// - **Edge quality**: exploitation pheromone (positive) − exploration pheromone (discouragement).
///
/// Lower (more negative) EFE means a better action.
pub fn expected_free_energy(node: &GraphNode, agent: &ActiveInferenceAgent) -> f64 {
    let belief = agent.get_belief(&node.id);

    // Epistemic: uncertainty about this node (high = explore it)
    let epistemic = 1.0 / belief.precision;

    // Pragmatic: how aligned is this node with the agent's goal
    let pragmatic =
        cosine_similarity(&node.embedding, &agent.goal_embedding).max(0.0) as f64;

    // Edge quality: exploitation signal minus exploration penalty
    let edge_quality =
        0.5 * node.pheromones.exploitation - 0.3 * node.pheromones.exploration;

    -(epistemic + pragmatic + edge_quality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::types::{NodePheromones, zero_embedding};

    fn make_node(id: &str, embedding: Embedding, pheromones: NodePheromones) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            label: id.to_string(),
            embedding,
            pheromones,
            degree: 0,
        }
    }

    fn make_agent(goal_embedding: Embedding, temperature: f64) -> ActiveInferenceAgent {
        ActiveInferenceAgent::new(
            "agent-1".into(),
            "Test Agent".into(),
            goal_embedding,
            temperature,
        )
    }

    #[test]
    fn efe_high_uncertainty_is_more_negative() {
        // Two nodes, same embedding and pheromones, but agent has
        // different belief precisions for them.
        let emb = zero_embedding();
        let node_a = make_node("a", emb, NodePheromones::default());
        let node_b = make_node("b", emb, NodePheromones::default());

        let mut agent = make_agent(zero_embedding(), 1.0);
        // a is well-known (high precision), b is uncertain (default low precision)
        agent.beliefs.insert(
            "a".into(),
            BeliefState::new(20.0, 10.0), // high precision
        );
        // b uses default (precision = 0.1) → high epistemic value

        let efe_a = expected_free_energy(&node_a, &agent);
        let efe_b = expected_free_energy(&node_b, &agent);

        // b should have lower (more negative) EFE because it's more uncertain
        assert!(
            efe_b < efe_a,
            "uncertain node should have lower EFE: efe_b={efe_b}, efe_a={efe_a}"
        );
    }

    #[test]
    fn efe_close_to_goal_is_more_negative() {
        // Node aligned with goal vs orthogonal node.
        let mut goal = zero_embedding();
        goal[0] = 1.0;

        let mut aligned_emb = zero_embedding();
        aligned_emb[0] = 1.0; // cosine = 1.0

        let orthogonal_emb = zero_embedding(); // cosine = 0.0

        let node_aligned = make_node("aligned", aligned_emb, NodePheromones::default());
        let node_ortho = make_node("ortho", orthogonal_emb, NodePheromones::default());

        let agent = make_agent(goal, 1.0);

        let efe_aligned = expected_free_energy(&node_aligned, &agent);
        let efe_ortho = expected_free_energy(&node_ortho, &agent);

        // Aligned should have lower (more negative) EFE due to pragmatic value
        assert!(
            efe_aligned < efe_ortho,
            "goal-aligned node should have lower EFE: aligned={efe_aligned}, ortho={efe_ortho}"
        );
    }

    #[test]
    fn efe_exploitation_pheromone_lowers_efe() {
        let emb = zero_embedding();
        let node_plain = make_node("plain", emb, NodePheromones::default());
        let node_exploit = make_node(
            "exploit",
            emb,
            NodePheromones {
                exploitation: 2.0,
                exploration: 0.0,
            },
        );

        let agent = make_agent(zero_embedding(), 1.0);

        let efe_plain = expected_free_energy(&node_plain, &agent);
        let efe_exploit = expected_free_energy(&node_exploit, &agent);

        // Exploitation pheromone should lower EFE
        assert!(
            efe_exploit < efe_plain,
            "exploitation should lower EFE: exploit={efe_exploit}, plain={efe_plain}"
        );
    }

    #[test]
    fn efe_exploration_pheromone_raises_efe() {
        let emb = zero_embedding();
        let node_plain = make_node("plain", emb, NodePheromones::default());
        let node_explored = make_node(
            "explored",
            emb,
            NodePheromones {
                exploitation: 0.0,
                exploration: 2.0,
            },
        );

        let agent = make_agent(zero_embedding(), 1.0);

        let efe_plain = expected_free_energy(&node_plain, &agent);
        let efe_explored = expected_free_energy(&node_explored, &agent);

        // Exploration pheromone should raise EFE (make it less preferred)
        assert!(
            efe_explored > efe_plain,
            "exploration pheromone should raise EFE: explored={efe_explored}, plain={efe_plain}"
        );
    }

    #[test]
    fn agent_observe_updates_beliefs() {
        let mut agent = make_agent(zero_embedding(), 1.0);
        assert!(!agent.beliefs.contains_key("node1"));

        agent.observe("node1", 50.0, 2.0);

        let belief = agent.get_belief("node1");
        // precision: 0.1 + 2.0 = 2.1
        assert!((belief.precision - 2.1).abs() < 1e-10);
        // mean: (0.1*20 + 2.0*50) / 2.1 = (2 + 100) / 2.1 ≈ 48.571
        let expected_mean = (0.1 * 20.0 + 2.0 * 50.0) / 2.1;
        assert!((belief.mean - expected_mean).abs() < 1e-10);
    }

    #[test]
    fn agent_get_belief_returns_default_for_unknown_node() {
        let agent = make_agent(zero_embedding(), 1.0);
        let belief = agent.get_belief("unknown");
        assert!((belief.mean - 20.0).abs() < f64::EPSILON);
        assert!((belief.precision - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn agent_multiple_observations_same_node() {
        let mut agent = make_agent(zero_embedding(), 1.0);
        agent.observe("n", 30.0, 1.0);
        agent.observe("n", 30.0, 1.0);
        agent.observe("n", 30.0, 1.0);

        let belief = agent.get_belief("n");
        // precision should be 0.1 + 3*1.0 = 3.1
        assert!((belief.precision - 3.1).abs() < 1e-10);
        // Mean converges toward 30
        assert!((belief.mean - 30.0).abs() < 1.0);
    }
}
