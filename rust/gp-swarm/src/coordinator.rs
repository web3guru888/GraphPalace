//! Multi-agent swarm coordinator (spec §10.1).
//!
//! Orchestrates the sense→decide→act→update cycle for multiple Active
//! Inference agents operating on a shared graph.

use gp_core::config::PheromoneConfig;
use gp_core::types::{GraphNode, EdgePheromones, NodePheromones};
use gp_agents::active_inference::{ActiveInferenceAgent, expected_free_energy};
use gp_stigmergy::{deposit_exploration, bulk_decay_nodes, bulk_decay_edges};
use rand::Rng;

use crate::convergence::{ConvergenceDetector, CycleHistory, CycleStats};
use crate::interest::compute_interest_score;

/// Result of a single agent's action during a cycle.
#[derive(Debug, Clone)]
pub struct AgentAction {
    /// Which agent performed this action.
    pub agent_id: String,
    /// Node that was selected for investigation.
    pub target_node_id: Option<String>,
    /// Path taken (list of node IDs).
    pub path: Option<Vec<String>>,
    /// Whether the investigation was productive.
    pub productive: bool,
    /// The EFE value of the selected action.
    pub efe_value: f64,
}

/// Result of a complete swarm cycle.
#[derive(Debug, Clone)]
pub struct CycleResult {
    /// Cycle number.
    pub cycle: usize,
    /// Actions taken by each agent.
    pub actions: Vec<AgentAction>,
    /// Number of productive agents.
    pub productive_count: usize,
    /// Whether convergence was detected after this cycle.
    pub converged: bool,
    /// Whether decay was applied this cycle.
    pub decay_applied: bool,
}

/// Multi-agent swarm coordinator.
///
/// Manages the collective exploration of the memory palace by:
/// 1. **SENSE**: Compute interest scores for frontier nodes
/// 2. **DECIDE**: Each agent selects a target via EFE minimization
/// 3. **ACT**: Agents investigate their targets
/// 4. **UPDATE**: Deposit pheromones, decay, check convergence
pub struct SwarmCoordinator {
    /// Active inference agents in the swarm.
    pub agents: Vec<ActiveInferenceAgent>,
    /// How often to apply pheromone decay (in cycles).
    pub decay_interval: usize,
    /// Maximum number of cycles before stopping.
    pub max_cycles: usize,
    /// Pheromone configuration for decay operations.
    pub pheromone_config: PheromoneConfig,
    /// Convergence detector.
    pub convergence_detector: ConvergenceDetector,
    /// Rolling cycle history.
    pub history: CycleHistory,
    /// Current cycle counter.
    pub current_cycle: usize,
}

impl SwarmCoordinator {
    /// Create a new coordinator with the given agents and configuration.
    pub fn new(
        agents: Vec<ActiveInferenceAgent>,
        decay_interval: usize,
        max_cycles: usize,
        pheromone_config: PheromoneConfig,
        convergence_detector: ConvergenceDetector,
    ) -> Self {
        Self {
            agents,
            decay_interval,
            max_cycles,
            pheromone_config,
            convergence_detector,
            history: CycleHistory::new(),
            current_cycle: 0,
        }
    }

    /// Run one cycle of the swarm.
    ///
    /// The `frontier` provides candidate nodes for investigation.
    /// `node_pheromones` and `edge_pheromones` are the mutable pheromone
    /// state that gets updated during the cycle.
    ///
    /// Returns the cycle result including agent actions and convergence status.
    pub fn run_cycle<R: Rng>(
        &mut self,
        frontier: &[GraphNode],
        node_pheromones: &mut [NodePheromones],
        edge_pheromones: &mut [EdgePheromones],
        rng: &mut R,
    ) -> CycleResult {
        let cycle = self.current_cycle;

        // 1. SENSE: Compute interest scores for frontier
        let mut interest_scores: Vec<(usize, f64)> = frontier
            .iter()
            .enumerate()
            .map(|(i, node)| (i, compute_interest_score(node, rng)))
            .collect();
        interest_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. DECIDE + ACT: Each agent selects and investigates a target
        let mut actions = Vec::new();
        let mut productive_count = 0;

        for agent in &mut self.agents {
            if frontier.is_empty() {
                actions.push(AgentAction {
                    agent_id: agent.id.clone(),
                    target_node_id: None,
                    path: None,
                    productive: false,
                    efe_value: 0.0,
                });
                continue;
            }

            // Select target by minimizing EFE across top frontier nodes
            let candidates: Vec<_> = interest_scores
                .iter()
                .take(frontier.len().min(10))
                .map(|(idx, _)| &frontier[*idx])
                .collect();

            let mut best_node: Option<&GraphNode> = None;
            let mut best_efe = f64::MAX;

            for node in &candidates {
                let efe = expected_free_energy(node, agent);
                if efe < best_efe {
                    best_efe = efe;
                    best_node = Some(node);
                }
            }

            let (target_id, productive) = if let Some(node) = best_node {
                // Update agent's belief about this node
                agent.observe(&node.id, node.pheromones.exploitation, 1.0);
                (Some(node.id.clone()), best_efe < -0.5)
            } else {
                (None, false)
            };

            if productive {
                productive_count += 1;
            }

            actions.push(AgentAction {
                agent_id: agent.id.clone(),
                target_node_id: target_id,
                path: None,
                productive,
                efe_value: best_efe,
            });
        }

        // 3. UPDATE: Deposit pheromones for productive agents
        for action in &actions {
            if action.productive {
                // Deposit exploitation on node pheromones
                for np in node_pheromones.iter_mut() {
                    np.exploitation += 0.1; // Small global boost for productive cycle
                }
            }
        }

        // Deposit exploration on all investigated nodes
        for np in node_pheromones.iter_mut().take(frontier.len().min(self.agents.len())) {
            deposit_exploration(np);
        }

        // 4. DECAY: Apply decay if it's time
        let decay_applied = cycle > 0 && cycle.is_multiple_of(self.decay_interval);
        if decay_applied {
            bulk_decay_nodes(node_pheromones, &self.pheromone_config);
            bulk_decay_edges(edge_pheromones, &self.pheromone_config);
        }

        // 5. Record stats and check convergence
        let pheromone_mean: f64 = if edge_pheromones.is_empty() {
            0.0
        } else {
            edge_pheromones.iter().map(|e| e.success + e.traversal + e.recency).sum::<f64>()
                / (edge_pheromones.len() as f64 * 3.0)
        };
        let pheromone_variance = if edge_pheromones.is_empty() {
            0.0
        } else {
            let values: Vec<f64> = edge_pheromones.iter()
                .map(|e| e.success + e.traversal + e.recency)
                .collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
        };

        self.history.record(CycleStats {
            cycle,
            new_nodes: productive_count, // approximate new discoveries
            mean_pheromone: pheromone_mean,
            pheromone_variance,
            frontier_size: frontier.len(),
            productive_agents: productive_count,
        });

        let converged = self.convergence_detector.is_converged(&self.history);
        self.current_cycle += 1;

        CycleResult {
            cycle,
            actions,
            productive_count,
            converged,
            decay_applied,
        }
    }

    /// Check if the swarm should stop (converged or max cycles reached).
    pub fn should_stop(&self) -> bool {
        self.current_cycle >= self.max_cycles
            || self.convergence_detector.is_converged(&self.history)
    }

    /// Get the current cycle number.
    pub fn cycle_count(&self) -> usize {
        self.current_cycle
    }

    /// Get the number of agents in the swarm.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::types::{NodePheromones, EdgePheromones, zero_embedding};
    use gp_agents::active_inference::ActiveInferenceAgent;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_agent(id: &str) -> ActiveInferenceAgent {
        ActiveInferenceAgent::new(
            id.to_string(),
            format!("Agent {id}"),
            zero_embedding(),
            0.5,
        )
    }

    fn make_frontier(n: usize) -> Vec<GraphNode> {
        (0..n).map(|i| GraphNode {
            id: format!("node_{i}"),
            label: format!("Node {i}"),
            embedding: zero_embedding(),
            pheromones: NodePheromones { exploitation: 0.1 * i as f64, exploration: 0.0 },
            degree: i,
        }).collect()
    }

    fn make_coordinator(num_agents: usize) -> SwarmCoordinator {
        let agents: Vec<_> = (0..num_agents).map(|i| make_agent(&format!("{i}"))).collect();
        SwarmCoordinator::new(
            agents,
            10,
            100,
            PheromoneConfig::default(),
            ConvergenceDetector::default(),
        )
    }

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn coordinator_creation() {
        let coord = make_coordinator(5);
        assert_eq!(coord.agent_count(), 5);
        assert_eq!(coord.cycle_count(), 0);
        assert!(!coord.should_stop());
    }

    #[test]
    fn single_cycle_returns_actions() {
        let mut coord = make_coordinator(3);
        let frontier = make_frontier(10);
        let mut node_ph = vec![NodePheromones::default(); 10];
        let mut edge_ph = vec![EdgePheromones::default(); 5];
        let mut rng = seeded_rng();

        let result = coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        assert_eq!(result.actions.len(), 3);
        assert_eq!(result.cycle, 0);
        assert_eq!(coord.cycle_count(), 1);
    }

    #[test]
    fn empty_frontier_no_panic() {
        let mut coord = make_coordinator(2);
        let mut node_ph = vec![];
        let mut edge_ph = vec![];
        let mut rng = seeded_rng();

        let result = coord.run_cycle(&[], &mut node_ph, &mut edge_ph, &mut rng);
        assert_eq!(result.actions.len(), 2);
        // All agents should have no target
        for action in &result.actions {
            assert!(action.target_node_id.is_none());
            assert!(!action.productive);
        }
    }

    #[test]
    fn decay_applied_at_interval() {
        let mut coord = make_coordinator(1);
        coord.decay_interval = 3;
        let frontier = make_frontier(5);
        let mut node_ph = vec![NodePheromones { exploitation: 1.0, exploration: 1.0 }; 5];
        let mut edge_ph = vec![EdgePheromones { success: 1.0, traversal: 1.0, recency: 1.0 }; 3];
        let mut rng = seeded_rng();

        // Cycle 0: no decay
        let r0 = coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        assert!(!r0.decay_applied);

        // Cycle 1: no decay
        let r1 = coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        assert!(!r1.decay_applied);

        // Cycle 2: no decay
        let r2 = coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        assert!(!r2.decay_applied);

        // Cycle 3: DECAY!
        let r3 = coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        assert!(r3.decay_applied, "Cycle 3 should trigger decay");
    }

    #[test]
    fn cycle_count_increments() {
        let mut coord = make_coordinator(1);
        let frontier = make_frontier(3);
        let mut node_ph = vec![NodePheromones::default(); 3];
        let mut edge_ph = vec![];
        let mut rng = seeded_rng();

        for i in 0..5 {
            assert_eq!(coord.cycle_count(), i);
            coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        }
        assert_eq!(coord.cycle_count(), 5);
    }

    #[test]
    fn should_stop_at_max_cycles() {
        let mut coord = make_coordinator(1);
        coord.max_cycles = 3;
        let frontier = make_frontier(5);
        let mut node_ph = vec![NodePheromones::default(); 5];
        let mut edge_ph = vec![];
        let mut rng = seeded_rng();

        for _ in 0..3 {
            coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        }
        assert!(coord.should_stop());
    }

    #[test]
    fn agents_select_different_from_frontier() {
        let mut coord = make_coordinator(3);
        let frontier = make_frontier(10);
        let mut node_ph = vec![NodePheromones::default(); 10];
        let mut edge_ph = vec![];
        let mut rng = seeded_rng();

        let result = coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        // All agents should have selected a target
        for action in &result.actions {
            assert!(action.target_node_id.is_some());
        }
    }

    #[test]
    fn history_records_each_cycle() {
        let mut coord = make_coordinator(2);
        let frontier = make_frontier(5);
        let mut node_ph = vec![NodePheromones::default(); 5];
        let mut edge_ph = vec![EdgePheromones::default(); 3];
        let mut rng = seeded_rng();

        for _ in 0..10 {
            coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        }
        assert_eq!(coord.history.len(), 10);
    }

    #[test]
    fn productive_count_tracked() {
        let mut coord = make_coordinator(3);
        let frontier = make_frontier(10);
        let mut node_ph = vec![NodePheromones::default(); 10];
        let mut edge_ph = vec![];
        let mut rng = seeded_rng();

        let result = coord.run_cycle(&frontier, &mut node_ph, &mut edge_ph, &mut rng);
        // productive_count should match number of productive actions
        let count = result.actions.iter().filter(|a| a.productive).count();
        assert_eq!(result.productive_count, count);
    }
}
