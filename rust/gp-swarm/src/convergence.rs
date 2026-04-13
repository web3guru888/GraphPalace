//! Convergence detection for swarm exploration (spec §10.3).
//!
//! Declares convergence when ≥ 2 of 3 criteria are met:
//! 1. Average growth rate falls below threshold
//! 2. Pheromone variance falls below threshold  
//! 3. Frontier size falls below threshold
//!
//! Uses a sliding window over recent cycle history.

use serde::{Deserialize, Serialize};

/// Per-cycle statistics tracked for convergence detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleStats {
    /// Cycle number.
    pub cycle: usize,
    /// Number of new nodes discovered this cycle.
    pub new_nodes: usize,
    /// Mean pheromone level across all edges.
    pub mean_pheromone: f64,
    /// Pheromone variance across all edges.
    pub pheromone_variance: f64,
    /// Number of frontier nodes (unexplored or low-exploration).
    pub frontier_size: usize,
    /// Number of agents that found useful results.
    pub productive_agents: usize,
}

/// Rolling history of cycle statistics.
#[derive(Debug, Clone, Default)]
pub struct CycleHistory {
    /// All recorded cycle stats.
    pub stats: Vec<CycleStats>,
}

impl CycleHistory {
    /// Create a new empty history.
    pub fn new() -> Self {
        Self { stats: Vec::new() }
    }

    /// Record stats for a completed cycle.
    pub fn record(&mut self, stats: CycleStats) {
        self.stats.push(stats);
    }

    /// Average growth (new nodes per cycle) over the last `window` cycles.
    pub fn avg_growth(&self, window: usize) -> f64 {
        let recent = self.recent(window);
        if recent.is_empty() {
            return f64::MAX;
        }
        let total: usize = recent.iter().map(|s| s.new_nodes).sum();
        total as f64 / recent.len() as f64
    }

    /// Average pheromone variance over the last `window` cycles.
    pub fn avg_pheromone_variance(&self, window: usize) -> f64 {
        let recent = self.recent(window);
        if recent.is_empty() {
            return f64::MAX;
        }
        let total: f64 = recent.iter().map(|s| s.pheromone_variance).sum();
        total / recent.len() as f64
    }

    /// Current frontier size (from the most recent cycle).
    pub fn current_frontier_size(&self) -> usize {
        self.stats.last().map(|s| s.frontier_size).unwrap_or(usize::MAX)
    }

    /// Get the last `window` cycle stats.
    fn recent(&self, window: usize) -> &[CycleStats] {
        let start = self.stats.len().saturating_sub(window);
        &self.stats[start..]
    }

    /// Total number of recorded cycles.
    pub fn len(&self) -> usize {
        self.stats.len()
    }

    /// Whether no cycles have been recorded.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }
}

/// 3-criteria convergence detector (spec §10.3).
///
/// Convergence is declared when at least 2 of 3 criteria are met:
/// 1. `avg_growth(window) < growth_threshold` — not much new being discovered
/// 2. `avg_pheromone_variance(window) < variance_threshold` — pheromones stabilized
/// 3. `frontier_size < frontier_threshold` — few unexplored areas left
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceDetector {
    /// Window size for rolling averages (default: 20 cycles).
    pub history_window: usize,
    /// Growth rate threshold (nodes/cycle, default: 5.0).
    pub growth_threshold: f64,
    /// Pheromone variance threshold (default: 0.05).
    pub variance_threshold: f64,
    /// Frontier size threshold (default: 10).
    pub frontier_threshold: usize,
}

impl Default for ConvergenceDetector {
    fn default() -> Self {
        Self {
            history_window: 20,
            growth_threshold: 5.0,
            variance_threshold: 0.05,
            frontier_threshold: 10,
        }
    }
}

impl ConvergenceDetector {
    /// Create from SwarmConfig values.
    pub fn from_config(
        window: usize,
        growth_threshold: f64,
        variance_threshold: f64,
        frontier_threshold: usize,
    ) -> Self {
        Self {
            history_window: window,
            growth_threshold,
            variance_threshold,
            frontier_threshold,
        }
    }

    /// Check whether the swarm has converged.
    ///
    /// Returns `true` if at least 2 of 3 criteria are met.
    pub fn is_converged(&self, history: &CycleHistory) -> bool {
        if history.is_empty() {
            return false;
        }
        let criteria = self.evaluate_criteria(history);
        criteria.iter().filter(|&&c| c).count() >= 2
    }

    /// Evaluate each convergence criterion individually.
    ///
    /// Returns `[growth_converged, variance_converged, frontier_converged]`.
    pub fn evaluate_criteria(&self, history: &CycleHistory) -> [bool; 3] {
        [
            history.avg_growth(self.history_window) < self.growth_threshold,
            history.avg_pheromone_variance(self.history_window) < self.variance_threshold,
            history.current_frontier_size() < self.frontier_threshold,
        ]
    }

    /// Number of criteria currently met (0-3).
    pub fn criteria_met_count(&self, history: &CycleHistory) -> usize {
        self.evaluate_criteria(history).iter().filter(|&&c| c).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(cycle: usize, new_nodes: usize, pheromone_variance: f64, frontier_size: usize) -> CycleStats {
        CycleStats {
            cycle,
            new_nodes,
            mean_pheromone: 0.5,
            pheromone_variance,
            frontier_size,
            productive_agents: 0,
        }
    }

    fn converged_history() -> CycleHistory {
        let mut h = CycleHistory::new();
        // Low growth, low variance, small frontier
        for i in 0..25 {
            h.record(make_stats(i, 1, 0.01, 5));
        }
        h
    }

    fn active_history() -> CycleHistory {
        let mut h = CycleHistory::new();
        // High growth, high variance, large frontier
        for i in 0..25 {
            h.record(make_stats(i, 20, 0.5, 100));
        }
        h
    }

    #[test]
    fn empty_history_not_converged() {
        let detector = ConvergenceDetector::default();
        let history = CycleHistory::new();
        assert!(!detector.is_converged(&history));
    }

    #[test]
    fn all_criteria_met_converges() {
        let detector = ConvergenceDetector::default();
        let history = converged_history();
        assert!(detector.is_converged(&history));
        assert_eq!(detector.criteria_met_count(&history), 3);
    }

    #[test]
    fn no_criteria_met_not_converged() {
        let detector = ConvergenceDetector::default();
        let history = active_history();
        assert!(!detector.is_converged(&history));
        assert_eq!(detector.criteria_met_count(&history), 0);
    }

    #[test]
    fn two_of_three_converges() {
        let detector = ConvergenceDetector::default();
        let mut h = CycleHistory::new();
        // Low growth + low variance, but large frontier
        for i in 0..25 {
            h.record(make_stats(i, 2, 0.02, 100));
        }
        assert!(detector.is_converged(&h));
        assert_eq!(detector.criteria_met_count(&h), 2);
    }

    #[test]
    fn one_of_three_not_converged() {
        let detector = ConvergenceDetector::default();
        let mut h = CycleHistory::new();
        // Only low growth, others high
        for i in 0..25 {
            h.record(make_stats(i, 1, 0.5, 100));
        }
        assert!(!detector.is_converged(&h));
        assert_eq!(detector.criteria_met_count(&h), 1);
    }

    #[test]
    fn growth_threshold_boundary() {
        let detector = ConvergenceDetector::default(); // threshold = 5.0
        let mut h = CycleHistory::new();
        // avg = 5 nodes/cycle → NOT less than 5.0 → criterion NOT met
        for i in 0..20 {
            h.record(make_stats(i, 5, 0.01, 3));
        }
        let criteria = detector.evaluate_criteria(&h);
        assert!(!criteria[0], "avg=5 should NOT be < threshold=5.0");
        assert!(criteria[1], "variance converged");
        assert!(criteria[2], "frontier converged");
    }

    #[test]
    fn growth_just_below_threshold() {
        let detector = ConvergenceDetector::default();
        let mut h = CycleHistory::new();
        for i in 0..20 {
            h.record(make_stats(i, 4, 0.5, 100));
        }
        let criteria = detector.evaluate_criteria(&h);
        assert!(criteria[0], "avg=4 should be < threshold=5.0");
    }

    #[test]
    fn window_only_considers_recent() {
        let detector = ConvergenceDetector::from_config(5, 5.0, 0.05, 10);
        let mut h = CycleHistory::new();
        // 20 active cycles
        for i in 0..20 {
            h.record(make_stats(i, 20, 0.5, 100));
        }
        // Then 5 converged cycles
        for i in 20..25 {
            h.record(make_stats(i, 1, 0.01, 3));
        }
        // With window=5, should look at last 5 → converged
        assert!(detector.is_converged(&h));
    }

    #[test]
    fn avg_growth_calculation() {
        let mut h = CycleHistory::new();
        h.record(make_stats(0, 10, 0.0, 0));
        h.record(make_stats(1, 20, 0.0, 0));
        h.record(make_stats(2, 30, 0.0, 0));
        // avg of last 2: (20+30)/2 = 25
        assert!((h.avg_growth(2) - 25.0).abs() < 1e-10);
        // avg of all: (10+20+30)/3 = 20
        assert!((h.avg_growth(10) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn avg_pheromone_variance_calculation() {
        let mut h = CycleHistory::new();
        h.record(make_stats(0, 0, 0.1, 0));
        h.record(make_stats(1, 0, 0.3, 0));
        // avg of both: (0.1+0.3)/2 = 0.2
        assert!((h.avg_pheromone_variance(10) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn frontier_size_from_latest() {
        let mut h = CycleHistory::new();
        h.record(make_stats(0, 0, 0.0, 100));
        h.record(make_stats(1, 0, 0.0, 50));
        h.record(make_stats(2, 0, 0.0, 7));
        assert_eq!(h.current_frontier_size(), 7);
    }

    #[test]
    fn frontier_empty_history_returns_max() {
        let h = CycleHistory::new();
        assert_eq!(h.current_frontier_size(), usize::MAX);
    }

    #[test]
    fn custom_config() {
        let detector = ConvergenceDetector::from_config(10, 2.0, 0.1, 5);
        assert_eq!(detector.history_window, 10);
        assert!((detector.growth_threshold - 2.0).abs() < f64::EPSILON);
        assert!((detector.variance_threshold - 0.1).abs() < f64::EPSILON);
        assert_eq!(detector.frontier_threshold, 5);
    }

    #[test]
    fn history_len_and_is_empty() {
        let mut h = CycleHistory::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        h.record(make_stats(0, 0, 0.0, 0));
        assert!(!h.is_empty());
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn evaluate_criteria_returns_three_bools() {
        let detector = ConvergenceDetector::default();
        let h = converged_history();
        let criteria = detector.evaluate_criteria(&h);
        assert_eq!(criteria.len(), 3);
        // All should be true for converged history
        assert!(criteria[0] && criteria[1] && criteria[2]);
    }
}
