//! A* pathfinding benchmarks for GraphPalace.
//!
//! Measures success rate, average latency, iteration count, and path length
//! over many random node pairs in palace graphs of configurable size.

use std::time::Instant;

use gp_core::config::{AStarConfig, CostWeights};
use gp_pathfinding::astar::SemanticAStar;
use gp_pathfinding::bench::PalaceGraph;
use serde::{Deserialize, Serialize};

/// Aggregated pathfinding results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathfindingResult {
    /// Fraction of trials that found a path.
    pub success_rate: f64,
    /// Mean time per pathfinding call (microseconds).
    pub avg_time_us: f64,
    /// Mean A* iterations per trial.
    pub avg_iterations: f64,
    /// Mean path length (nodes) for successful trials.
    pub avg_path_length: f64,
    /// Total trials executed.
    pub total_paths: usize,
    /// Number of trials that found a path.
    pub successful_paths: usize,
}

/// Pathfinding benchmark runner.
///
/// Uses [`PalaceGraph`] from `gp-pathfinding::bench` which implements
/// [`GraphAccess`] and generates a complete palace hierarchy with tunnels
/// and halls.
pub struct PathfindingBenchmark;

impl PathfindingBenchmark {
    /// Run pathfinding trials on a generated palace graph.
    ///
    /// * `num_wings` — number of wings in the palace.
    /// * `rooms_per_wing` — rooms per wing.
    /// * `num_trials` — number of random (start, goal) pairs to test.
    /// * `seed` — seed for deterministic pair selection.
    pub fn run(
        num_wings: usize,
        rooms_per_wing: usize,
        num_trials: usize,
        seed: u64,
    ) -> PathfindingResult {
        let graph = PalaceGraph::build(num_wings, rooms_per_wing, 1, 2);

        // Collect all node IDs so we can index into them.
        let node_ids: Vec<String> = Self::collect_node_ids(&graph, num_wings, rooms_per_wing);
        if node_ids.is_empty() {
            return PathfindingResult {
                success_rate: 0.0,
                avg_time_us: 0.0,
                avg_iterations: 0.0,
                avg_path_length: 0.0,
                total_paths: 0,
                successful_paths: 0,
            };
        }

        let config = AStarConfig::default();
        let weights = CostWeights::default();
        let astar = SemanticAStar::new(config, weights);

        let mut state = seed;
        let mut total_time_us = 0.0f64;
        let mut total_iterations = 0usize;
        let mut total_path_length = 0usize;
        let mut successes = 0usize;

        for _ in 0..num_trials {
            // Deterministic pair selection via LCG.
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let start_idx = (state >> 33) as usize % node_ids.len();
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let goal_idx = (state >> 33) as usize % node_ids.len();

            let start_id = &node_ids[start_idx];
            let goal_id = &node_ids[goal_idx];

            let t0 = Instant::now();
            let result = astar.find_path(&graph, start_id, goal_id);
            let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

            total_time_us += elapsed_us;
            if let Some(path) = result {
                successes += 1;
                total_iterations += path.iterations;
                total_path_length += path.path.len();
            }
        }

        let n = num_trials.max(1) as f64;
        let s = successes.max(1) as f64;
        PathfindingResult {
            success_rate: successes as f64 / n,
            avg_time_us: total_time_us / n,
            avg_iterations: total_iterations as f64 / s,
            avg_path_length: total_path_length as f64 / s,
            total_paths: num_trials,
            successful_paths: successes,
        }
    }

    /// Run trials restricted to same-wing paths.
    pub fn run_same_wing(
        num_wings: usize,
        rooms_per_wing: usize,
        num_trials: usize,
        seed: u64,
    ) -> PathfindingResult {
        let graph = PalaceGraph::build(num_wings, rooms_per_wing, 1, 2);
        let config = AStarConfig::default();
        let weights = CostWeights::default();
        let astar = SemanticAStar::new(config, weights);

        let mut state = seed;
        let mut total_time_us = 0.0f64;
        let mut total_iterations = 0usize;
        let mut total_path_length = 0usize;
        let mut successes = 0usize;

        for trial in 0..num_trials {
            // Pick a wing, then two rooms within it.
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let w = (state >> 33) as usize % num_wings;
            let r1 = trial % rooms_per_wing;
            let r2 = (trial + 1) % rooms_per_wing;
            let start_id = format!("room_{w}_{r1}");
            let goal_id = format!("room_{w}_{r2}");

            let t0 = Instant::now();
            let result = astar.find_path(&graph, &start_id, &goal_id);
            let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

            total_time_us += elapsed_us;
            if let Some(path) = result {
                successes += 1;
                total_iterations += path.iterations;
                total_path_length += path.path.len();
            }
        }

        let n = num_trials.max(1) as f64;
        let s = successes.max(1) as f64;
        PathfindingResult {
            success_rate: successes as f64 / n,
            avg_time_us: total_time_us / n,
            avg_iterations: total_iterations as f64 / s,
            avg_path_length: total_path_length as f64 / s,
            total_paths: num_trials,
            successful_paths: successes,
        }
    }

    /// Run trials restricted to cross-wing paths via tunnels.
    pub fn run_cross_wing(
        num_wings: usize,
        rooms_per_wing: usize,
        num_trials: usize,
        seed: u64,
    ) -> PathfindingResult {
        if num_wings < 2 {
            return PathfindingResult {
                success_rate: 0.0,
                avg_time_us: 0.0,
                avg_iterations: 0.0,
                avg_path_length: 0.0,
                total_paths: 0,
                successful_paths: 0,
            };
        }
        let graph = PalaceGraph::build(num_wings, rooms_per_wing, 1, 2);
        let config = AStarConfig::default();
        let weights = CostWeights::default();
        let astar = SemanticAStar::new(config, weights);

        let mut state = seed;
        let mut total_time_us = 0.0f64;
        let mut total_iterations = 0usize;
        let mut total_path_length = 0usize;
        let mut successes = 0usize;

        for _ in 0..num_trials {
            // Pick two different wings.
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let w1 = (state >> 33) as usize % num_wings;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mut w2 = (state >> 33) as usize % num_wings;
            if w2 == w1 {
                w2 = (w1 + 1) % num_wings;
            }
            let start_id = format!("room_{w1}_0");
            let goal_id = format!("room_{w2}_0");

            let t0 = Instant::now();
            let result = astar.find_path(&graph, &start_id, &goal_id);
            let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

            total_time_us += elapsed_us;
            if let Some(path) = result {
                successes += 1;
                total_iterations += path.iterations;
                total_path_length += path.path.len();
            }
        }

        let n = num_trials.max(1) as f64;
        let s = successes.max(1) as f64;
        PathfindingResult {
            success_rate: successes as f64 / n,
            avg_time_us: total_time_us / n,
            avg_iterations: total_iterations as f64 / s,
            avg_path_length: total_path_length as f64 / s,
            total_paths: num_trials,
            successful_paths: successes,
        }
    }

    /// Collect all deterministic node IDs that PalaceGraph generates.
    fn collect_node_ids(
        _graph: &PalaceGraph,
        num_wings: usize,
        rooms_per_wing: usize,
    ) -> Vec<String> {
        let mut ids = vec!["palace".to_string()];
        for w in 0..num_wings {
            ids.push(format!("wing_{w}"));
            for r in 0..rooms_per_wing {
                ids.push(format!("room_{w}_{r}"));
                // 1 closet per room, 2 drawers per closet (as built above)
                ids.push(format!("closet_{w}_{r}_0"));
                ids.push(format!("drawer_{w}_{r}_0_0"));
                ids.push(format!("drawer_{w}_{r}_0_1"));
            }
        }
        ids
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pathfinding_result_serialization() {
        let r = PathfindingResult {
            success_rate: 0.85,
            avg_time_us: 123.4,
            avg_iterations: 50.0,
            avg_path_length: 4.2,
            total_paths: 100,
            successful_paths: 85,
        };
        let json = serde_json::to_string(&r).unwrap();
        let deser: PathfindingResult = serde_json::from_str(&json).unwrap();
        assert!((deser.success_rate - 0.85).abs() < 1e-9);
        assert_eq!(deser.total_paths, 100);
    }

    #[test]
    fn run_basic_pathfinding() {
        let result = PathfindingBenchmark::run(2, 3, 10, 42);
        assert_eq!(result.total_paths, 10);
        assert!(result.success_rate >= 0.0);
        assert!(result.avg_time_us > 0.0);
    }

    #[test]
    fn same_wing_pathfinding() {
        let result = PathfindingBenchmark::run_same_wing(3, 4, 10, 42);
        assert_eq!(result.total_paths, 10);
        // Same-wing should generally succeed because rooms are connected
        // via halls within a wing.
        assert!(result.success_rate > 0.0);
    }

    #[test]
    fn cross_wing_pathfinding() {
        let result = PathfindingBenchmark::run_cross_wing(3, 3, 10, 42);
        assert_eq!(result.total_paths, 10);
        // Cross-wing paths use tunnels between room_X_0 nodes.
        assert!(result.success_rate > 0.0);
    }

    #[test]
    fn cross_wing_single_wing_returns_empty() {
        let result = PathfindingBenchmark::run_cross_wing(1, 3, 10, 42);
        assert_eq!(result.total_paths, 0);
    }

    #[test]
    fn pathfinding_deterministic() {
        let r1 = PathfindingBenchmark::run(2, 3, 5, 42);
        let r2 = PathfindingBenchmark::run(2, 3, 5, 42);
        assert_eq!(r1.successful_paths, r2.successful_paths);
        assert_eq!(r1.total_paths, r2.total_paths);
    }

    #[test]
    fn larger_palace_pathfinding() {
        let result = PathfindingBenchmark::run(4, 5, 20, 99);
        assert_eq!(result.total_paths, 20);
        // Larger palace → more nodes, should still find some paths.
        assert!(result.successful_paths > 0);
    }

    #[test]
    fn avg_path_length_positive_when_successful() {
        let result = PathfindingBenchmark::run(3, 4, 20, 7);
        if result.successful_paths > 0 {
            assert!(result.avg_path_length > 0.0);
        }
    }

    #[test]
    fn collect_node_ids_correct_count() {
        let graph = PalaceGraph::build(2, 3, 1, 2);
        let ids = PathfindingBenchmark::collect_node_ids(&graph, 2, 3);
        // 1 palace + 2 wings + 2×3 rooms + 2×3×1 closets + 2×3×1×2 drawers
        // = 1 + 2 + 6 + 6 + 12 = 27
        assert_eq!(ids.len(), 27);
    }
}
