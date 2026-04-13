//! Comparison framework — runs every benchmark and produces a combined report.

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::pathfinding::{PathfindingBenchmark, PathfindingResult};
use crate::recall::{RecallBenchmark, RecallResult};
use crate::throughput::{
    measure_decay_throughput, measure_export_throughput, measure_insert_throughput,
    measure_search_throughput, ThroughputResult,
};

// ---------------------------------------------------------------------------
// Suite configuration
// ---------------------------------------------------------------------------

/// Configuration for a complete benchmark suite run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteConfig {
    /// Number of drawers for recall & throughput tests.
    pub num_drawers: usize,
    /// Number of recall queries.
    pub num_recall_queries: usize,
    /// Number of wings for pathfinding tests.
    pub pathfinding_wings: usize,
    /// Rooms per wing for pathfinding tests.
    pub pathfinding_rooms: usize,
    /// Number of pathfinding trials.
    pub pathfinding_trials: usize,
    /// Number of search-throughput queries.
    pub search_queries: usize,
    /// Seed for reproducibility.
    pub seed: u64,
}

impl Default for SuiteConfig {
    fn default() -> Self {
        Self {
            num_drawers: 100,
            num_recall_queries: 20,
            pathfinding_wings: 3,
            pathfinding_rooms: 4,
            pathfinding_trials: 30,
            search_queries: 50,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Complete benchmark report containing all results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Recall metrics.
    pub recall: RecallResult,
    /// General pathfinding metrics.
    pub pathfinding: PathfindingResult,
    /// Same-wing pathfinding metrics.
    pub pathfinding_same_wing: PathfindingResult,
    /// Cross-wing pathfinding metrics.
    pub pathfinding_cross_wing: PathfindingResult,
    /// Throughput measurements.
    pub throughput: Vec<ThroughputResult>,
    /// ISO-8601 timestamp of the report.
    pub timestamp: String,
    /// Suite configuration as a JSON string.
    pub config: String,
}

impl BenchmarkReport {
    /// Serialize the report to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }

    /// Render the report as a Markdown document with tables.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# GraphPalace Benchmark Report\n\n");
        md.push_str(&format!("**Generated**: {}\n\n", self.timestamp));

        // -- Recall ----------------------------------------------------------
        md.push_str("## Recall Metrics\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Recall@1 | {:.4} |\n", self.recall.recall_at_1));
        md.push_str(&format!("| Recall@5 | {:.4} |\n", self.recall.recall_at_5));
        md.push_str(&format!(
            "| Recall@10 | {:.4} |\n",
            self.recall.recall_at_10
        ));
        md.push_str(&format!(
            "| Recall@20 | {:.4} |\n",
            self.recall.recall_at_20
        ));
        md.push_str(&format!("| MRR | {:.4} |\n", self.recall.mrr));
        md.push_str(&format!("| Queries | {} |\n", self.recall.num_queries));
        md.push_str(&format!("| Drawers | {} |\n\n", self.recall.num_drawers));

        // -- Pathfinding -----------------------------------------------------
        md.push_str("## Pathfinding Metrics\n\n");
        md.push_str("| Scenario | Success% | Avg µs | Avg Iters | Avg Path Len | Trials |\n");
        md.push_str("|----------|----------|--------|-----------|--------------|--------|\n");
        for (label, pf) in [
            ("General", &self.pathfinding),
            ("Same-Wing", &self.pathfinding_same_wing),
            ("Cross-Wing", &self.pathfinding_cross_wing),
        ] {
            md.push_str(&format!(
                "| {} | {:.1}% | {:.1} | {:.1} | {:.1} | {} |\n",
                label,
                pf.success_rate * 100.0,
                pf.avg_time_us,
                pf.avg_iterations,
                pf.avg_path_length,
                pf.total_paths,
            ));
        }
        md.push('\n');

        // -- Throughput ------------------------------------------------------
        md.push_str("## Throughput\n\n");
        md.push_str("| Operation | Ops/sec | Total Ops | Time (ms) |\n");
        md.push_str("|-----------|---------|-----------|----------|\n");
        for t in &self.throughput {
            md.push_str(&format!(
                "| {} | {:.0} | {} | {:.2} |\n",
                t.operation, t.ops_per_sec, t.total_ops, t.total_time_ms,
            ));
        }
        md.push('\n');

        md
    }
}

// ---------------------------------------------------------------------------
// Suite runner
// ---------------------------------------------------------------------------

/// Runs the full benchmark suite.
pub struct BenchmarkSuite {
    config: SuiteConfig,
}

impl BenchmarkSuite {
    /// Create a new suite with the given configuration.
    pub fn new(config: SuiteConfig) -> Self {
        Self { config }
    }

    /// Create a suite with default (moderate-size) configuration.
    pub fn default_suite() -> Self {
        Self::new(SuiteConfig::default())
    }

    /// Run all benchmarks and return a combined report.
    pub fn run_all(&mut self) -> BenchmarkReport {
        let c = &self.config;

        // 1. Recall
        let mut recall_bench = RecallBenchmark::new(c.num_drawers, c.num_recall_queries, c.seed);
        let recall = recall_bench.run();

        // 2. Pathfinding (general / same-wing / cross-wing)
        let pathfinding = PathfindingBenchmark::run(
            c.pathfinding_wings,
            c.pathfinding_rooms,
            c.pathfinding_trials,
            c.seed,
        );
        let pathfinding_same_wing = PathfindingBenchmark::run_same_wing(
            c.pathfinding_wings,
            c.pathfinding_rooms,
            c.pathfinding_trials,
            c.seed,
        );
        let pathfinding_cross_wing = PathfindingBenchmark::run_cross_wing(
            c.pathfinding_wings,
            c.pathfinding_rooms,
            c.pathfinding_trials,
            c.seed,
        );

        // 3. Throughput
        let throughput = vec![
            measure_insert_throughput(c.num_drawers, c.seed),
            measure_search_throughput(c.num_drawers, c.search_queries, c.seed),
            measure_decay_throughput(c.num_drawers, c.seed),
            measure_export_throughput(c.num_drawers, c.seed),
        ];

        let config_json = serde_json::to_string(&self.config)
            .unwrap_or_else(|_| "{}".to_string());

        BenchmarkReport {
            recall,
            pathfinding,
            pathfinding_same_wing,
            pathfinding_cross_wing,
            throughput,
            timestamp: Utc::now().to_rfc3339(),
            config: config_json,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_config_default() {
        let c = SuiteConfig::default();
        assert_eq!(c.num_drawers, 100);
        assert_eq!(c.seed, 42);
    }

    #[test]
    fn suite_config_serialization() {
        let c = SuiteConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let deser: SuiteConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.num_drawers, c.num_drawers);
    }

    #[test]
    fn run_small_suite() {
        let config = SuiteConfig {
            num_drawers: 16,
            num_recall_queries: 5,
            pathfinding_wings: 2,
            pathfinding_rooms: 3,
            pathfinding_trials: 5,
            search_queries: 5,
            seed: 42,
        };
        let mut suite = BenchmarkSuite::new(config);
        let report = suite.run_all();

        // Sanity checks on the report.
        assert!(report.recall.num_queries > 0);
        assert!(report.pathfinding.total_paths > 0);
        assert_eq!(report.throughput.len(), 4);
        assert!(!report.timestamp.is_empty());
    }

    #[test]
    fn report_to_json() {
        let config = SuiteConfig {
            num_drawers: 8,
            num_recall_queries: 3,
            pathfinding_wings: 2,
            pathfinding_rooms: 2,
            pathfinding_trials: 3,
            search_queries: 3,
            seed: 1,
        };
        let mut suite = BenchmarkSuite::new(config);
        let report = suite.run_all();

        let json = report.to_json();
        assert!(json.contains("recall"));
        assert!(json.contains("pathfinding"));
        assert!(json.contains("throughput"));
    }

    #[test]
    fn report_to_markdown() {
        let config = SuiteConfig {
            num_drawers: 8,
            num_recall_queries: 3,
            pathfinding_wings: 2,
            pathfinding_rooms: 2,
            pathfinding_trials: 3,
            search_queries: 3,
            seed: 1,
        };
        let mut suite = BenchmarkSuite::new(config);
        let report = suite.run_all();

        let md = report.to_markdown();
        assert!(md.contains("# GraphPalace Benchmark Report"));
        assert!(md.contains("Recall@1"));
        assert!(md.contains("Ops/sec"));
        assert!(md.contains("Same-Wing"));
        assert!(md.contains("Cross-Wing"));
    }

    #[test]
    fn report_json_round_trip() {
        let config = SuiteConfig {
            num_drawers: 8,
            num_recall_queries: 3,
            pathfinding_wings: 2,
            pathfinding_rooms: 2,
            pathfinding_trials: 3,
            search_queries: 3,
            seed: 1,
        };
        let mut suite = BenchmarkSuite::new(config);
        let report = suite.run_all();

        let json = report.to_json();
        let deser: BenchmarkReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.recall.num_queries, report.recall.num_queries);
        assert_eq!(deser.throughput.len(), 4);
    }

    #[test]
    fn default_suite_runs() {
        // Use a small config override via the new() constructor.
        let config = SuiteConfig {
            num_drawers: 16,
            num_recall_queries: 4,
            pathfinding_wings: 2,
            pathfinding_rooms: 2,
            pathfinding_trials: 4,
            search_queries: 4,
            seed: 7,
        };
        let mut suite = BenchmarkSuite::new(config);
        let report = suite.run_all();
        assert!(report.throughput.iter().all(|t| t.ops_per_sec > 0.0));
    }
}
