//! Throughput benchmarks for GraphPalace operations.
//!
//! Measures operations-per-second for insert, search, pheromone decay,
//! and export operations.

use std::time::Instant;

use gp_core::config::GraphPalaceConfig;
use gp_core::types::DrawerSource;
use gp_embeddings::engine::MockEmbeddingEngine;
use gp_palace::GraphPalace;
use gp_storage::InMemoryBackend;
use serde::{Deserialize, Serialize};

use crate::generators::generate_palace;

/// Result of a single throughput measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputResult {
    /// Name of the operation measured.
    pub operation: String,
    /// Throughput in operations per second.
    pub ops_per_sec: f64,
    /// Total number of operations performed.
    pub total_ops: usize,
    /// Total wall-clock time in milliseconds.
    pub total_time_ms: f64,
}

// ---------------------------------------------------------------------------
// Individual throughput benchmarks
// ---------------------------------------------------------------------------

/// Measure drawer insertion throughput.
///
/// Creates a palace and adds `num_drawers` drawers, measuring total time.
pub fn measure_insert_throughput(num_drawers: usize, _seed: u64) -> ThroughputResult {
    let config = GraphPalaceConfig::default();
    let storage = InMemoryBackend::new();
    let engine = Box::new(MockEmbeddingEngine::new());
    let mut palace = GraphPalace::new(config, storage, engine)
        .expect("palace creation should succeed");

    let t0 = Instant::now();

    for i in 0..num_drawers {
        let content = format!("benchmark drawer content number {i} for throughput testing");
        let wing = if i % 2 == 0 { "Wing-A" } else { "Wing-B" };
        let room = match i % 4 {
            0 => "Room-Alpha",
            1 => "Room-Beta",
            2 => "Room-Gamma",
            _ => "Room-Delta",
        };
        palace
            .add_drawer(&content, wing, room, DrawerSource::Conversation)
            .expect("insert should succeed");
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ops_per_sec = if elapsed_ms > 0.0 {
        num_drawers as f64 / (elapsed_ms / 1000.0)
    } else {
        f64::INFINITY
    };

    ThroughputResult {
        operation: "insert_drawer".into(),
        ops_per_sec,
        total_ops: num_drawers,
        total_time_ms: elapsed_ms,
    }
}

/// Measure semantic search throughput.
///
/// Populates a palace with `num_drawers` drawers, then runs `num_queries`
/// search operations, measuring total search time.
pub fn measure_search_throughput(
    num_drawers: usize,
    num_queries: usize,
    _seed: u64,
) -> ThroughputResult {
    let wings = 2;
    let rooms_per_wing = 4;
    let drawers_per_room = (num_drawers / (wings * rooms_per_wing)).max(1);

    let (mut palace, contents) = generate_palace(wings, rooms_per_wing, drawers_per_room, 0);

    // Build query strings from content.
    let queries: Vec<String> = contents
        .iter()
        .cycle()
        .take(num_queries)
        .enumerate()
        .map(|(i, (c, _, _))| {
            if i % 2 == 0 {
                c.clone()
            } else {
                c.split_whitespace().take(5).collect::<Vec<_>>().join(" ")
            }
        })
        .collect();

    let t0 = Instant::now();
    for q in &queries {
        let _ = palace.search(q, 10);
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let ops_per_sec = if elapsed_ms > 0.0 {
        num_queries as f64 / (elapsed_ms / 1000.0)
    } else {
        f64::INFINITY
    };

    ThroughputResult {
        operation: "search".into(),
        ops_per_sec,
        total_ops: num_queries,
        total_time_ms: elapsed_ms,
    }
}

/// Measure pheromone decay throughput.
///
/// Creates a palace and runs `num_cycles` decay cycles, measuring time.
pub fn measure_decay_throughput(num_drawers: usize, _seed: u64) -> ThroughputResult {
    let wings = 2;
    let rooms_per_wing = 4;
    let drawers_per_room = (num_drawers / (wings * rooms_per_wing)).max(1);

    let (mut palace, _contents) = generate_palace(wings, rooms_per_wing, drawers_per_room, 0);

    let num_cycles: usize = 50;

    let t0 = Instant::now();
    for _ in 0..num_cycles {
        palace.decay_pheromones().expect("decay should succeed");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let ops_per_sec = if elapsed_ms > 0.0 {
        num_cycles as f64 / (elapsed_ms / 1000.0)
    } else {
        f64::INFINITY
    };

    ThroughputResult {
        operation: "pheromone_decay_cycle".into(),
        ops_per_sec,
        total_ops: num_cycles,
        total_time_ms: elapsed_ms,
    }
}

/// Measure export throughput.
///
/// Creates a palace and exports it `num_exports` times.
pub fn measure_export_throughput(num_drawers: usize, _seed: u64) -> ThroughputResult {
    let wings = 2;
    let rooms_per_wing = 4;
    let drawers_per_room = (num_drawers / (wings * rooms_per_wing)).max(1);

    let (palace, _contents) = generate_palace(wings, rooms_per_wing, drawers_per_room, 0);

    let num_exports: usize = 20;

    let t0 = Instant::now();
    for _ in 0..num_exports {
        let export = palace.export().expect("export should succeed");
        // Force serialisation so we measure real work.
        let _json = export.to_json().expect("to_json should succeed");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let ops_per_sec = if elapsed_ms > 0.0 {
        num_exports as f64 / (elapsed_ms / 1000.0)
    } else {
        f64::INFINITY
    };

    ThroughputResult {
        operation: "export_json".into(),
        ops_per_sec,
        total_ops: num_exports,
        total_time_ms: elapsed_ms,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_result_serialization() {
        let r = ThroughputResult {
            operation: "test".into(),
            ops_per_sec: 1000.0,
            total_ops: 100,
            total_time_ms: 100.0,
        };
        let json = serde_json::to_string(&r).unwrap();
        let deser: ThroughputResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.operation, "test");
        assert!((deser.ops_per_sec - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn insert_throughput_positive() {
        let result = measure_insert_throughput(50, 0);
        assert_eq!(result.operation, "insert_drawer");
        assert_eq!(result.total_ops, 50);
        assert!(result.ops_per_sec > 0.0);
        assert!(result.total_time_ms > 0.0);
    }

    #[test]
    fn search_throughput_positive() {
        let result = measure_search_throughput(20, 10, 0);
        assert_eq!(result.operation, "search");
        assert_eq!(result.total_ops, 10);
        assert!(result.ops_per_sec > 0.0);
        assert!(result.total_time_ms > 0.0);
    }

    #[test]
    fn decay_throughput_positive() {
        let result = measure_decay_throughput(20, 0);
        assert_eq!(result.operation, "pheromone_decay_cycle");
        assert!(result.total_ops > 0);
        assert!(result.ops_per_sec > 0.0);
    }

    #[test]
    fn export_throughput_positive() {
        let result = measure_export_throughput(20, 0);
        assert_eq!(result.operation, "export_json");
        assert!(result.total_ops > 0);
        assert!(result.ops_per_sec > 0.0);
    }

    #[test]
    fn insert_throughput_single() {
        let result = measure_insert_throughput(1, 0);
        assert_eq!(result.total_ops, 1);
        assert!(result.ops_per_sec > 0.0);
    }

    #[test]
    fn search_throughput_many_queries() {
        let result = measure_search_throughput(16, 50, 0);
        assert_eq!(result.total_ops, 50);
    }
}
