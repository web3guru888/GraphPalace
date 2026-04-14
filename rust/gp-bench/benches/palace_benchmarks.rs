//! Criterion benchmark harness for GraphPalace.
//!
//! Run with: `cargo bench -p gp-bench`
//!
//! These are micro-benchmarks for core operations. For the full suite
//! with recall/pathfinding metrics, use `gp_bench::comparison::BenchmarkSuite`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use gp_core::config::GraphPalaceConfig;
use gp_core::types::DrawerSource;
use gp_embeddings::engine::MockEmbeddingEngine;
use gp_palace::GraphPalace;
use gp_pathfinding::SemanticAStar;
use gp_pathfinding::bench::PalaceGraph;
use gp_storage::InMemoryBackend;

/// Helper: create and populate a small palace with `n` drawers.
fn make_palace(n: usize) -> GraphPalace {
    let config = GraphPalaceConfig::default();
    let storage = InMemoryBackend::new();
    let engine = Box::new(MockEmbeddingEngine::new());
    let mut palace = GraphPalace::new(config, storage, engine).unwrap();
    for i in 0..n {
        let content = format!("benchmark drawer content item {i}");
        let wing = if i % 2 == 0 { "Wing-A" } else { "Wing-B" };
        let room = match i % 3 {
            0 => "Room-X",
            1 => "Room-Y",
            _ => "Room-Z",
        };
        palace
            .add_drawer(&content, wing, room, DrawerSource::Conversation)
            .unwrap();
    }
    palace
}

fn search_benchmark(c: &mut Criterion) {
    let mut palace = make_palace(100);
    c.bench_function("search_100_drawers", |b| {
        b.iter(|| {
            let _ = palace.search(black_box("benchmark drawer content"), 10);
        });
    });
}

fn insert_benchmark(c: &mut Criterion) {
    c.bench_function("insert_drawer", |b| {
        let config = GraphPalaceConfig::default();
        let storage = InMemoryBackend::new();
        let engine = Box::new(MockEmbeddingEngine::new());
        let mut palace = GraphPalace::new(config, storage, engine).unwrap();
        let mut idx = 0u64;
        b.iter(|| {
            let content = format!("bench insert {idx}");
            idx += 1;
            palace
                .add_drawer(black_box(&content), "Bench-Wing", "Bench-Room", DrawerSource::Conversation)
                .unwrap();
        });
    });
}

fn pathfinding_benchmark(c: &mut Criterion) {
    let graph = PalaceGraph::build(3, 4, 1, 2);
    let config = gp_core::config::AStarConfig::default();
    let weights = gp_core::config::CostWeights::default();
    let astar = SemanticAStar::new(config, weights);

    c.bench_function("astar_cross_wing_3x4", |b| {
        b.iter(|| {
            let _ = astar.find_path(black_box(&graph), "room_0_0", "room_2_0");
        });
    });
}

fn pheromone_benchmark(c: &mut Criterion) {
    let mut palace = make_palace(50);
    c.bench_function("pheromone_decay_50_drawers", |b| {
        b.iter(|| {
            palace.decay_pheromones().unwrap();
        });
    });
}

fn export_benchmark(c: &mut Criterion) {
    let palace = make_palace(100);
    c.bench_function("export_json_100_drawers", |b| {
        b.iter(|| {
            let export = palace.export().unwrap();
            let _ = black_box(export.to_json().unwrap());
        });
    });
}

criterion_group!(
    benches,
    search_benchmark,
    insert_benchmark,
    pathfinding_benchmark,
    pheromone_benchmark,
    export_benchmark,
);
criterion_main!(benches);
