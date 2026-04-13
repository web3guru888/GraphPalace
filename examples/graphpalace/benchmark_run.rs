//! Benchmark Run
//!
//! Demonstrates how to generate a test palace, run recall and pathfinding
//! benchmarks, and print structured results.
//!
//! Note: This example references GraphPalace crate APIs. It compiles
//! against the types but requires gp-bench and gp-palace crates.

use gp_bench::{
    BenchScale, PathfindingBenchmark, RecallBenchmark, TestPalaceGenerator, ThroughputBenchmark,
};

fn main() {
    println!("=== GraphPalace Benchmark Suite ===\n");

    // Generate a deterministic test palace (seed=42 for reproducibility)
    let generator = TestPalaceGenerator::new(42);
    let mut palace = generator.generate(1_000); // 1,000 drawers
    let status = palace.status().unwrap();
    println!(
        "Test palace: {} wings, {} rooms, {} drawers\n",
        status.wings, status.rooms, status.drawers
    );

    // --- Recall Benchmark ---
    // Target: ≥96.6% (MemPalace's LongMemEval score)
    println!("--- Recall Benchmark (target: ≥96.6%) ---");
    let recall_bench = RecallBenchmark::new()
        .k(10)
        .num_queries(500)
        .similarity_threshold(0.0);

    let recall = recall_bench.run(&palace).unwrap();
    println!("  Recall@10:  {:.1}%", recall.recall * 100.0);
    println!("  MRR:        {:.3}", recall.mrr);
    println!("  Avg latency: {:.1}ms", recall.avg_latency_ms);
    println!("  P99 latency: {:.1}ms", recall.p99_latency_ms);
    println!(
        "  Result: {}",
        if recall.recall >= 0.966 {
            "✓ PASS"
        } else {
            "✗ BELOW TARGET"
        }
    );

    // --- Pathfinding Benchmark ---
    // Target: ≥90.9% success (STAN_X v8)
    println!("\n--- Pathfinding Benchmark (target: ≥90.9%) ---");
    let path_bench = PathfindingBenchmark::new()
        .num_pairs(200)
        .max_iterations(10_000)
        .include_cross_wing(true);

    let pathfinding = path_bench.run(&palace).unwrap();
    println!("  Success rate: {:.1}%", pathfinding.success_rate * 100.0);
    println!("  Avg cost:     {:.3}", pathfinding.avg_cost);
    println!("  Avg latency:  {:.1}ms", pathfinding.avg_latency_ms);
    println!("  Avg expanded: {:.0} nodes", pathfinding.avg_nodes_expanded);
    println!(
        "  Result: {}",
        if pathfinding.success_rate >= 0.909 {
            "✓ PASS"
        } else {
            "✗ BELOW TARGET"
        }
    );

    // --- Throughput Benchmark ---
    println!("\n--- Throughput Benchmark ---");
    let tp_bench = ThroughputBenchmark::new()
        .warmup_iterations(50)
        .measurement_iterations(500);

    let throughput = tp_bench.run(&mut palace).unwrap();
    println!("  Insert:  {:.0} drawers/sec", throughput.insert_rate);
    println!("  Search:  {:.0} queries/sec", throughput.search_rate);
    println!("  Decay:   {:.1}ms per cycle", throughput.decay_ms);
    println!("  Export:  {:.1}ms", throughput.export_ms);
    println!("  Import:  {:.1}ms", throughput.import_ms);

    // --- Summary ---
    println!("\n=== Summary ===");
    println!(
        "Recall:      {:.1}% (target 96.6%, delta {:+.1}%)",
        recall.recall * 100.0,
        (recall.recall - 0.966) * 100.0
    );
    println!(
        "Pathfinding: {:.1}% (target 90.9%, delta {:+.1}%)",
        pathfinding.success_rate * 100.0,
        (pathfinding.success_rate - 0.909) * 100.0
    );
    println!("Throughput:  {:.0} inserts/s, {:.0} searches/s",
        throughput.insert_rate, throughput.search_rate);
}
