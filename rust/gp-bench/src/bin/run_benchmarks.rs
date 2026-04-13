//! Full benchmark runner for GraphPalace.
//!
//! Runs recall, pathfinding, and throughput benchmarks at multiple scales
//! and outputs a comprehensive comparison report.

use gp_bench::comparison::{BenchmarkSuite, SuiteConfig};
use gp_bench::pathfinding::PathfindingBenchmark;
use gp_bench::recall::RecallBenchmark;
use gp_bench::throughput::*;

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║       GraphPalace Benchmark Suite — Full Run            ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Recall at multiple scales ──────────────────────────────────────
    eprintln!("═══ RECALL BENCHMARKS ═══");
    let recall_scales = [
        (16, 10, "tiny (16 drawers)"),
        (100, 30, "small (100 drawers)"),
        (500, 50, "medium (500 drawers)"),
        (1000, 80, "large (1,000 drawers)"),
        (5000, 100, "xlarge (5,000 drawers)"),
    ];

    println!("## Recall Benchmarks\n");
    println!("| Scale | Drawers | Queries | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |");
    println!("|-------|---------|---------|----------|----------|-----------|-----------|-----|");

    for (drawers, queries, label) in &recall_scales {
        eprint!("  Running {label}...");
        let mut bench = RecallBenchmark::new(*drawers, *queries, 42);
        let r = bench.run();
        eprintln!(" done (recall@10={:.1}%)", r.recall_at_10 * 100.0);
        println!(
            "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            label, r.num_drawers, r.num_queries,
            r.recall_at_1, r.recall_at_5, r.recall_at_10, r.recall_at_20, r.mrr,
        );
    }
    println!();

    // ── Pathfinding at multiple scales ─────────────────────────────────
    eprintln!("\n═══ PATHFINDING BENCHMARKS ═══");
    let pf_scales = [
        (2, 4, 50, "small (2 wings × 4 rooms)"),
        (4, 6, 100, "medium (4 wings × 6 rooms)"),
        (6, 8, 200, "large (6 wings × 8 rooms)"),
        (8, 10, 300, "xlarge (8 wings × 10 rooms)"),
    ];

    println!("## Pathfinding Benchmarks\n");
    println!("### General (random start/goal)\n");
    println!("| Scale | Trials | Success% | Avg µs | Avg Iters | Avg Path Len |");
    println!("|-------|--------|----------|--------|-----------|--------------|");

    for (wings, rooms, trials, label) in &pf_scales {
        eprint!("  Running {label}...");
        let r = PathfindingBenchmark::run(*wings, *rooms, *trials, 42);
        eprintln!(" done (success={:.1}%)", r.success_rate * 100.0);
        println!(
            "| {} | {} | {:.1}% | {:.1} | {:.1} | {:.1} |",
            label, r.total_paths, r.success_rate * 100.0,
            r.avg_time_us, r.avg_iterations, r.avg_path_length,
        );
    }
    println!();

    println!("### Same-Wing Pathfinding\n");
    println!("| Scale | Trials | Success% | Avg µs | Avg Iters | Avg Path Len |");
    println!("|-------|--------|----------|--------|-----------|--------------|");

    for (wings, rooms, trials, label) in &pf_scales {
        let r = PathfindingBenchmark::run_same_wing(*wings, *rooms, *trials, 42);
        println!(
            "| {} | {} | {:.1}% | {:.1} | {:.1} | {:.1} |",
            label, r.total_paths, r.success_rate * 100.0,
            r.avg_time_us, r.avg_iterations, r.avg_path_length,
        );
    }
    println!();

    println!("### Cross-Wing Pathfinding\n");
    println!("| Scale | Trials | Success% | Avg µs | Avg Iters | Avg Path Len |");
    println!("|-------|--------|----------|--------|-----------|--------------|");

    for (wings, rooms, trials, label) in &pf_scales {
        let r = PathfindingBenchmark::run_cross_wing(*wings, *rooms, *trials, 42);
        println!(
            "| {} | {} | {:.1}% | {:.1} | {:.1} | {:.1} |",
            label, r.total_paths, r.success_rate * 100.0,
            r.avg_time_us, r.avg_iterations, r.avg_path_length,
        );
    }
    println!();

    // ── Throughput ─────────────────────────────────────────────────────
    eprintln!("\n═══ THROUGHPUT BENCHMARKS ═══");
    let throughput_scales = [
        (100, "100 drawers"),
        (500, "500 drawers"),
        (1000, "1,000 drawers"),
        (5000, "5,000 drawers"),
    ];

    println!("## Throughput Benchmarks\n");
    println!("### Insert Throughput\n");
    println!("| Scale | Ops | Time (ms) | Ops/sec |");
    println!("|-------|-----|-----------|---------|");

    for (n, label) in &throughput_scales {
        eprint!("  Insert {label}...");
        let r = measure_insert_throughput(*n, 42);
        eprintln!(" {:.0} ops/sec", r.ops_per_sec);
        println!("| {} | {} | {:.2} | {:.0} |", label, r.total_ops, r.total_time_ms, r.ops_per_sec);
    }
    println!();

    println!("### Search Throughput\n");
    println!("| Scale | Queries | Time (ms) | Queries/sec |");
    println!("|-------|---------|-----------|-------------|");

    for (n, label) in &throughput_scales {
        eprint!("  Search {label}...");
        let r = measure_search_throughput(*n, 100, 42);
        eprintln!(" {:.0} qps", r.ops_per_sec);
        println!("| {} | {} | {:.2} | {:.0} |", label, r.total_ops, r.total_time_ms, r.ops_per_sec);
    }
    println!();

    println!("### Pheromone Decay Throughput\n");
    println!("| Scale | Cycles | Time (ms) | Cycles/sec |");
    println!("|-------|--------|-----------|------------|");

    for (n, label) in &throughput_scales {
        let r = measure_decay_throughput(*n, 42);
        println!("| {} | {} | {:.2} | {:.0} |", label, r.total_ops, r.total_time_ms, r.ops_per_sec);
    }
    println!();

    println!("### Export Throughput\n");
    println!("| Scale | Exports | Time (ms) | Exports/sec |");
    println!("|-------|---------|-----------|-------------|");

    for (n, label) in &throughput_scales {
        let r = measure_export_throughput(*n, 42);
        println!("| {} | {} | {:.2} | {:.0} |", label, r.total_ops, r.total_time_ms, r.ops_per_sec);
    }
    println!();

    // ── Full suite for JSON ────────────────────────────────────────────
    eprintln!("\n═══ FULL SUITE (default config) ═══");
    let config = SuiteConfig {
        num_drawers: 500,
        num_recall_queries: 50,
        pathfinding_wings: 4,
        pathfinding_rooms: 6,
        pathfinding_trials: 100,
        search_queries: 100,
        seed: 42,
    };
    let mut suite = BenchmarkSuite::new(config);
    let report = suite.run_all();
    eprintln!("  Suite complete. Recall@10={:.4}, Pathfinding success={:.1}%",
        report.recall.recall_at_10, report.pathfinding.success_rate * 100.0);

    // Print JSON report to stderr for capture
    eprintln!("\n═══ JSON REPORT ═══");
    eprintln!("{}", report.to_json());

    eprintln!("\n✅ All benchmarks complete.");
}
