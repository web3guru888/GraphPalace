//! TF-IDF recall benchmark runner for GraphPalace.
//!
//! Compares recall metrics between MockEmbeddingEngine and TfIdfEmbeddingEngine
//! to show the benefit of real semantic embeddings.

use gp_bench::recall::{RecallBenchmark, TfIdfRecallBenchmark};

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║   GraphPalace TF-IDF vs Mock Recall Comparison          ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!();

    let scales = [
        (16, 10, "tiny (16 drawers)"),
        (100, 30, "small (100 drawers)"),
        (500, 50, "medium (500 drawers)"),
    ];

    println!("## Mock Engine Recall\n");
    println!("| Scale | Drawers | Queries | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |");
    println!("|-------|---------|---------|----------|----------|-----------|-----------|-----|");

    for (drawers, queries, label) in &scales {
        eprint!("  Mock {label}...");
        let mut bench = RecallBenchmark::new(*drawers, *queries, 42);
        let r = bench.run();
        eprintln!(" R@10={:.1}%", r.recall_at_10 * 100.0);
        println!(
            "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            label, r.num_drawers, r.num_queries,
            r.recall_at_1, r.recall_at_5, r.recall_at_10, r.recall_at_20, r.mrr,
        );
    }
    println!();

    println!("## TF-IDF Engine Recall\n");
    println!("| Scale | Drawers | Queries | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |");
    println!("|-------|---------|---------|----------|----------|-----------|-----------|-----|");

    for (drawers, queries, label) in &scales {
        eprint!("  TF-IDF {label}...");
        let mut bench = TfIdfRecallBenchmark::new(*drawers, *queries, 42);
        let r = bench.run();
        eprintln!(" R@10={:.1}%", r.recall_at_10 * 100.0);
        println!(
            "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            label, r.num_drawers, r.num_queries,
            r.recall_at_1, r.recall_at_5, r.recall_at_10, r.recall_at_20, r.mrr,
        );
    }
    println!();

    eprintln!("\n✅ TF-IDF benchmark comparison complete.");
}
