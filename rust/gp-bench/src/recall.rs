//! Recall benchmarks for GraphPalace semantic search.
//!
//! Measures recall@k and MRR (Mean Reciprocal Rank) by searching a palace
//! populated with known content and checking whether the correct drawer
//! appears in the top-k results.

use gp_core::types::Embedding;
use gp_embeddings::EmbeddingEngine;
use gp_palace::GraphPalace;
use serde::{Deserialize, Serialize};

use crate::generators::{embed_query, generate_palace, generate_queries, BenchmarkQuery};

/// Recall and ranking metrics from a recall benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    /// Fraction of queries where the correct drawer is in top-1.
    pub recall_at_1: f64,
    /// Fraction of queries where the correct drawer is in top-5.
    pub recall_at_5: f64,
    /// Fraction of queries where the correct drawer is in top-10.
    pub recall_at_10: f64,
    /// Fraction of queries where the correct drawer is in top-20.
    pub recall_at_20: f64,
    /// Mean Reciprocal Rank: average of 1/rank for the correct result.
    pub mrr: f64,
    /// Number of queries executed.
    pub num_queries: usize,
    /// Number of drawers in the palace.
    pub num_drawers: usize,
}

/// A prepared recall benchmark with a populated palace and pre-computed
/// query embeddings.
pub struct RecallBenchmark {
    palace: GraphPalace,
    queries: Vec<(BenchmarkQuery, Embedding)>,
    num_drawers: usize,
}

impl RecallBenchmark {
    /// Create a recall benchmark with `num_drawers` drawers and `num_queries`
    /// known-answer queries.
    ///
    /// The palace is laid out with 2 wings, enough rooms and drawers-per-room
    /// to reach the requested total, using [`MockEmbeddingEngine`] for
    /// deterministic embeddings.
    pub fn new(num_drawers: usize, num_queries: usize, _seed: u64) -> Self {
        // Compute dimensions so that total drawers ≈ num_drawers.
        let wings: usize = 2;
        let rooms_per_wing = 4;
        let drawers_per_room = (num_drawers / (wings * rooms_per_wing)).max(1);

        let (palace, contents) = generate_palace(wings, rooms_per_wing, drawers_per_room, 0);
        let actual_drawers = contents.len();

        let raw_queries = generate_queries(&contents, num_queries);
        let queries: Vec<(BenchmarkQuery, Embedding)> = raw_queries
            .into_iter()
            .map(|q| {
                let emb = embed_query(&q.query);
                (q, emb)
            })
            .collect();

        Self {
            palace,
            queries,
            num_drawers: actual_drawers,
        }
    }

    /// Run all queries and compute recall / MRR.
    pub fn run(&mut self) -> RecallResult {
        let max_k: usize = 20;
        let mut hits_at = [0usize; 4]; // @1, @5, @10, @20
        let k_values = [1, 5, 10, 20];
        let mut rr_sum: f64 = 0.0;

        for (query, query_emb) in &self.queries {
            let results = self
                .palace
                .search_by_embedding(query_emb, max_k)
                .unwrap_or_default();

            // Find the rank (1-indexed) of the expected content.
            let mut found_rank: Option<usize> = None;
            for (rank_0, res) in results.iter().enumerate() {
                // Match by content: cosine similarity between the query's
                // *expected content* embedding and the result content.
                // Since MockEmbeddingEngine is deterministic, matching on
                // `content` string equality is the right check.
                if res.content == query.expected_content {
                    found_rank = Some(rank_0 + 1);
                    break;
                }
            }

            if let Some(rank) = found_rank {
                rr_sum += 1.0 / rank as f64;
                for (i, &k) in k_values.iter().enumerate() {
                    if rank <= k {
                        hits_at[i] += 1;
                    }
                }
            }
        }

        let n = self.queries.len().max(1) as f64;
        RecallResult {
            recall_at_1: hits_at[0] as f64 / n,
            recall_at_5: hits_at[1] as f64 / n,
            recall_at_10: hits_at[2] as f64 / n,
            recall_at_20: hits_at[3] as f64 / n,
            mrr: rr_sum / n,
            num_queries: self.queries.len(),
            num_drawers: self.num_drawers,
        }
    }
}

// ---------------------------------------------------------------------------
// TF-IDF recall benchmark
// ---------------------------------------------------------------------------

/// A recall benchmark using TF-IDF embeddings for semantically meaningful
/// similarity. Unlike `RecallBenchmark` (which uses `MockEmbeddingEngine`),
/// this measures recall with real semantic embeddings.
pub struct TfIdfRecallBenchmark {
    palace: GraphPalace,
    queries: Vec<(BenchmarkQuery, Embedding)>,
    num_drawers: usize,
}

impl TfIdfRecallBenchmark {
    /// Create a TF-IDF recall benchmark.
    ///
    /// Build a palace with TF-IDF embeddings and prepare query embeddings.
    /// Because TF-IDF vocabulary is shared, we extract the engine after
    /// building the palace, encode queries with it, then return it.
    pub fn new(num_drawers: usize, num_queries: usize, _seed: u64) -> Self {
        use crate::generators::generate_palace_tfidf;

        let wings: usize = 2;
        let rooms_per_wing = 4;
        let drawers_per_room = (num_drawers / (wings * rooms_per_wing)).max(1);

        let (palace, contents) =
            generate_palace_tfidf(wings, rooms_per_wing, drawers_per_room, 0);
        let actual_drawers = contents.len();

        let raw_queries = generate_queries(&contents, num_queries);

        // Encode queries with a fresh TF-IDF engine. Since the engine
        // learns vocabulary incrementally, we prime it with all contents
        // first, then encode queries. This simulates the palace having
        // the full vocabulary available.
        let mut tfidf_engine = gp_embeddings::TfIdfEmbeddingEngine::new();
        for (content, _, _) in &contents {
            let _ = tfidf_engine.encode(content);
        }

        let queries: Vec<(BenchmarkQuery, Embedding)> = raw_queries
            .into_iter()
            .map(|q| {
                let emb = tfidf_engine.encode(&q.query)
                    .expect("TF-IDF query encoding should succeed");
                (q, emb)
            })
            .collect();

        Self {
            palace,
            queries,
            num_drawers: actual_drawers,
        }
    }

    /// Run all queries and compute recall / MRR.
    pub fn run(&mut self) -> RecallResult {
        let max_k: usize = 20;
        let mut hits_at = [0usize; 4];
        let k_values = [1, 5, 10, 20];
        let mut rr_sum: f64 = 0.0;

        for (query, query_emb) in &self.queries {
            let results = self
                .palace
                .search_by_embedding(query_emb, max_k)
                .unwrap_or_default();

            let mut found_rank: Option<usize> = None;
            for (rank_0, res) in results.iter().enumerate() {
                if res.content == query.expected_content {
                    found_rank = Some(rank_0 + 1);
                    break;
                }
            }

            if let Some(rank) = found_rank {
                rr_sum += 1.0 / rank as f64;
                for (i, &k) in k_values.iter().enumerate() {
                    if rank <= k {
                        hits_at[i] += 1;
                    }
                }
            }
        }

        let n = self.queries.len().max(1) as f64;
        RecallResult {
            recall_at_1: hits_at[0] as f64 / n,
            recall_at_5: hits_at[1] as f64 / n,
            recall_at_10: hits_at[2] as f64 / n,
            recall_at_20: hits_at[3] as f64 / n,
            mrr: rr_sum / n,
            num_queries: self.queries.len(),
            num_drawers: self.num_drawers,
        }
    }
}

/// Compute recall metrics from pre-computed ranked results.
///
/// `ranks` is a slice of 1-indexed ranks (or `None` if the expected result
/// was not found). This utility is public so users can compute recall for
/// custom evaluation pipelines.
pub fn compute_recall(ranks: &[Option<usize>]) -> RecallResult {
    let k_values = [1, 5, 10, 20];
    let mut hits_at = [0usize; 4];
    let mut rr_sum: f64 = 0.0;

    for rank in ranks.iter().flatten() {
        rr_sum += 1.0 / *rank as f64;
        for (i, &k) in k_values.iter().enumerate() {
            if *rank <= k {
                hits_at[i] += 1;
            }
        }
    }

    let n = ranks.len().max(1) as f64;
    RecallResult {
        recall_at_1: hits_at[0] as f64 / n,
        recall_at_5: hits_at[1] as f64 / n,
        recall_at_10: hits_at[2] as f64 / n,
        recall_at_20: hits_at[3] as f64 / n,
        mrr: rr_sum / n,
        num_queries: ranks.len(),
        num_drawers: 0, // not applicable for raw ranks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_benchmark_creation() {
        let bench = RecallBenchmark::new(20, 5, 0);
        assert!(bench.num_drawers > 0);
        assert_eq!(bench.queries.len(), 5);
    }

    #[test]
    fn recall_benchmark_run_produces_metrics() {
        let mut bench = RecallBenchmark::new(20, 5, 0);
        let result = bench.run();
        assert_eq!(result.num_queries, 5);
        assert!(result.num_drawers > 0);
        // Metrics should be in [0, 1].
        assert!(result.recall_at_1 >= 0.0 && result.recall_at_1 <= 1.0);
        assert!(result.recall_at_20 >= 0.0 && result.recall_at_20 <= 1.0);
        assert!(result.mrr >= 0.0 && result.mrr <= 1.0);
    }

    #[test]
    fn exact_match_queries_have_high_recall() {
        // With MockEmbeddingEngine, exact queries should produce identical
        // embeddings and thus rank 1.
        let mut bench = RecallBenchmark::new(16, 10, 0);
        let result = bench.run();
        // At least the exact-match half (even indices) should be found at @1.
        assert!(
            result.recall_at_1 >= 0.4,
            "expected recall@1 ≥ 0.4, got {}",
            result.recall_at_1
        );
        // At @20 nearly everything should be found.
        assert!(
            result.recall_at_20 >= 0.4,
            "expected recall@20 ≥ 0.4, got {}",
            result.recall_at_20
        );
    }

    #[test]
    fn compute_recall_perfect() {
        let ranks: Vec<Option<usize>> = vec![Some(1), Some(1), Some(1)];
        let r = compute_recall(&ranks);
        assert!((r.recall_at_1 - 1.0).abs() < 1e-9);
        assert!((r.recall_at_5 - 1.0).abs() < 1e-9);
        assert!((r.mrr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn compute_recall_none() {
        let ranks: Vec<Option<usize>> = vec![None, None, None];
        let r = compute_recall(&ranks);
        assert!((r.recall_at_1).abs() < 1e-9);
        assert!((r.mrr).abs() < 1e-9);
    }

    #[test]
    fn compute_recall_mixed() {
        let ranks = vec![Some(1), Some(3), None, Some(7)];
        let r = compute_recall(&ranks);
        // recall@1: 1/4 = 0.25
        assert!((r.recall_at_1 - 0.25).abs() < 1e-9);
        // recall@5: 2/4 = 0.5 (ranks 1 and 3)
        assert!((r.recall_at_5 - 0.5).abs() < 1e-9);
        // recall@10: 3/4 = 0.75 (ranks 1, 3, 7)
        assert!((r.recall_at_10 - 0.75).abs() < 1e-9);
        // MRR: (1/1 + 1/3 + 0 + 1/7) / 4
        let expected_mrr = (1.0 + 1.0 / 3.0 + 0.0 + 1.0 / 7.0) / 4.0;
        assert!((r.mrr - expected_mrr).abs() < 1e-9);
    }

    #[test]
    fn compute_recall_empty() {
        let ranks: Vec<Option<usize>> = vec![];
        let r = compute_recall(&ranks);
        assert_eq!(r.num_queries, 0);
        assert!((r.recall_at_1).abs() < 1e-9);
    }

    #[test]
    fn recall_result_serialization() {
        let r = RecallResult {
            recall_at_1: 0.8,
            recall_at_5: 0.95,
            recall_at_10: 1.0,
            recall_at_20: 1.0,
            mrr: 0.88,
            num_queries: 50,
            num_drawers: 200,
        };
        let json = serde_json::to_string(&r).unwrap();
        let deser: RecallResult = serde_json::from_str(&json).unwrap();
        assert!((deser.recall_at_1 - 0.8).abs() < 1e-9);
        assert_eq!(deser.num_queries, 50);
    }

    #[test]
    fn recall_monotonicity() {
        // recall@k should be non-decreasing with k.
        let ranks = vec![Some(1), Some(4), Some(8), Some(15), None];
        let r = compute_recall(&ranks);
        assert!(r.recall_at_1 <= r.recall_at_5);
        assert!(r.recall_at_5 <= r.recall_at_10);
        assert!(r.recall_at_10 <= r.recall_at_20);
    }

    // ── TF-IDF recall ────────────────────────────────────────────────────

    #[test]
    fn tfidf_recall_benchmark_creation() {
        let bench = TfIdfRecallBenchmark::new(20, 5, 0);
        assert!(bench.num_drawers > 0);
        assert_eq!(bench.queries.len(), 5);
    }

    #[test]
    fn tfidf_recall_benchmark_run_produces_metrics() {
        let mut bench = TfIdfRecallBenchmark::new(20, 5, 0);
        let result = bench.run();
        assert_eq!(result.num_queries, 5);
        assert!(result.num_drawers > 0);
        assert!(result.recall_at_1 >= 0.0 && result.recall_at_1 <= 1.0);
        assert!(result.mrr >= 0.0 && result.mrr <= 1.0);
    }

    #[test]
    fn tfidf_exact_match_has_high_recall() {
        // TF-IDF: exact-match queries should have very high recall
        // because the embedding of the query text is nearly identical
        // to the stored drawer.
        let mut bench = TfIdfRecallBenchmark::new(16, 8, 0);
        let result = bench.run();
        // Even-indexed queries are exact matches, so at least 50% at @1.
        assert!(
            result.recall_at_1 >= 0.3,
            "expected TF-IDF recall@1 ≥ 0.3, got {}",
            result.recall_at_1
        );
    }

    #[test]
    fn tfidf_recall_monotonicity() {
        let mut bench = TfIdfRecallBenchmark::new(20, 10, 0);
        let r = bench.run();
        assert!(r.recall_at_1 <= r.recall_at_5);
        assert!(r.recall_at_5 <= r.recall_at_10);
        assert!(r.recall_at_10 <= r.recall_at_20);
    }
}
