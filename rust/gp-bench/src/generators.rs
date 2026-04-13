//! Deterministic test-data generators for GraphPalace benchmarks.
//!
//! Every function in this module is seeded so that `same seed → same output`.

use gp_core::config::GraphPalaceConfig;
use gp_core::types::{DrawerSource, Embedding};
use gp_embeddings::engine::{EmbeddingEngine, MockEmbeddingEngine};
use gp_palace::GraphPalace;
use gp_storage::InMemoryBackend;

/// Number of embedding dimensions (must match `EMBEDDING_DIM`).
const DIM: usize = 384;

// ---------------------------------------------------------------------------
// Seeded embedding generation
// ---------------------------------------------------------------------------

/// Generate a deterministic pseudo-random embedding from `text` and `seed`.
///
/// Algorithm: hash the concatenation of `text` and `seed` with a simple LCG,
/// then L2-normalise so cosine similarity is just a dot product.
pub fn generate_embedding(text: &str, seed: u64) -> Embedding {
    // Combine text bytes with seed via FNV-1a to get initial state.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
    for &b in text.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }

    // Fill dimensions with an LCG seeded by h.
    let mut state = h;
    let mut emb = [0.0f32; DIM];
    for slot in emb.iter_mut() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *slot = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
    }

    // L2 normalise.
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for slot in emb.iter_mut() {
            *slot /= norm;
        }
    }
    emb
}

// ---------------------------------------------------------------------------
// Palace generation
// ---------------------------------------------------------------------------

/// Wing names drawn round-robin for deterministic naming.
const WING_NAMES: &[&str] = &[
    "Science",
    "History",
    "Technology",
    "Philosophy",
    "Mathematics",
    "Literature",
    "Economics",
    "Biology",
];

/// Room topics drawn round-robin for deterministic naming.
const ROOM_TOPICS: &[&str] = &[
    "Foundations",
    "Theory",
    "Applications",
    "Case Studies",
    "Methods",
    "Analysis",
    "Experiments",
    "Models",
    "Surveys",
    "Frontiers",
];

/// Content snippets used to build drawer content deterministically.
const CONTENT_SNIPPETS: &[&str] = &[
    "quantum entanglement enables instantaneous correlation",
    "photosynthesis converts light energy into chemical energy",
    "general relativity describes gravity as spacetime curvature",
    "the mitochondria is the powerhouse of the cell",
    "neural networks learn hierarchical representations",
    "supply and demand determine market equilibrium",
    "evolution by natural selection drives biodiversity",
    "plate tectonics shapes the continents over millions of years",
    "the Fibonacci sequence appears throughout nature",
    "machine learning models generalize from training data",
    "entropy measures the disorder in a thermodynamic system",
    "DNA encodes genetic instructions for biological development",
    "black holes warp spacetime beyond the event horizon",
    "algorithms solve computational problems efficiently",
    "symbiosis describes mutually beneficial relationships",
    "climate change is driven by greenhouse gas emissions",
    "prime numbers are fundamental to number theory",
    "the scientific method relies on hypothesis testing",
    "neurons communicate through electrochemical signals",
    "the universe is expanding at an accelerating rate",
];

/// Build a [`GraphPalace`] instance populated with the given dimensions.
///
/// Uses [`MockEmbeddingEngine`] so the output is deterministic and
/// embeddings are the same hash-based vectors used by search.
///
/// Returns `(palace, drawer_contents)` where `drawer_contents` is a list
/// of `(content, wing_name, room_name)` tuples in insertion order.
pub fn generate_palace(
    num_wings: usize,
    rooms_per_wing: usize,
    drawers_per_room: usize,
    _seed: u64,
) -> (GraphPalace, Vec<(String, String, String)>) {
    let config = GraphPalaceConfig::default();
    let storage = InMemoryBackend::new();
    let engine = Box::new(MockEmbeddingEngine::new());
    let mut palace = GraphPalace::new(config, storage, engine)
        .expect("palace creation should succeed");

    let mut contents: Vec<(String, String, String)> = Vec::new();
    let mut snippet_idx: usize = 0;

    for w in 0..num_wings {
        let wing_name = WING_NAMES[w % WING_NAMES.len()];

        for r in 0..rooms_per_wing {
            let room_name = ROOM_TOPICS[r % ROOM_TOPICS.len()];

            for d in 0..drawers_per_room {
                let base = CONTENT_SNIPPETS[snippet_idx % CONTENT_SNIPPETS.len()];
                // Make each drawer content unique by appending indices.
                let content = format!("{base} (w{w} r{r} d{d})");
                snippet_idx += 1;

                palace
                    .add_drawer(&content, wing_name, room_name, DrawerSource::Conversation)
                    .expect("add_drawer should succeed");

                contents.push((content, wing_name.to_string(), room_name.to_string()));
            }
        }
    }

    (palace, contents)
}

// ---------------------------------------------------------------------------
// Query generation
// ---------------------------------------------------------------------------

/// A benchmark query together with its expected answer.
#[derive(Debug, Clone)]
pub struct BenchmarkQuery {
    /// The search query text.
    pub query: String,
    /// Content of the drawer that should match.
    pub expected_content: String,
    /// Wing where the expected drawer lives.
    pub wing: String,
}

/// Generate benchmark queries with known-answer drawers.
///
/// The first `num_queries` drawers from `drawer_contents` are used to
/// generate queries.  Half are *exact-match* (query == content) and half
/// are *partial-match* (first 8 words of content).
pub fn generate_queries(
    drawer_contents: &[(String, String, String)],
    num_queries: usize,
) -> Vec<BenchmarkQuery> {
    let effective = num_queries.min(drawer_contents.len());
    let mut queries = Vec::with_capacity(effective);

    for (i, (content, wing, _room)) in drawer_contents.iter().take(effective).enumerate() {
        let query_text = if i % 2 == 0 {
            // Exact match — search for the full content.
            content.clone()
        } else {
            // Partial match — first several words.
            content
                .split_whitespace()
                .take(8)
                .collect::<Vec<_>>()
                .join(" ")
        };

        queries.push(BenchmarkQuery {
            query: query_text,
            expected_content: content.clone(),
            wing: wing.clone(),
        });
    }

    queries
}

/// Generate an embedding for a query using [`MockEmbeddingEngine`].
pub fn embed_query(query: &str) -> Embedding {
    let mut engine = MockEmbeddingEngine::new();
    engine.encode(query).expect("query encoding should succeed")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_embedding_deterministic() {
        let e1 = generate_embedding("hello", 42);
        let e2 = generate_embedding("hello", 42);
        assert_eq!(e1, e2, "same text + seed must produce identical embeddings");
    }

    #[test]
    fn generate_embedding_different_seeds_differ() {
        let e1 = generate_embedding("hello", 1);
        let e2 = generate_embedding("hello", 2);
        assert_ne!(e1, e2, "different seeds should produce different embeddings");
    }

    #[test]
    fn generate_embedding_different_text_differs() {
        let e1 = generate_embedding("alpha", 0);
        let e2 = generate_embedding("beta", 0);
        assert_ne!(e1, e2);
    }

    #[test]
    fn generate_embedding_is_normalised() {
        let e = generate_embedding("normalisation test", 99);
        let norm: f64 = e.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected unit norm, got {norm}"
        );
    }

    #[test]
    fn generate_palace_creates_correct_structure() {
        let (palace, contents) = generate_palace(2, 3, 4, 0);
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 2);
        assert_eq!(status.room_count, 6); // 2 wings × 3 rooms
        assert_eq!(status.drawer_count, 24); // 2 × 3 × 4
        assert_eq!(contents.len(), 24);
    }

    #[test]
    fn generate_palace_single_wing() {
        let (palace, contents) = generate_palace(1, 1, 1, 0);
        let status = palace.status().unwrap();
        assert_eq!(status.wing_count, 1);
        assert_eq!(status.room_count, 1);
        assert_eq!(status.drawer_count, 1);
        assert_eq!(contents.len(), 1);
    }

    #[test]
    fn generate_palace_contents_are_unique() {
        let (_palace, contents) = generate_palace(2, 2, 5, 0);
        let mut set = std::collections::HashSet::new();
        for (c, _, _) in &contents {
            assert!(set.insert(c.clone()), "duplicate content: {c}");
        }
    }

    #[test]
    fn generate_queries_exact_and_partial() {
        let (_palace, contents) = generate_palace(1, 1, 10, 0);
        let queries = generate_queries(&contents, 6);
        assert_eq!(queries.len(), 6);

        // Even indices are exact, odd are partial.
        assert_eq!(queries[0].query, queries[0].expected_content);
        assert_ne!(queries[1].query, queries[1].expected_content);
        // Partial should be a prefix (first 8 words).
        let words: Vec<_> = queries[1].expected_content.split_whitespace().take(8).collect();
        assert_eq!(queries[1].query, words.join(" "));
    }

    #[test]
    fn generate_queries_capped_to_available() {
        let (_palace, contents) = generate_palace(1, 1, 3, 0);
        let queries = generate_queries(&contents, 100);
        assert_eq!(queries.len(), 3, "should cap to available drawers");
    }

    #[test]
    fn embed_query_works() {
        let emb = embed_query("test embedding");
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn embed_query_deterministic() {
        let e1 = embed_query("same query");
        let e2 = embed_query("same query");
        assert_eq!(e1, e2);
    }
}
