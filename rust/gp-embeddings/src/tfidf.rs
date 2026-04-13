//! TF-IDF embedding engine for GraphPalace.
//!
//! Produces genuinely semantic embeddings by computing TF-IDF vectors
//! over a learned vocabulary, then projecting to 384 dimensions via
//! a seeded random projection matrix (Johnson-Lindenstrauss).
//!
//! This engine doesn't need any external model files — it builds
//! its vocabulary incrementally from inserted text. Texts that share
//! words will have high cosine similarity; texts with completely
//! different words will be nearly orthogonal.
//!
//! # How it works
//!
//! 1. **Tokenise** the input text into lowercase word tokens.
//! 2. **TF** = term frequency within the document (count / total_words).
//! 3. **IDF** = log(N / df_t) where N = total documents seen, df_t = docs
//!    containing term t.
//! 4. The TF-IDF vector lives in vocabulary-dimension space (potentially huge).
//! 5. A **random projection matrix** (384 × vocab_size), seeded for
//!    reproducibility, projects this sparse vector to 384 dims.
//! 6. The result is **L2-normalised** so cosine similarity = dot product.
//!
//! After each new document, the IDF values change, so embeddings for
//! previously seen texts would shift slightly. This is acceptable for
//! a memory palace where recent context matters most.

use std::collections::HashMap;

use gp_core::{Embedding, GraphPalaceError, Result, EMBEDDING_DIM};

use crate::engine::EmbeddingEngine;

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------

/// Simple whitespace + punctuation tokeniser. Lowercases everything,
/// strips non-alphanumeric chars except hyphens, removes tokens shorter
/// than 2 characters.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == ';' || c == ':' || c == '!' || c == '?' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == '"' || c == '\'')
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-'))
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Random projection
// ---------------------------------------------------------------------------

/// Stable hash for a term string. Used as the term identifier for random
/// projection so that the mapping is vocabulary-order-independent.
fn term_hash(term: &str) -> u64 {
    let mut h: u64 = 0x517c_c1b7_2722_0a95;
    for &b in term.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Deterministic pseudo-random f32 from two seeds using SplitMix64.
/// Returns a value in [-1, 1].
fn seeded_random(dim: u64, term: u64) -> f32 {
    // SplitMix64 — fast, well-distributed, deterministic
    let mut z = dim
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ term.wrapping_mul(0x6c62_272e_07bb_0142);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // Map full u64 range to [-1, 1]
    (z as i64 as f64 / i64::MAX as f64) as f32
}

/// Generate one element of the random projection matrix for a given
/// (output_dim, term_index) pair.
///
/// Uses a sparse random projection (Achlioptas 2003):
/// - With prob 1/3: +sqrt(3/EMBEDDING_DIM)
/// - With prob 1/3: 0
/// - With prob 1/3: -sqrt(3/EMBEDDING_DIM)
///
/// This preserves distances better than dense projection and produces
/// more discriminative embeddings for small vocabularies.
#[cfg(test)]
fn projection_value(output_dim: usize, term_idx: usize) -> f32 {
    let r = seeded_random(output_dim as u64, term_idx as u64);
    let scale = (3.0f32 / EMBEDDING_DIM as f32).sqrt();
    if r < -0.333 {
        -scale
    } else if r > 0.333 {
        scale
    } else {
        0.0
    }
}

/// Stable projection value using a term's string hash instead of its
/// vocabulary index.  This ensures the projection for a given word is
/// identical regardless of insertion order, making embeddings
/// reproducible across palace reload cycles.
fn projection_value_stable(output_dim: usize, term: &str) -> f32 {
    let r = seeded_random(output_dim as u64, term_hash(term));
    let scale = (3.0f32 / EMBEDDING_DIM as f32).sqrt();
    if r < -0.333 {
        -scale
    } else if r > 0.333 {
        scale
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// TfIdfEmbeddingEngine
// ---------------------------------------------------------------------------

/// A real semantic embedding engine based on TF-IDF + random projection.
///
/// Builds vocabulary incrementally. Texts with shared words produce
/// similar embeddings; texts with no word overlap produce near-orthogonal
/// embeddings.
///
/// # Thread safety
///
/// Not `Send`/`Sync` by default — use behind a `Mutex` if needed.
pub struct TfIdfEmbeddingEngine {
    /// term → index in the vocabulary
    vocab: HashMap<String, usize>,
    /// For each term, the number of documents containing it
    doc_freq: HashMap<String, usize>,
    /// Total unique documents seen
    total_docs: usize,
    /// Set of document hashes to avoid double-counting
    seen_docs: std::collections::HashSet<u64>,
    /// Pre-built vocabulary mode (false = learn incrementally)
    frozen: bool,
}

impl std::fmt::Debug for TfIdfEmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TfIdfEmbeddingEngine")
            .field("vocab_size", &self.vocab.len())
            .field("total_docs", &self.total_docs)
            .field("frozen", &self.frozen)
            .finish()
    }
}

impl TfIdfEmbeddingEngine {
    /// Create a new TF-IDF engine with an empty vocabulary.
    ///
    /// The vocabulary grows as new texts are encoded.
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            doc_freq: HashMap::new(),
            total_docs: 0,
            seen_docs: std::collections::HashSet::new(),
            frozen: false,
        }
    }

    /// Create a TF-IDF engine with a pre-built vocabulary from a corpus.
    ///
    /// After construction the vocabulary is frozen — new terms encountered
    /// during `encode()` are silently ignored (treated as out-of-vocabulary).
    pub fn from_corpus(texts: &[&str]) -> Self {
        let mut engine = Self::new();
        for text in texts {
            engine.learn_document(text);
        }
        engine.frozen = true;
        engine
    }

    /// Freeze the vocabulary — no new terms will be added.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Unfreeze the vocabulary — new terms will be learned.
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }

    /// Return current vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Return total documents seen.
    pub fn total_docs(&self) -> usize {
        self.total_docs
    }

    /// Compute a simple hash of text for deduplication.
    fn text_hash(text: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in text.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// Learn a document: update vocabulary and document frequencies.
    /// Deduplicates by content hash — encoding the same text twice
    /// won't change IDF values.
    fn learn_document(&mut self, text: &str) {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }

        // Deduplicate: skip if already seen
        let hash = Self::text_hash(text);
        if !self.seen_docs.insert(hash) {
            // Already learned this document — just ensure vocab exists
            if !self.frozen {
                for token in &tokens {
                    let next_idx = self.vocab.len();
                    self.vocab.entry(token.clone()).or_insert(next_idx);
                }
            }
            return;
        }

        self.total_docs += 1;

        // Collect unique terms in this document
        let mut seen_in_doc: std::collections::HashSet<&str> =
            std::collections::HashSet::new();

        for token in &tokens {
            // Add to vocabulary if not frozen
            if !self.frozen {
                let next_idx = self.vocab.len();
                self.vocab.entry(token.clone()).or_insert(next_idx);
            }

            // Track document frequency (once per document per term)
            if seen_in_doc.insert(token) {
                *self.doc_freq.entry(token.clone()).or_insert(0) += 1;
            }
        }
    }

    /// Compute the TF-IDF vector for a text, then project to 384 dims.
    fn compute_embedding(&self, text: &str) -> Embedding {
        let tokens = tokenize(text);
        let total_terms = tokens.len() as f32;
        if total_terms == 0.0 {
            return [0.0f32; EMBEDDING_DIM];
        }

        // Compute term frequencies
        let mut tf: HashMap<&str, f32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0.0) += 1.0;
        }
        // Normalize TF by document length
        for val in tf.values_mut() {
            *val /= total_terms;
        }

        // Compute TF-IDF and project to 384 dims in one pass
        // (avoid materializing the full sparse vector)
        let n = (self.total_docs.max(1)) as f32;
        let mut emb = [0.0f32; EMBEDDING_DIM];

        for (term, &term_tf) in &tf {
            // Only include terms that are in the vocabulary (OOV terms are ignored)
            if self.vocab.contains_key(*term) {
                let df = self.doc_freq.get(*term).copied().unwrap_or(1) as f32;
                let idf = (n / df).ln() + 1.0; // smooth IDF
                let tfidf = term_tf * idf;

                // Accumulate random projection using stable term hash
                // (not vocabulary index) so embeddings are reproducible
                // across palace reload cycles.
                for (dim, slot) in emb.iter_mut().enumerate() {
                    *slot += tfidf * projection_value_stable(dim, term);
                }
            }
        }

        // L2 normalize
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in emb.iter_mut() {
                *v = (*v as f64 / norm) as f32;
            }
        }

        emb
    }
}

impl Default for TfIdfEmbeddingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingEngine for TfIdfEmbeddingEngine {
    fn encode(&mut self, text: &str) -> Result<Embedding> {
        if text.is_empty() {
            return Err(GraphPalaceError::Embedding(
                "cannot encode empty text".into(),
            ));
        }

        // Learn from this document (if not frozen)
        if !self.frozen {
            self.learn_document(text);
        }

        Ok(self.compute_embedding(text))
    }

    fn batch_encode(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        // First pass: learn all documents
        if !self.frozen {
            for text in texts {
                if text.is_empty() {
                    return Err(GraphPalaceError::Embedding(
                        "cannot encode empty text".into(),
                    ));
                }
                self.learn_document(text);
            }
        }

        // Second pass: compute embeddings with updated IDF
        texts
            .iter()
            .map(|text| {
                if text.is_empty() {
                    return Err(GraphPalaceError::Embedding(
                        "cannot encode empty text".into(),
                    ));
                }
                Ok(self.compute_embedding(text))
            })
            .collect()
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::cosine_similarity;

    #[test]
    fn tokenize_basic() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // "a" and "is" are length 2, should be included
        assert!(tokens.contains(&"is".to_string()));
        // "a" is length 1, should be excluded
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn tokenize_punctuation_only() {
        let tokens = tokenize("!!! ???");
        assert!(tokens.is_empty());
    }

    #[test]
    fn new_engine_empty_vocab() {
        let engine = TfIdfEmbeddingEngine::new();
        assert_eq!(engine.vocab_size(), 0);
        assert_eq!(engine.total_docs(), 0);
    }

    #[test]
    fn encode_builds_vocab() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let _emb = engine.encode("the cat sat on the mat").unwrap();
        assert!(engine.vocab_size() > 0);
        assert_eq!(engine.total_docs(), 1);
    }

    #[test]
    fn encode_empty_text_errors() {
        let mut engine = TfIdfEmbeddingEngine::new();
        assert!(engine.encode("").is_err());
    }

    #[test]
    fn encode_dimension_is_384() {
        let engine = TfIdfEmbeddingEngine::new();
        assert_eq!(engine.dimension(), 384);
    }

    #[test]
    fn encode_produces_normalised_vectors() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let emb = engine.encode("the quick brown fox jumps over the lazy dog").unwrap();
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected unit norm, got {norm}"
        );
    }

    #[test]
    fn identical_text_produces_same_embedding() {
        let mut engine = TfIdfEmbeddingEngine::new();
        // Encode twice — second time IDF is updated but tokens are the same
        let _e0 = engine.encode("hello world test").unwrap();
        let e1 = engine.encode("hello world test").unwrap();
        let e2 = engine.encode("hello world test").unwrap();
        // After the vocabulary stabilizes, same text should produce same embedding
        assert_eq!(e1, e2, "same text must produce identical embeddings");
    }

    #[test]
    fn similar_texts_have_high_similarity() {
        let mut engine = TfIdfEmbeddingEngine::new();
        // Seed with some documents to build vocabulary
        let _ = engine.encode("the dog sat on the rug");
        let _ = engine.encode("the cat sat on the mat");
        let _ = engine.encode("quantum physics equations describe reality");
        let _ = engine.encode("the weather is nice today outside");

        // Freeze so embeddings are stable
        engine.freeze();

        let cat_emb = engine.compute_embedding("the cat sat on the mat");
        let dog_emb = engine.compute_embedding("the dog sat on the rug");
        let physics_emb = engine.compute_embedding("quantum physics equations describe reality");

        let cat_dog_sim = cosine_similarity(&cat_emb, &dog_emb);
        let cat_physics_sim = cosine_similarity(&cat_emb, &physics_emb);

        assert!(
            cat_dog_sim > cat_physics_sim,
            "cat/dog ({cat_dog_sim:.4}) should be more similar than cat/physics ({cat_physics_sim:.4})"
        );
    }

    #[test]
    fn dissimilar_texts_have_low_similarity() {
        let mut engine = TfIdfEmbeddingEngine::new();
        // Build a rich corpus so IDF values are discriminative
        let corpus = &[
            "quantum physics equations and wave functions",
            "quantum mechanics uncertainty principle",
            "particle physics standard model bosons",
            "cooking recipes for chocolate cake with frosting",
            "baking bread sourdough yeast flour",
            "grilling steak barbecue temperature",
            "neural network deep learning gradient descent",
            "machine learning training optimization",
            "gardening tips for growing tomatoes in summer",
            "planting flowers watering soil sunlight",
            "astronomy telescope galaxies nebula stars",
            "music piano violin symphony orchestra",
            "programming rust python javascript code",
            "mathematics calculus algebra geometry proofs",
            "history ancient rome empire republic",
            "geography mountains rivers oceans continents",
        ];
        for text in corpus {
            let _ = engine.encode(text);
        }
        engine.freeze();

        let physics_emb = engine.compute_embedding("quantum physics equations and wave functions");
        let cooking_emb = engine.compute_embedding("cooking recipes for chocolate cake with frosting");
        let physics2_emb = engine.compute_embedding("quantum mechanics uncertainty principle");

        let physics_cooking = cosine_similarity(&physics_emb, &cooking_emb);
        let physics_physics2 = cosine_similarity(&physics_emb, &physics2_emb);

        // Physics texts should be more similar to each other than to cooking
        assert!(
            physics_physics2 > physics_cooking,
            "physics/physics2 ({physics_physics2:.4}) should be > physics/cooking ({physics_cooking:.4})"
        );
    }

    #[test]
    fn batch_encode_consistent_with_encode() {
        // Pre-train vocabulary first, then freeze and compare
        let mut engine = TfIdfEmbeddingEngine::new();
        let _ = engine.encode("alpha beta gamma");
        let _ = engine.encode("delta epsilon zeta");
        engine.freeze();

        let texts = &["alpha beta gamma", "delta epsilon zeta"];
        let batch = engine.batch_encode(texts).unwrap();
        assert_eq!(batch.len(), 2);

        // Individual encode with frozen engine should match
        let e0 = engine.compute_embedding("alpha beta gamma");
        let e1 = engine.compute_embedding("delta epsilon zeta");

        // Compare with tolerance for f32 floating-point precision
        let sim0 = cosine_similarity(&batch[0], &e0);
        let sim1 = cosine_similarity(&batch[1], &e1);
        assert!(
            (sim0 - 1.0).abs() < 1e-5,
            "batch[0] should match individual encode, sim={sim0}"
        );
        assert!(
            (sim1 - 1.0).abs() < 1e-5,
            "batch[1] should match individual encode, sim={sim1}"
        );
    }

    #[test]
    fn batch_encode_empty_text_errors() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let result = engine.batch_encode(&["ok", ""]);
        assert!(result.is_err());
    }

    #[test]
    fn from_corpus_builds_frozen_vocab() {
        let corpus = &[
            "the cat sat on the mat",
            "the dog sat on the rug",
            "quantum physics describes reality",
        ];
        let mut engine = TfIdfEmbeddingEngine::from_corpus(corpus);
        assert!(engine.vocab_size() > 0);
        assert_eq!(engine.total_docs(), 3);

        // Encoding new text should work (OOV terms ignored)
        let emb = engine.encode("unknown words plus cat").unwrap();
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!(norm > 0.0, "should produce non-zero embedding for known terms");
    }

    #[test]
    fn freeze_prevents_vocab_growth() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let _ = engine.encode("alpha beta gamma");
        let size_before = engine.vocab_size();
        engine.freeze();
        let _ = engine.encode("delta epsilon zeta");
        assert_eq!(engine.vocab_size(), size_before, "vocab should not grow when frozen");
    }

    #[test]
    fn unfreeze_allows_vocab_growth() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let _ = engine.encode("alpha beta");
        engine.freeze();
        let size_frozen = engine.vocab_size();
        engine.unfreeze();
        let _ = engine.encode("gamma delta epsilon");
        assert!(engine.vocab_size() > size_frozen, "vocab should grow after unfreeze");
    }

    #[test]
    fn semantic_similarity_cat_mat_vs_physics() {
        // Key test: "cat sat on mat" should be closer to "dog sat on rug"
        // than to "quantum physics equations" — this is what makes it SEMANTIC
        let mut engine = TfIdfEmbeddingEngine::new();

        // Build a small corpus
        let corpus = &[
            "the cat sat on the mat",
            "the dog sat on the rug",
            "quantum physics equations and particle theory",
            "machine learning neural networks deep learning",
            "cooking recipes chocolate cake desserts",
            "the cat played with a ball of yarn",
            "the dog chased the ball across the yard",
            "quantum entanglement superposition wave collapse",
        ];
        for text in corpus {
            let _ = engine.encode(text);
        }
        engine.freeze();

        let cat_mat = engine.compute_embedding("the cat sat on the mat");
        let dog_rug = engine.compute_embedding("the dog sat on the rug");
        let quantum = engine.compute_embedding("quantum physics equations and particle theory");
        let cooking = engine.compute_embedding("cooking recipes chocolate cake desserts");

        let cat_dog = cosine_similarity(&cat_mat, &dog_rug);
        let cat_quantum = cosine_similarity(&cat_mat, &quantum);
        let cat_cooking = cosine_similarity(&cat_mat, &cooking);

        // Cat/dog share many words (the, sat, on, the) — should be high similarity
        assert!(
            cat_dog > 0.5,
            "cat/dog similarity should be > 0.5, got {cat_dog:.4}"
        );
        // Cat/quantum share almost no words — should be low
        assert!(
            cat_dog > cat_quantum,
            "cat/dog ({cat_dog:.4}) should be > cat/quantum ({cat_quantum:.4})"
        );
        assert!(
            cat_dog > cat_cooking,
            "cat/dog ({cat_dog:.4}) should be > cat/cooking ({cat_cooking:.4})"
        );
    }

    #[test]
    fn projection_values_are_deterministic() {
        let v1 = projection_value(0, 42);
        let v2 = projection_value(0, 42);
        assert_eq!(v1, v2, "same inputs should produce same projection value");
    }

    #[test]
    fn projection_values_differ_across_dims() {
        let v1 = projection_value(0, 42);
        let v2 = projection_value(1, 42);
        assert_ne!(v1, v2, "different dims should produce different values");
    }

    #[test]
    fn debug_format() {
        let engine = TfIdfEmbeddingEngine::new();
        let debug = format!("{engine:?}");
        assert!(debug.contains("TfIdfEmbeddingEngine"));
        assert!(debug.contains("vocab_size: 0"));
    }

    #[test]
    fn default_trait() {
        let engine = TfIdfEmbeddingEngine::default();
        assert_eq!(engine.vocab_size(), 0);
    }

    #[test]
    fn unicode_text_works() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let emb = engine.encode("こんにちは世界 hello world 🌍").unwrap();
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "unicode embedding should be normalised, got norm={norm}"
        );
    }

    #[test]
    fn long_text_works() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let long_text = "word ".repeat(10000);
        let emb = engine.encode(&long_text).unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
    }

    #[test]
    fn single_word_text() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let emb = engine.encode("hello").unwrap();
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4 || norm < 1e-6,
            "should be unit-normalised or zero, got {norm}"
        );
    }

    #[test]
    fn vocab_grows_incrementally() {
        let mut engine = TfIdfEmbeddingEngine::new();
        let _ = engine.encode("alpha beta");
        let v1 = engine.vocab_size();
        let _ = engine.encode("gamma delta");
        let v2 = engine.vocab_size();
        assert!(v2 > v1, "vocabulary should grow with new terms");
    }

    #[test]
    fn overlapping_text_similarity_higher_than_disjoint() {
        let mut engine = TfIdfEmbeddingEngine::new();

        // Build some vocabulary
        let _ = engine.encode("the quick brown fox jumps over");
        let _ = engine.encode("the lazy brown dog sleeps under");
        let _ = engine.encode("abstract mathematical topology proofs");
        engine.freeze();

        let a = engine.compute_embedding("the quick brown fox jumps over");
        let b = engine.compute_embedding("the lazy brown dog sleeps under");
        let c = engine.compute_embedding("abstract mathematical topology proofs");

        let ab = cosine_similarity(&a, &b);
        let ac = cosine_similarity(&a, &c);

        assert!(
            ab > ac,
            "overlapping texts ({ab:.4}) should be more similar than disjoint ({ac:.4})"
        );
    }
}
