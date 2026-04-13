//! Embedding engine trait and mock implementation.

use gp_core::{Embedding, GraphPalaceError, Result, EMBEDDING_DIM};

/// Trait for producing fixed-dimension embeddings from text.
pub trait EmbeddingEngine {
    /// Encode a single text string into a fixed-size embedding vector.
    fn encode(&mut self, text: &str) -> Result<Embedding>;

    /// Encode multiple texts in a batch. Default implementation calls `encode`
    /// for each text sequentially.
    fn batch_encode(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Return the dimensionality of the embeddings produced by this engine.
    fn dimension(&self) -> usize;
}

/// A deterministic mock embedding engine that hashes text to produce
/// reproducible pseudo-random embedding vectors. Same text always
/// produces the same embedding.
///
/// Uses a simple FNV-1a–style hash seeded per dimension to generate
/// each component, then L2-normalises the result.
pub struct MockEmbeddingEngine;

impl MockEmbeddingEngine {
    pub fn new() -> Self {
        Self
    }

    /// Deterministic hash of `text` to produce an embedding.
    fn hash_embed(text: &str) -> Embedding {
        let mut emb = [0.0f32; EMBEDDING_DIM];
        let bytes = text.as_bytes();

        for (i, slot) in emb.iter_mut().enumerate() {
            // FNV-1a with a per-dimension seed.
            let mut h: u64 = 0xcbf2_9ce4_8422_2325_u64.wrapping_add(i as u64 * 0x100_0193);
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            // Map to [-1, 1] range.
            *slot = ((h as i64) as f64 / i64::MAX as f64) as f32;
        }

        // L2 normalise so cosine similarity is just dot product.
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in emb.iter_mut() {
                *v = (*v as f64 / norm) as f32;
            }
        }

        emb
    }
}

impl Default for MockEmbeddingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingEngine for MockEmbeddingEngine {
    fn encode(&mut self, text: &str) -> Result<Embedding> {
        if text.is_empty() {
            return Err(GraphPalaceError::Embedding(
                "cannot encode empty text".into(),
            ));
        }
        Ok(Self::hash_embed(text))
    }

    fn batch_encode(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_engine_deterministic() {
        let mut engine = MockEmbeddingEngine::new();
        let e1 = engine.encode("hello world").unwrap();
        let e2 = engine.encode("hello world").unwrap();
        assert_eq!(e1, e2, "same text must produce identical embeddings");
    }

    #[test]
    fn mock_engine_different_texts_differ() {
        let mut engine = MockEmbeddingEngine::new();
        let e1 = engine.encode("hello").unwrap();
        let e2 = engine.encode("goodbye").unwrap();
        assert_ne!(e1, e2, "different text should produce different embeddings");
    }

    #[test]
    fn mock_engine_dimension() {
        let engine = MockEmbeddingEngine::new();
        assert_eq!(engine.dimension(), EMBEDDING_DIM);
        assert_eq!(engine.dimension(), 384);
    }

    #[test]
    fn mock_engine_normalised() {
        let mut engine = MockEmbeddingEngine::new();
        let emb = engine.encode("test normalisation").unwrap();
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "embedding should be unit-normalised, got norm={norm}"
        );
    }

    #[test]
    fn mock_engine_empty_text_errors() {
        let mut engine = MockEmbeddingEngine::new();
        assert!(engine.encode("").is_err());
    }

    #[test]
    fn mock_engine_batch_encode() {
        let mut engine = MockEmbeddingEngine::new();
        let results = engine.batch_encode(&["foo", "bar", "baz"]).unwrap();
        assert_eq!(results.len(), 3);
        // Verify determinism by re-encoding individually.
        assert_eq!(results[0], engine.encode("foo").unwrap());
        assert_eq!(results[1], engine.encode("bar").unwrap());
        assert_eq!(results[2], engine.encode("baz").unwrap());
    }

    #[test]
    fn mock_engine_batch_encode_empty_text_errors() {
        let mut engine = MockEmbeddingEngine::new();
        let result = engine.batch_encode(&["ok", ""]);
        assert!(result.is_err());
    }
}
