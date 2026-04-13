//! Embedding engine trait and mock implementation.

use std::path::Path;

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

/// Create the best available embedding engine.
///
/// If the `onnx` feature is enabled **and** the specified `model_dir`
/// contains `model.onnx` + `tokenizer.json`, returns an
/// [`OnnxEmbeddingEngine`](crate::onnx::OnnxEmbeddingEngine).
/// Otherwise falls back to [`MockEmbeddingEngine`].
///
/// Pass `None` to always get a mock engine (useful for tests).
///
/// # Examples
///
/// ```rust
/// use gp_embeddings::engine::auto_engine;
///
/// let mut engine = auto_engine(None);
/// let emb = engine.encode("test").unwrap();
/// assert_eq!(emb.len(), 384);
/// ```
pub fn auto_engine(model_dir: Option<&Path>) -> Box<dyn EmbeddingEngine> {
    #[cfg(feature = "onnx")]
    if let Some(dir) = model_dir
        && dir.join("model.onnx").exists()
        && dir.join("tokenizer.json").exists()
        && let Ok(engine) = crate::onnx::OnnxEmbeddingEngine::from_pretrained(dir)
    {
        return Box::new(engine);
    }

    // Suppress unused-variable warning when `onnx` feature is disabled.
    let _ = model_dir;

    #[cfg(feature = "tfidf")]
    {
        return Box::new(crate::tfidf::TfIdfEmbeddingEngine::new());
    }

    #[allow(unreachable_code)]
    Box::new(MockEmbeddingEngine::new())
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

    // -------------------------------------------------------------------
    // auto_engine tests
    // -------------------------------------------------------------------

    #[test]
    fn auto_engine_none_returns_mock() {
        let mut engine = auto_engine(None);
        let emb = engine.encode("auto_engine test").unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
    }

    #[test]
    fn auto_engine_nonexistent_dir_returns_mock() {
        let mut engine = auto_engine(Some(Path::new("/nonexistent/model/dir")));
        let emb = engine.encode("fallback test").unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
    }

    #[test]
    fn auto_engine_empty_dir_returns_mock() {
        let dir = std::env::temp_dir().join("gp_auto_engine_empty");
        let _ = std::fs::create_dir_all(&dir);
        let mut engine = auto_engine(Some(&dir));
        let emb = engine.encode("empty dir").unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_engine_dimension_is_384() {
        let engine = auto_engine(None);
        assert_eq!(engine.dimension(), 384);
    }

    #[test]
    fn auto_engine_batch_encode_works() {
        let mut engine = auto_engine(None);
        let results = engine.batch_encode(&["alpha", "beta"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_ne!(results[0], results[1]);
    }

    #[test]
    fn mock_engine_default_trait() {
        let engine = MockEmbeddingEngine;
        assert_eq!(engine.dimension(), EMBEDDING_DIM);
    }

    #[test]
    fn mock_engine_unicode_text() {
        let mut engine = MockEmbeddingEngine::new();
        let emb = engine.encode("こんにちは世界 🌍").unwrap();
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn mock_engine_long_text() {
        let mut engine = MockEmbeddingEngine::new();
        let long = "a".repeat(100_000);
        let emb = engine.encode(&long).unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
    }

    #[test]
    fn mock_engine_single_char() {
        let mut engine = MockEmbeddingEngine::new();
        let emb = engine.encode("x").unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn auto_engine_encode_rejects_empty() {
        let mut engine = auto_engine(None);
        assert!(engine.encode("").is_err());
    }
}
