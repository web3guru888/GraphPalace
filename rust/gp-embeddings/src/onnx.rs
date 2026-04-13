//! Real ONNX-based embedding engine using sentence-transformers models.
//!
//! This module provides [`OnnxEmbeddingEngine`], a production-quality embedding
//! engine that runs ONNX models (e.g. `all-MiniLM-L6-v2`) through the
//! [ort](https://docs.rs/ort) runtime and the HuggingFace
//! [tokenizers](https://docs.rs/tokenizers) crate.
//!
//! # Feature gate
//!
//! Everything in this module is gated behind the `onnx` feature flag:
//!
//! ```toml
//! gp-embeddings = { path = "gp-embeddings", features = ["onnx"] }
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "onnx")]
//! # {
//! use gp_embeddings::OnnxEmbeddingEngine;
//! use std::path::Path;
//!
//! let mut engine = OnnxEmbeddingEngine::from_pretrained(
//!     Path::new("models/all-MiniLM-L6-v2"),
//! ).expect("load model");
//!
//! let emb = engine.encode("Hello, world!").unwrap();
//! assert_eq!(emb.len(), 384);
//! # }
//! ```

use std::path::Path;

use ort::{session::Session, value::Tensor};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use gp_core::{Embedding, GraphPalaceError, Result, EMBEDDING_DIM};

use crate::EmbeddingEngine;

/// Maximum input sequence length in tokens. Inputs longer than this are
/// truncated by the tokenizer.
const MAX_SEQ_LEN: usize = 256;

/// Number of intra-op threads used by the ONNX runtime.
const INTRA_THREADS: usize = 4;

/// A real embedding engine backed by an ONNX model and HuggingFace tokenizer.
///
/// Typical workflow:
/// 1. Load with [`OnnxEmbeddingEngine::from_pretrained`] (expects a directory
///    containing `model.onnx` and `tokenizer.json`).
/// 2. Call [`encode`](EmbeddingEngine::encode) or
///    [`batch_encode`](EmbeddingEngine::batch_encode).
///
/// The engine performs **mean pooling** over non-padding tokens followed by
/// **L2 normalisation**, producing unit-length embeddings of dimension
/// [`EMBEDDING_DIM`] (384).
pub struct OnnxEmbeddingEngine {
    session: Session,
    tokenizer: Tokenizer,
}

impl std::fmt::Debug for OnnxEmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEmbeddingEngine")
            .field("session", &"<ort::Session>")
            .field("tokenizer", &"<tokenizers::Tokenizer>")
            .finish()
    }
}

impl OnnxEmbeddingEngine {
    /// Create an engine from explicit model and tokenizer file paths.
    ///
    /// # Errors
    ///
    /// Returns [`GraphPalaceError::Embedding`] if either file cannot be loaded
    /// or the ONNX session fails to initialise.
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| GraphPalaceError::Embedding(format!("session builder: {e}")))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| GraphPalaceError::Embedding(format!("optimization level: {e}")))?
            .with_intra_threads(INTRA_THREADS)
            .map_err(|e| GraphPalaceError::Embedding(format!("intra threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                GraphPalaceError::Embedding(format!(
                    "load model {}: {e}",
                    model_path.display()
                ))
            })?;

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            GraphPalaceError::Embedding(format!(
                "load tokenizer {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        Ok(Self { session, tokenizer })
    }

    /// Create an engine from a model directory that contains both
    /// `model.onnx` and `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// Returns [`GraphPalaceError::Embedding`] if the expected files are
    /// missing or cannot be loaded.
    pub fn from_pretrained(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(GraphPalaceError::Embedding(format!(
                "model.onnx not found in {}",
                model_dir.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(GraphPalaceError::Embedding(format!(
                "tokenizer.json not found in {}",
                model_dir.display()
            )));
        }

        Self::new(&model_path, &tokenizer_path)
    }

    /// Encode a single text into an embedding using the ONNX model.
    ///
    /// Steps: tokenize → build input tensors → run ONNX inference →
    /// mean-pool over non-padding tokens → L2-normalise.
    fn encode_single(&mut self, text: &str) -> Result<Embedding> {
        // Tokenize with truncation.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| GraphPalaceError::Embedding(format!("tokenize: {e}")))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let seq_len = ids.len().min(MAX_SEQ_LEN);

        // Build 1-D vectors, then create ort Tensors with shape [1, seq_len].
        let id_vec: Vec<i64> = ids[..seq_len].iter().map(|&x| x as i64).collect();
        let mask_vec: Vec<i64> = mask[..seq_len].iter().map(|&x| x as i64).collect();

        let token_type_vec: Vec<i64> = vec![0i64; seq_len];

        let input_ids = Tensor::from_array(([1usize, seq_len], id_vec.into_boxed_slice()))
            .map_err(|e| GraphPalaceError::Embedding(format!("create input_ids tensor: {e}")))?;
        let attention_mask =
            Tensor::from_array(([1usize, seq_len], mask_vec.into_boxed_slice()))
                .map_err(|e| {
                    GraphPalaceError::Embedding(format!("create attention_mask tensor: {e}"))
                })?;
        let token_type_ids =
            Tensor::from_array(([1usize, seq_len], token_type_vec.into_boxed_slice()))
                .map_err(|e| {
                    GraphPalaceError::Embedding(format!("create token_type_ids tensor: {e}"))
                })?;

        // Run inference.
        let outputs = self
            .session
            .run(ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids
            })
            .map_err(|e| GraphPalaceError::Embedding(format!("inference: {e}")))?;

        // Extract token embeddings: flat slice with shape [1, seq_len, hidden_dim].
        let output_val = &outputs["last_hidden_state"];
        let (shape, data) = output_val.try_extract_tensor::<f32>().map_err(|e| {
            GraphPalaceError::Embedding(format!("extract tensor: {e}"))
        })?;

        let hidden_dim = *shape.last().unwrap_or(&0) as usize;
        if hidden_dim != EMBEDDING_DIM {
            return Err(GraphPalaceError::InvalidEmbeddingDim {
                expected: EMBEDDING_DIM,
                got: hidden_dim,
            });
        }

        // Mean-pool over non-padding tokens.
        mean_pool_and_normalise(data, &mask[..seq_len], seq_len, hidden_dim)
    }

    /// Efficient batch encoding: pad to max batch length, run a single
    /// inference call.
    fn encode_batch_inner(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Configure tokenizer for batch mode.
        self.tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));
        let _ = self
            .tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..TruncationParams::default()
            }))
            .map_err(|e| GraphPalaceError::Embedding(format!("truncation config: {e}")))?;

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| GraphPalaceError::Embedding(format!("batch tokenize: {e}")))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        // Build flat vectors for [batch_size, seq_len] tensors.
        let mut id_vec = Vec::with_capacity(batch_size * seq_len);
        let mut mask_vec = Vec::with_capacity(batch_size * seq_len);
        for enc in &encodings {
            for &id in enc.get_ids() {
                id_vec.push(id as i64);
            }
            for &m in enc.get_attention_mask() {
                mask_vec.push(m as i64);
            }
        }

        let token_type_vec: Vec<i64> = vec![0i64; batch_size * seq_len];

        let input_ids =
            Tensor::from_array(([batch_size, seq_len], id_vec.into_boxed_slice())).map_err(
                |e| GraphPalaceError::Embedding(format!("create batch input_ids tensor: {e}")),
            )?;
        let attention_mask =
            Tensor::from_array(([batch_size, seq_len], mask_vec.into_boxed_slice())).map_err(
                |e| {
                    GraphPalaceError::Embedding(format!(
                        "create batch attention_mask tensor: {e}"
                    ))
                },
            )?;
        let token_type_ids =
            Tensor::from_array(([batch_size, seq_len], token_type_vec.into_boxed_slice()))
                .map_err(|e| {
                    GraphPalaceError::Embedding(format!(
                        "create batch token_type_ids tensor: {e}"
                    ))
                })?;

        // Run inference.
        let outputs = self
            .session
            .run(ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids
            })
            .map_err(|e| GraphPalaceError::Embedding(format!("batch inference: {e}")))?;

        // Extract: flat slice with shape [batch_size, seq_len, hidden_dim].
        let output_val = &outputs["last_hidden_state"];
        let (shape, data) = output_val.try_extract_tensor::<f32>().map_err(|e| {
            GraphPalaceError::Embedding(format!("extract batch tensor: {e}"))
        })?;

        let hidden_dim = *shape.last().unwrap_or(&0) as usize;
        if hidden_dim != EMBEDDING_DIM {
            return Err(GraphPalaceError::InvalidEmbeddingDim {
                expected: EMBEDDING_DIM,
                got: hidden_dim,
            });
        }

        // Mean-pool each sample individually.
        let stride = seq_len * hidden_dim;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let sample_data = &data[i * stride..(i + 1) * stride];
            let sample_mask: Vec<u32> =
                encodings[i].get_attention_mask().to_vec();
            let emb =
                mean_pool_and_normalise(sample_data, &sample_mask, seq_len, hidden_dim)?;
            results.push(emb);
        }

        // Reset tokenizer padding so future single-encode calls aren't padded.
        self.tokenizer.with_padding(None);

        Ok(results)
    }
}

impl EmbeddingEngine for OnnxEmbeddingEngine {
    fn encode(&mut self, text: &str) -> Result<Embedding> {
        if text.is_empty() {
            return Err(GraphPalaceError::Embedding(
                "cannot encode empty text".into(),
            ));
        }
        self.encode_single(text)
    }

    fn batch_encode(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        for t in texts {
            if t.is_empty() {
                return Err(GraphPalaceError::Embedding(
                    "cannot encode empty text in batch".into(),
                ));
            }
        }
        self.encode_batch_inner(texts)
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Mean-pool token embeddings (respecting the attention mask) and L2-normalise
/// the result.
///
/// `data` is a flat slice of shape `[seq_len, hidden_dim]` (one sample).
/// `mask` contains `1` for real tokens and `0` for padding.
fn mean_pool_and_normalise(
    data: &[f32],
    mask: &[u32],
    seq_len: usize,
    hidden_dim: usize,
) -> Result<Embedding> {
    let mut sum = [0.0f64; EMBEDDING_DIM];
    let mut count = 0.0f64;

    for t in 0..seq_len {
        if t < mask.len() && mask[t] == 0 {
            continue;
        }
        count += 1.0;
        let offset = t * hidden_dim;
        for d in 0..hidden_dim.min(EMBEDDING_DIM) {
            sum[d] += data[offset + d] as f64;
        }
    }

    if count == 0.0 {
        return Err(GraphPalaceError::Embedding(
            "all tokens masked — cannot mean-pool".into(),
        ));
    }

    // Mean.
    let mut emb = [0.0f32; EMBEDDING_DIM];
    let mut norm_sq = 0.0f64;
    for d in 0..EMBEDDING_DIM {
        let v = sum[d] / count;
        norm_sq += v * v;
        emb[d] = v as f32;
    }

    // L2 normalise.
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        for v in emb.iter_mut() {
            *v = (*v as f64 / norm) as f32;
        }
    }

    Ok(emb)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Unit tests that don't require a real ONNX model
    // -------------------------------------------------------------------

    #[test]
    fn max_seq_len_is_256() {
        assert_eq!(MAX_SEQ_LEN, 256);
    }

    #[test]
    fn embedding_dim_matches() {
        assert_eq!(EMBEDDING_DIM, 384);
    }

    #[test]
    fn mean_pool_basic_produces_unit_vector() {
        // Two "tokens", all dimensions the same → after normalise, should be
        // a unit vector.
        let mut data = vec![0.0f32; 2 * EMBEDDING_DIM];
        for d in 0..EMBEDDING_DIM {
            data[d] = (d as f32 + 1.0) * 0.01;
            data[EMBEDDING_DIM + d] = (d as f32 + 1.0) * 0.01;
        }
        let mask = [1u32, 1];
        let emb = mean_pool_and_normalise(&data, &mask, 2, EMBEDDING_DIM).unwrap();
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected unit vector, got norm={norm}"
        );
    }

    #[test]
    fn mean_pool_respects_mask() {
        // Token 0 has meaningful values, token 1 has 999.0 (noise).
        // With mask = [1, 0], only token 0 should contribute.
        let mut data = vec![0.0f32; 2 * EMBEDDING_DIM];
        for d in 0..EMBEDDING_DIM {
            data[d] = (d as f32 + 1.0) * 0.1;
            data[EMBEDDING_DIM + d] = 999.0;
        }
        let mask = [1u32, 0];
        let emb = mean_pool_and_normalise(&data, &mask, 2, EMBEDDING_DIM).unwrap();

        // Compute expected normalised token-0.
        let raw: Vec<f64> = (0..EMBEDDING_DIM)
            .map(|d| (d as f64 + 1.0) * 0.1)
            .collect();
        let raw_norm: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        for d in 0..EMBEDDING_DIM {
            let expected = (raw[d] / raw_norm) as f32;
            assert!(
                (emb[d] - expected).abs() < 1e-5,
                "dim {d}: expected {expected}, got {}",
                emb[d]
            );
        }
    }

    #[test]
    fn mean_pool_all_masked_errors() {
        let data = vec![1.0f32; 3 * EMBEDDING_DIM];
        let mask = [0u32, 0, 0];
        let result = mean_pool_and_normalise(&data, &mask, 3, EMBEDDING_DIM);
        assert!(result.is_err());
    }

    #[test]
    fn mean_pool_single_token() {
        let mut data = vec![0.0f32; EMBEDDING_DIM];
        for d in 0..EMBEDDING_DIM {
            data[d] = (d as f32) * 0.05;
        }
        let mask = [1u32];
        let emb = mean_pool_and_normalise(&data, &mask, 1, EMBEDDING_DIM).unwrap();
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn from_pretrained_missing_model_errors() {
        let dir = std::env::temp_dir().join("gp_onnx_test_missing_model");
        let _ = std::fs::create_dir_all(&dir);
        let result = OnnxEmbeddingEngine::from_pretrained(&dir);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("model.onnx"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_pretrained_missing_tokenizer_errors() {
        let dir = std::env::temp_dir().join("gp_onnx_test_missing_tok");
        let _ = std::fs::create_dir_all(&dir);
        // Create a dummy model.onnx so we get past the first check.
        let _ = std::fs::write(dir.join("model.onnx"), b"dummy");
        let result = OnnxEmbeddingEngine::from_pretrained(&dir);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("tokenizer.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn intra_threads_constant() {
        assert_eq!(INTRA_THREADS, 4);
    }

    #[test]
    fn mean_pool_average_is_correct() {
        // Two tokens: [1,2,3,...,384] and [3,4,5,...,386].
        // Mean should be [2,3,4,...,385], then normalised.
        let mut data = vec![0.0f32; 2 * EMBEDDING_DIM];
        for d in 0..EMBEDDING_DIM {
            data[d] = (d + 1) as f32;
            data[EMBEDDING_DIM + d] = (d + 3) as f32;
        }
        let mask = [1u32, 1];
        let emb = mean_pool_and_normalise(&data, &mask, 2, EMBEDDING_DIM).unwrap();

        // The mean values before normalisation.
        let mean_vals: Vec<f64> = (0..EMBEDDING_DIM)
            .map(|d| ((d + 1) as f64 + (d + 3) as f64) / 2.0)
            .collect();
        let norm: f64 = mean_vals.iter().map(|x| x * x).sum::<f64>().sqrt();
        for d in 0..EMBEDDING_DIM {
            let expected = (mean_vals[d] / norm) as f32;
            assert!(
                (emb[d] - expected).abs() < 1e-4,
                "dim {d}: expected {expected}, got {}",
                emb[d]
            );
        }
    }
}
