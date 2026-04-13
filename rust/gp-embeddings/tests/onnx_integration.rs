//! Integration tests for the ONNX embedding engine.
//!
//! These tests require a real ONNX model. To run them:
//!
//! 1. Download the model:
//!    ```bash
//!    mkdir -p models/all-MiniLM-L6-v2
//!    wget -O models/all-MiniLM-L6-v2/model.onnx \
//!      https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
//!    wget -O models/all-MiniLM-L6-v2/tokenizer.json \
//!      https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
//!    ```
//!
//! 2. Set the environment variable:
//!    ```bash
//!    export GP_ONNX_MODEL_DIR=models/all-MiniLM-L6-v2
//!    ```
//!
//! 3. Run:
//!    ```bash
//!    cargo test -p gp-embeddings --features onnx -- --test-threads=1
//!    ```

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use gp_embeddings::{EmbeddingEngine, OnnxEmbeddingEngine};
use gp_embeddings::similarity::cosine_similarity;

/// Resolve the model directory from the `GP_ONNX_MODEL_DIR` environment
/// variable. Returns `None` if unset or the directory doesn't contain the
/// required files.
fn model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("GP_ONNX_MODEL_DIR").ok()?);
    if dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists() {
        Some(dir)
    } else {
        None
    }
}

macro_rules! skip_without_model {
    () => {
        match model_dir() {
            Some(d) => d,
            None => {
                eprintln!(
                    "SKIPPED: set GP_ONNX_MODEL_DIR to a directory with model.onnx + tokenizer.json"
                );
                return;
            }
        }
    };
}

#[test]
fn onnx_engine_produces_384_dim() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    let emb = engine.encode("Hello world").unwrap();
    assert_eq!(emb.len(), 384);
}

#[test]
fn onnx_engine_unit_normalised() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    let emb = engine.encode("Unit normalisation test").unwrap();
    let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "expected unit vector, got norm={norm}"
    );
}

#[test]
fn onnx_engine_deterministic() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    let e1 = engine.encode("deterministic check").unwrap();
    let e2 = engine.encode("deterministic check").unwrap();
    assert_eq!(e1, e2);
}

#[test]
fn onnx_engine_similar_texts_high_sim() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    let e1 = engine.encode("The cat sat on the mat").unwrap();
    let e2 = engine.encode("A cat was sitting on a mat").unwrap();
    let sim = cosine_similarity(&e1, &e2);
    assert!(
        sim > 0.75,
        "similar sentences should have high similarity, got {sim}"
    );
}

#[test]
fn onnx_engine_different_texts_lower_sim() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    let e1 = engine.encode("The cat sat on the mat").unwrap();
    let e2 = engine.encode("Quantum computing is a new paradigm").unwrap();
    let sim = cosine_similarity(&e1, &e2);
    assert!(
        sim < 0.5,
        "unrelated sentences should have lower similarity, got {sim}"
    );
}

#[test]
fn onnx_engine_batch_encode() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    let texts = &["hello", "world", "foo bar"];
    let batch = engine.batch_encode(texts).unwrap();
    assert_eq!(batch.len(), 3);
    for emb in &batch {
        assert_eq!(emb.len(), 384);
        let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }
}

#[test]
fn onnx_engine_batch_matches_single() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    let texts = &["alpha", "beta"];
    let batch = engine.batch_encode(texts).unwrap();
    let single_a = engine.encode("alpha").unwrap();
    let single_b = engine.encode("beta").unwrap();
    // Batch and single results should be very close (not necessarily identical
    // due to padding differences).
    let sim_a = cosine_similarity(&batch[0], &single_a);
    let sim_b = cosine_similarity(&batch[1], &single_b);
    assert!(sim_a > 0.99, "batch vs single alpha: {sim_a}");
    assert!(sim_b > 0.99, "batch vs single beta: {sim_b}");
}

#[test]
fn onnx_engine_empty_text_errors() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    assert!(engine.encode("").is_err());
}

#[test]
fn onnx_engine_dimension() {
    let dir = skip_without_model!();
    let engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    assert_eq!(engine.dimension(), 384);
}

#[test]
fn onnx_engine_long_text_truncates() {
    let dir = skip_without_model!();
    let mut engine = OnnxEmbeddingEngine::from_pretrained(&dir).unwrap();
    let long = "word ".repeat(1000);
    let emb = engine.encode(&long).unwrap();
    assert_eq!(emb.len(), 384);
    let norm: f64 = emb.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3);
}
