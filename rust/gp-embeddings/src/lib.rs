//! Embedding vectors and similarity for GraphPalace.
//!
//! Cosine distance, vector quantisation, and embedding storage.
//!
//! # Feature flags
//!
//! | Feature    | Description |
//! |------------|-------------|
//! | `tfidf`    | Real semantic TF-IDF embeddings (pure Rust, no model files) |
//! | `onnx`     | Real ONNX inference via `ort` + HuggingFace `tokenizers` |
//! | `download` | Download models from HuggingFace Hub (implies `onnx`) |

pub mod cache;
pub mod engine;
pub mod similarity;
pub mod tfidf;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "download")]
pub mod download;

// Re-export main public types at crate root.
pub use cache::EmbeddingCache;
pub use engine::{EmbeddingEngine, MockEmbeddingEngine};
pub use similarity::{cosine_similarity, find_top_k};
pub use tfidf::TfIdfEmbeddingEngine;

#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbeddingEngine;
