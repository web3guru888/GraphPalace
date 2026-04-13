//! Embedding vectors and similarity for GraphPalace.
//!
//! Cosine distance, vector quantisation, and embedding storage.
//!
//! # Feature flags
//!
//! | Feature    | Description |
//! |------------|-------------|
//! | `onnx`     | Real ONNX inference via `ort` + HuggingFace `tokenizers` |
//! | `download` | Download models from HuggingFace Hub (implies `onnx`) |

pub mod cache;
pub mod engine;
pub mod similarity;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "download")]
pub mod download;

// Re-export main public types at crate root.
pub use cache::EmbeddingCache;
pub use engine::{EmbeddingEngine, MockEmbeddingEngine};
pub use similarity::{cosine_similarity, find_top_k};

#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbeddingEngine;
