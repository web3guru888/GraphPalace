//! Embedding vectors and similarity for GraphPalace.
//!
//! Cosine distance, vector quantisation, and embedding storage.

pub mod cache;
pub mod engine;
pub mod similarity;

// Re-export main public types at crate root.
pub use cache::EmbeddingCache;
pub use engine::{EmbeddingEngine, MockEmbeddingEngine};
pub use similarity::{cosine_similarity, find_top_k};
