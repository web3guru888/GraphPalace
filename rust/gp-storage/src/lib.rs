//! `gp-storage` — Storage backends for GraphPalace.
//!
//! Provides both a safe Rust wrapper around Kuzu's C API (behind the `kuzu-ffi`
//! feature flag) and a pure-Rust in-memory backend for testing and WASM builds.

#[cfg(feature = "kuzu-ffi")]
pub mod ffi;

#[cfg(feature = "kuzu-ffi")]
pub mod safe;

pub mod backend;
pub mod hnsw;
pub mod memory;
pub mod palace_ops;
pub mod schema_init;

// Re-exports
pub use backend::{StorageBackend, Value};
pub use memory::InMemoryBackend;
pub use palace_ops::*;
pub use schema_init::init_schema;
