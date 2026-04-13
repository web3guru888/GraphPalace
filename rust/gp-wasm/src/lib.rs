//! `gp-wasm` — WebAssembly bindings for GraphPalace.
//!
//! Provides a complete in-memory palace implementation and browser
//! integration types including:
//!
//! - [`palace`] — In-memory palace with full CRUD operations
//! - [`js_api`] — JavaScript-facing API (GraphPalaceWasm)
//! - [`worker`] — Web Worker message types for async operations
//! - [`persistence`] — Storage backend traits (Memory, IndexedDB, OPFS)

pub mod js_api;
pub mod palace;
pub mod persistence;
pub mod worker;

pub use js_api::GraphPalaceWasm;
pub use palace::InMemoryPalace;
pub use persistence::{ImportMode, MemoryPersistence, PalacePersistence, StorageBackend};
pub use worker::{WorkerRequest, WorkerResponse};
