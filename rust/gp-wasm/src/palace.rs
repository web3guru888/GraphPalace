//! High-level palace operations for WASM.
//!
//! This module will wrap the low-level `gp-core` APIs into ergonomic
//! async operations suitable for the browser environment.
//!
//! Phase 6 will add:
//! - `WasmPalaceBuilder` — configure and construct a palace with WASM-compatible storage
//! - `WasmPersistence` — IndexedDB / OPFS adapter implementing gp-core's storage trait
//! - `WasmWorkerBridge` — communicate with Web Workers for background indexing
//! - Streaming search results via `ReadableStream`
