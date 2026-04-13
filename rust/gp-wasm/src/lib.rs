//! `gp-wasm` — WebAssembly bindings for GraphPalace.
//!
//! This crate will provide wasm-bindgen entry points for running
//! GraphPalace in the browser. Currently contains type stubs.
//!
//! Phase 6 will add:
//! - wasm-bindgen annotations
//! - JavaScript-facing API
//! - IndexedDB/OPFS persistence
//! - Web Worker integration

pub mod js_api;
pub mod palace;
