//! `gp-core` — Core types for GraphPalace.
//!
//! This crate defines all node types, edge types, pheromone fields,
//! configuration, errors, and the Cypher DDL schema for GraphPalace.

pub mod config;
pub mod error;
pub mod schema;
pub mod types;

// Re-export commonly used items at crate root.
pub use config::*;
pub use error::{GraphPalaceError, Result};
pub use types::*;
