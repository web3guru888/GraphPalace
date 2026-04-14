//! `gp-palace` — Unified GraphPalace orchestrator.
//!
//! Ties together all GraphPalace crates into a single coherent API:
//! storage, embeddings, pathfinding, stigmergy, knowledge graph, and
//! export/import. This is the main entry point for applications that
//! want a complete memory palace with semantic search, A* navigation,
//! and pheromone-guided reinforcement.

pub mod export;
pub mod lifecycle;
pub mod palace;
pub mod search;

// Re-export primary public types.
pub use export::{ImportMode, ImportStats, PalaceExport};
pub use lifecycle::{ColdSpot, HotPath, KgRelationship, PalaceStatus};
pub use palace::GraphPalace;
pub use search::{DuplicateMatch, SearchResult};
