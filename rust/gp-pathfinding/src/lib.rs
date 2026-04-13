//! Pheromone-weighted pathfinding for GraphPalace.
//!
//! A*, Dijkstra, and ant-colony-inspired traversal over stigmergic graphs.
//!
//! # Modules
//!
//! - [`edge_cost`] — Composite cost model combining semantic, pheromone,
//!   and structural components (spec §5.1–5.2).
//! - [`heuristic`] — Adaptive heuristic that shifts between cross-domain
//!   and same-domain modes based on semantic similarity (spec §5.3).
//! - [`provenance`] — Data structures for path result provenance tracking.
//! - [`astar`] — Semantic A* implementation using the above components
//!   (spec §5.4).

pub mod astar;
pub mod edge_cost;
pub mod heuristic;
pub mod provenance;

// Re-export primary public API.
pub use astar::{GraphAccess, SemanticAStar};
pub use edge_cost::{composite_edge_cost, pheromone_cost, semantic_cost, structural_cost_for};
pub use heuristic::semantic_heuristic;
pub use provenance::{PathResult, ProvenanceStep};
