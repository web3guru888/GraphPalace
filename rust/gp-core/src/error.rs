//! Error types for GraphPalace operations.

use thiserror::Error;

/// Top-level error type for all GraphPalace operations.
#[derive(Debug, Error)]
pub enum GraphPalaceError {
    #[error("Node not found: {id}")]
    NodeNotFound { id: String },

    #[error("Edge not found: {from} -> {to}")]
    EdgeNotFound { from: String, to: String },

    #[error("Wing not found: {name}")]
    WingNotFound { name: String },

    #[error("Room not found: {name} in wing {wing}")]
    RoomNotFound { name: String, wing: String },

    #[error("Duplicate node: {id}")]
    DuplicateNode { id: String },

    #[error("Invalid embedding dimension: expected {expected}, got {got}")]
    InvalidEmbeddingDim { expected: usize, got: usize },

    #[error("Schema error: {0}")]
    Schema(String),

    #[error("Pheromone error: {0}")]
    Pheromone(String),

    #[error("Pathfinding error: {0}")]
    Pathfinding(String),

    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Max iterations reached: {0}")]
    MaxIterations(usize),

    #[error("Invalid parameter: {param} = {value} ({reason})")]
    InvalidParameter {
        param: String,
        value: String,
        reason: String,
    },
}

/// Result type alias for GraphPalace operations.
pub type Result<T> = std::result::Result<T, GraphPalaceError>;
