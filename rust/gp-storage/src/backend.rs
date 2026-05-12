//! Storage backend trait for GraphPalace.
//!
//! Defines the [`StorageBackend`] trait that abstracts over different storage
//! implementations (Kuzu FFI, in-memory, etc.).

use gp_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Value type for query results
// ---------------------------------------------------------------------------

/// A dynamically-typed value returned from storage queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    /// A list of floats (used for embeddings).
    FloatList(Vec<f32>),
    /// A list of values (generic).
    List(Vec<Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "NULL"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::String(s) => write!(f, "{s}"),
            Value::FloatList(v) => write!(f, "{v:?}"),
            Value::List(v) => write!(f, "{v:?}"),
        }
    }
}

impl Value {
    /// Try to extract as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to extract as i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to extract as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to extract as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract as a float list (embedding).
    pub fn as_float_list(&self) -> Option<&[f32]> {
        match self {
            Value::FloatList(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// StorageBackend trait
// ---------------------------------------------------------------------------

/// The core storage abstraction for GraphPalace.
///
/// Implementations must be `Send + Sync` for use in concurrent contexts.
/// The default backend for testing is [`InMemoryBackend`](crate::InMemoryBackend).
pub trait StorageBackend: Send + Sync {
    /// Execute a read query, returning rows as vectors of column-name → value maps.
    fn execute_query(&self, cypher: &str) -> Result<Vec<HashMap<String, Value>>>;

    /// Execute a write query, returning the number of rows affected.
    fn execute_write(&self, cypher: &str) -> Result<u64>;

    /// Initialise the palace schema (create node tables, rel tables, indexes).
    fn init_schema(&self) -> Result<()>;

    /// Execute a write query and return any single-column string result.
    /// Default implementation delegates to `execute_query`.
    fn execute_returning(&self, cypher: &str) -> Result<Option<String>> {
        let rows = self.execute_query(cypher)?;
        if let Some(row) = rows.first() {
            if let Some((_, val)) = row.iter().next() {
                return Ok(Some(val.to_string()));
            }
        }
        Ok(None)
    }
}
