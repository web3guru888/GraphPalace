//! Schema initialization for GraphPalace storage.
//!
//! Uses `gp_core::schema::schema_ddl()` to generate and execute the full
//! palace schema DDL, creating all 7 node tables, 11 edge tables, and indexes.

use crate::backend::StorageBackend;
use gp_core::Result;

/// Initialize the full palace schema on the given backend.
///
/// For the [`InMemoryBackend`], this just sets a flag. For the Kuzu FFI
/// backend, it executes all DDL statements from `gp_core::schema::schema_ddl()`.
pub fn init_schema(backend: &dyn StorageBackend) -> Result<()> {
    backend.init_schema()
}

/// Get the DDL statements that would be executed.
pub fn schema_statements() -> Vec<&'static str> {
    gp_core::schema::schema_ddl()
}

/// Get the pheromone decay DDL statements.
pub fn decay_statements() -> Vec<&'static str> {
    gp_core::schema::decay_statements()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryBackend;

    #[test]
    fn test_init_schema() {
        let b = InMemoryBackend::new();
        init_schema(&b).unwrap();
        assert!(b.read_data().schema_initialized);
    }

    #[test]
    fn test_schema_statements_not_empty() {
        let stmts = schema_statements();
        assert!(!stmts.is_empty());
        // Should have at least 7 CREATE NODE TABLE + 11 CREATE REL TABLE
        assert!(stmts.len() >= 18);
    }

    #[test]
    fn test_decay_statements_exist() {
        let stmts = decay_statements();
        assert!(!stmts.is_empty());
    }

    #[test]
    fn test_double_init_is_ok() {
        let b = InMemoryBackend::new();
        init_schema(&b).unwrap();
        init_schema(&b).unwrap(); // should not error
    }
}
