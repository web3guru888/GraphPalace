//! Safe Rust wrappers around the raw Kuzu FFI.
//!
//! Every struct owns its Kuzu handle and implements [`Drop`] for RAII cleanup.
//! All unsafe code in the crate is confined to this module.

use crate::ffi::*;
use gp_core::{GraphPalaceError, Result};
use std::collections::HashMap;
use std::ffi::{CStr, CString};

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// A Kuzu database instance. Owns the underlying `kuzu_database` handle.
pub struct Database {
    raw: kuzu_database,
}

// Safety: Kuzu documents that kuzu_database is thread-safe.
unsafe impl Send for Database {}
unsafe impl Sync for Database {}

impl Database {
    /// Open (or create) a Kuzu database at `path`.
    pub fn new(path: &str) -> Result<Self> {
        Self::open_with_config(path, kuzu_system_config::default())
    }

    /// Open (or create) a read-only Kuzu database at `path`.
    pub fn open_read_only(path: &str) -> Result<Self> {
        let mut cfg = kuzu_system_config::default();
        cfg.read_only = true;
        Self::open_with_config(path, cfg)
    }

    /// Open with explicit configuration.
    pub fn open_with_config(path: &str, config: kuzu_system_config) -> Result<Self> {
        let c_path = CString::new(path).map_err(|e| {
            GraphPalaceError::Storage(format!("invalid path: {e}"))
        })?;
        let mut raw = kuzu_database {
            _database: std::ptr::null_mut(),
        };
        let state = unsafe {
            kuzu_database_init(c_path.as_ptr(), config, &mut raw)
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(format!(
                "failed to open database at '{path}'"
            )));
        }
        Ok(Self { raw })
    }

    /// Get a mutable pointer to the inner handle (for connection init).
    pub(crate) fn raw_mut(&mut self) -> *mut kuzu_database {
        &mut self.raw
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        unsafe {
            kuzu_database_destroy(&mut self.raw);
        }
    }
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// A connection to a Kuzu database.
pub struct Connection {
    raw: kuzu_connection,
}

unsafe impl Send for Connection {}

impl Connection {
    /// Create a new connection to the given database.
    pub fn new(db: &mut Database) -> Result<Self> {
        let mut raw = kuzu_connection {
            _connection: std::ptr::null_mut(),
        };
        let state = unsafe {
            kuzu_connection_init(db.raw_mut(), &mut raw)
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(
                "failed to create connection".into(),
            ));
        }
        Ok(Self { raw })
    }

    /// Execute a Cypher query, returning a [`QueryResult`].
    pub fn query(&mut self, cypher: &str) -> Result<QueryResult> {
        let c_query = CString::new(cypher).map_err(|e| {
            GraphPalaceError::Storage(format!("invalid query string: {e}"))
        })?;
        let mut raw = kuzu_query_result {
            _query_result: std::ptr::null_mut(),
            _is_owned_by_cpp: false,
        };
        let state = unsafe {
            kuzu_connection_query(&mut self.raw, c_query.as_ptr(), &mut raw)
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(format!(
                "query execution failed: {cypher}"
            )));
        }
        let qr = QueryResult { raw };
        if !qr.is_success() {
            let msg = qr.error_message().unwrap_or_default();
            return Err(GraphPalaceError::Storage(format!(
                "query error: {msg}"
            )));
        }
        Ok(qr)
    }

    /// Prepare a parameterised Cypher query.
    pub fn prepare(&mut self, cypher: &str) -> Result<PreparedStatement> {
        let c_query = CString::new(cypher).map_err(|e| {
            GraphPalaceError::Storage(format!("invalid query string: {e}"))
        })?;
        let mut raw = kuzu_prepared_statement {
            _prepared_statement: std::ptr::null_mut(),
            _bound_values: std::ptr::null_mut(),
        };
        let state = unsafe {
            kuzu_connection_prepare(&mut self.raw, c_query.as_ptr(), &mut raw)
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(format!(
                "prepare failed: {cypher}"
            )));
        }
        Ok(PreparedStatement { raw })
    }

    /// Execute a prepared statement.
    pub fn execute(&mut self, stmt: &mut PreparedStatement) -> Result<QueryResult> {
        let mut raw = kuzu_query_result {
            _query_result: std::ptr::null_mut(),
            _is_owned_by_cpp: false,
        };
        let state = unsafe {
            kuzu_connection_execute(&mut self.raw, &mut stmt.raw, &mut raw)
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(
                "execute prepared statement failed".into(),
            ));
        }
        Ok(QueryResult { raw })
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        unsafe {
            kuzu_connection_destroy(&mut self.raw);
        }
    }
}

// ---------------------------------------------------------------------------
// PreparedStatement
// ---------------------------------------------------------------------------

/// A prepared (parameterised) Cypher statement.
pub struct PreparedStatement {
    raw: kuzu_prepared_statement,
}

impl PreparedStatement {
    /// Bind a string parameter.
    pub fn bind_string(&mut self, name: &str, value: &str) -> Result<()> {
        let c_name = CString::new(name).unwrap();
        let c_val = CString::new(value).unwrap();
        let state = unsafe {
            kuzu_prepared_statement_bind_string(
                &mut self.raw,
                c_name.as_ptr(),
                c_val.as_ptr(),
            )
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(format!(
                "bind_string failed for '{name}'"
            )));
        }
        Ok(())
    }

    /// Bind an i64 parameter.
    pub fn bind_i64(&mut self, name: &str, value: i64) -> Result<()> {
        let c_name = CString::new(name).unwrap();
        let state = unsafe {
            kuzu_prepared_statement_bind_int64(&mut self.raw, c_name.as_ptr(), value)
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(format!(
                "bind_i64 failed for '{name}'"
            )));
        }
        Ok(())
    }

    /// Bind an f64 parameter.
    pub fn bind_f64(&mut self, name: &str, value: f64) -> Result<()> {
        let c_name = CString::new(name).unwrap();
        let state = unsafe {
            kuzu_prepared_statement_bind_double(&mut self.raw, c_name.as_ptr(), value)
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(format!(
                "bind_f64 failed for '{name}'"
            )));
        }
        Ok(())
    }

    /// Bind a bool parameter.
    pub fn bind_bool(&mut self, name: &str, value: bool) -> Result<()> {
        let c_name = CString::new(name).unwrap();
        let state = unsafe {
            kuzu_prepared_statement_bind_bool(&mut self.raw, c_name.as_ptr(), value)
        };
        if state != kuzu_state::KuzuSuccess {
            return Err(GraphPalaceError::Storage(format!(
                "bind_bool failed for '{name}'"
            )));
        }
        Ok(())
    }
}

impl Drop for PreparedStatement {
    fn drop(&mut self) {
        unsafe {
            kuzu_prepared_statement_destroy(&mut self.raw);
        }
    }
}

// ---------------------------------------------------------------------------
// QueryResult
// ---------------------------------------------------------------------------

/// A query result with column metadata and row iteration.
pub struct QueryResult {
    raw: kuzu_query_result,
}

impl QueryResult {
    /// Whether the query succeeded.
    pub fn is_success(&self) -> bool {
        unsafe {
            kuzu_query_result_is_success(
                &self.raw as *const kuzu_query_result as *mut kuzu_query_result,
            )
        }
    }

    /// The error message (if the query failed).
    pub fn error_message(&self) -> Option<String> {
        unsafe {
            let ptr = kuzu_query_result_get_error_message(
                &self.raw as *const kuzu_query_result as *mut kuzu_query_result,
            );
            if ptr.is_null() {
                return None;
            }
            let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            kuzu_free_string(ptr);
            Some(s)
        }
    }

    /// Number of columns in the result.
    pub fn num_columns(&self) -> u64 {
        unsafe {
            kuzu_query_result_get_num_columns(
                &self.raw as *const kuzu_query_result as *mut kuzu_query_result,
            )
        }
    }

    /// Number of tuples in the result.
    pub fn num_tuples(&self) -> u64 {
        unsafe {
            kuzu_query_result_get_num_tuples(
                &self.raw as *const kuzu_query_result as *mut kuzu_query_result,
            )
        }
    }

    /// Retrieve column names.
    pub fn column_names(&self) -> Vec<String> {
        let n = self.num_columns();
        let mut names = Vec::with_capacity(n as usize);
        for i in 0..n {
            let mut ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let state = unsafe {
                kuzu_query_result_get_column_name(
                    &self.raw as *const kuzu_query_result as *mut kuzu_query_result,
                    i,
                    &mut ptr,
                )
            };
            if state == kuzu_state::KuzuSuccess && !ptr.is_null() {
                let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
                unsafe { kuzu_free_string(ptr) };
                names.push(s);
            } else {
                names.push(format!("col_{i}"));
            }
        }
        names
    }

    /// Get the next row, or `None` when exhausted.
    pub fn next(&mut self) -> Option<Row> {
        unsafe {
            if !kuzu_query_result_has_next(&mut self.raw) {
                return None;
            }
            let mut ft = kuzu_flat_tuple {
                _flat_tuple: std::ptr::null_mut(),
                _is_owned_by_cpp: false,
            };
            let state = kuzu_query_result_get_next(&mut self.raw, &mut ft);
            if state != kuzu_state::KuzuSuccess {
                return None;
            }
            Some(Row {
                tuple: ft,
                num_columns: self.num_columns(),
            })
        }
    }

    /// Collect all rows into a `Vec<HashMap<String, String>>` using
    /// `kuzu_value_to_string`.
    pub fn collect_all(&mut self) -> Vec<HashMap<String, String>> {
        let col_names = self.column_names();
        let mut rows = Vec::new();
        while let Some(row) = self.next() {
            let mut map = HashMap::new();
            for (i, name) in col_names.iter().enumerate() {
                if let Some(s) = row.get_string(i as u64) {
                    map.insert(name.clone(), s);
                }
            }
            rows.push(map);
        }
        rows
    }
}

impl Drop for QueryResult {
    fn drop(&mut self) {
        unsafe {
            kuzu_query_result_destroy(&mut self.raw);
        }
    }
}

// ---------------------------------------------------------------------------
// Row
// ---------------------------------------------------------------------------

/// A single row (flat tuple) from a query result.
pub struct Row {
    tuple: kuzu_flat_tuple,
    num_columns: u64,
}

impl Row {
    /// Get a value as a string (using `kuzu_value_to_string`).
    pub fn get_string(&self, idx: u64) -> Option<String> {
        if idx >= self.num_columns {
            return None;
        }
        unsafe {
            let mut val = kuzu_value {
                _value: std::ptr::null_mut(),
                _is_owned_by_cpp: false,
            };
            let state = kuzu_flat_tuple_get_value(
                &self.tuple as *const kuzu_flat_tuple as *mut kuzu_flat_tuple,
                idx,
                &mut val,
            );
            if state != kuzu_state::KuzuSuccess {
                return None;
            }
            if kuzu_value_is_null(&mut val) {
                return None;
            }
            let ptr = kuzu_value_to_string(&mut val);
            if ptr.is_null() {
                return None;
            }
            let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            kuzu_free_string(ptr);
            Some(s)
        }
    }

    /// Get a value as i64.
    pub fn get_i64(&self, idx: u64) -> Option<i64> {
        if idx >= self.num_columns {
            return None;
        }
        unsafe {
            let mut val = kuzu_value {
                _value: std::ptr::null_mut(),
                _is_owned_by_cpp: false,
            };
            let state = kuzu_flat_tuple_get_value(
                &self.tuple as *const kuzu_flat_tuple as *mut kuzu_flat_tuple,
                idx,
                &mut val,
            );
            if state != kuzu_state::KuzuSuccess {
                return None;
            }
            let mut result: i64 = 0;
            let state = kuzu_value_get_int64(&mut val, &mut result);
            if state == kuzu_state::KuzuSuccess {
                Some(result)
            } else {
                None
            }
        }
    }

    /// Get a value as f64.
    pub fn get_f64(&self, idx: u64) -> Option<f64> {
        if idx >= self.num_columns {
            return None;
        }
        unsafe {
            let mut val = kuzu_value {
                _value: std::ptr::null_mut(),
                _is_owned_by_cpp: false,
            };
            let state = kuzu_flat_tuple_get_value(
                &self.tuple as *const kuzu_flat_tuple as *mut kuzu_flat_tuple,
                idx,
                &mut val,
            );
            if state != kuzu_state::KuzuSuccess {
                return None;
            }
            let mut result: f64 = 0.0;
            let state = kuzu_value_get_double(&mut val, &mut result);
            if state == kuzu_state::KuzuSuccess {
                Some(result)
            } else {
                None
            }
        }
    }

    /// Get a value as bool.
    pub fn get_bool(&self, idx: u64) -> Option<bool> {
        if idx >= self.num_columns {
            return None;
        }
        unsafe {
            let mut val = kuzu_value {
                _value: std::ptr::null_mut(),
                _is_owned_by_cpp: false,
            };
            let state = kuzu_flat_tuple_get_value(
                &self.tuple as *const kuzu_flat_tuple as *mut kuzu_flat_tuple,
                idx,
                &mut val,
            );
            if state != kuzu_state::KuzuSuccess {
                return None;
            }
            let mut result: bool = false;
            let state = kuzu_value_get_bool(&mut val, &mut result);
            if state == kuzu_state::KuzuSuccess {
                Some(result)
            } else {
                None
            }
        }
    }
}
