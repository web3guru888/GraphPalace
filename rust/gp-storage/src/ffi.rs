//! Raw FFI bindings to Kuzu's C API.
//!
//! This module is gated behind `#[cfg(feature = "kuzu-ffi")]` so the crate
//! compiles without the actual Kuzu shared library.
//!
//! All types here are opaque pointers matching `kuzu.h` from the Kuzu C API.

#![allow(non_camel_case_types, dead_code)]

use libc::{c_char, c_void};

// ---------------------------------------------------------------------------
// Opaque types (match kuzu.h structs)
// ---------------------------------------------------------------------------

/// Runtime configuration for opening a Kuzu database.
#[repr(C)]
pub struct kuzu_system_config {
    pub buffer_pool_size: u64,
    pub max_num_threads: u64,
    pub enable_compression: bool,
    pub read_only: bool,
    pub max_db_size: u64,
    pub auto_checkpoint: bool,
    pub checkpoint_threshold: u64,
}

impl Default for kuzu_system_config {
    fn default() -> Self {
        Self {
            buffer_pool_size: 0, // 0 = Kuzu default
            max_num_threads: 0,  // 0 = Kuzu default
            enable_compression: true,
            read_only: false,
            max_db_size: 1 << 43, // 8 TB
            auto_checkpoint: true,
            checkpoint_threshold: 1 << 28, // 256 MB
        }
    }
}

/// Manages all database components.
#[repr(C)]
pub struct kuzu_database {
    pub _database: *mut c_void,
}

/// A connection to a Kuzu database.
#[repr(C)]
pub struct kuzu_connection {
    pub _connection: *mut c_void,
}

/// A parameterised query that can be re-executed.
#[repr(C)]
pub struct kuzu_prepared_statement {
    pub _prepared_statement: *mut c_void,
    pub _bound_values: *mut c_void,
}

/// Stores the result of a query.
#[repr(C)]
pub struct kuzu_query_result {
    pub _query_result: *mut c_void,
    pub _is_owned_by_cpp: bool,
}

/// Stores a vector of values (one row).
#[repr(C)]
pub struct kuzu_flat_tuple {
    pub _flat_tuple: *mut c_void,
    pub _is_owned_by_cpp: bool,
}

/// Kuzu internal representation of data types.
#[repr(C)]
pub struct kuzu_logical_type {
    pub _data_type: *mut c_void,
}

/// A single Kuzu value with any internal data type.
#[repr(C)]
pub struct kuzu_value {
    pub _value: *mut c_void,
    pub _is_owned_by_cpp: bool,
}

/// Return state for Kuzu C API functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum kuzu_state {
    KuzuSuccess = 0,
    KuzuError = 1,
}

// ---------------------------------------------------------------------------
// External function declarations
// ---------------------------------------------------------------------------

extern "C" {
    // -- Database lifecycle ---------------------------------------------------

    /// Create/open a Kuzu database at the given path.
    pub fn kuzu_database_init(
        database_path: *const c_char,
        system_config: kuzu_system_config,
        out_database: *mut kuzu_database,
    ) -> kuzu_state;

    /// Destroy a database handle and free associated memory.
    pub fn kuzu_database_destroy(database: *mut kuzu_database);

    // -- Connection lifecycle -------------------------------------------------

    /// Create a connection to an open database.
    pub fn kuzu_connection_init(
        database: *mut kuzu_database,
        out_connection: *mut kuzu_connection,
    ) -> kuzu_state;

    /// Destroy a connection handle.
    pub fn kuzu_connection_destroy(connection: *mut kuzu_connection);

    // -- Query execution ------------------------------------------------------

    /// Execute a Cypher query string.
    pub fn kuzu_connection_query(
        connection: *mut kuzu_connection,
        query: *const c_char,
        out_query_result: *mut kuzu_query_result,
    ) -> kuzu_state;

    /// Prepare a parameterised Cypher query.
    pub fn kuzu_connection_prepare(
        connection: *mut kuzu_connection,
        query: *const c_char,
        out_prepared_statement: *mut kuzu_prepared_statement,
    ) -> kuzu_state;

    /// Execute a prepared statement.
    pub fn kuzu_connection_execute(
        connection: *mut kuzu_connection,
        prepared_statement: *mut kuzu_prepared_statement,
        out_query_result: *mut kuzu_query_result,
    ) -> kuzu_state;

    // -- Prepared statement binding -------------------------------------------

    pub fn kuzu_prepared_statement_destroy(
        prepared_statement: *mut kuzu_prepared_statement,
    );

    pub fn kuzu_prepared_statement_is_success(
        prepared_statement: *mut kuzu_prepared_statement,
    ) -> bool;

    pub fn kuzu_prepared_statement_get_error_message(
        prepared_statement: *mut kuzu_prepared_statement,
    ) -> *mut c_char;

    pub fn kuzu_prepared_statement_bind_string(
        prepared_statement: *mut kuzu_prepared_statement,
        param_name: *const c_char,
        value: *const c_char,
    ) -> kuzu_state;

    pub fn kuzu_prepared_statement_bind_int64(
        prepared_statement: *mut kuzu_prepared_statement,
        param_name: *const c_char,
        value: i64,
    ) -> kuzu_state;

    pub fn kuzu_prepared_statement_bind_double(
        prepared_statement: *mut kuzu_prepared_statement,
        param_name: *const c_char,
        value: f64,
    ) -> kuzu_state;

    pub fn kuzu_prepared_statement_bind_bool(
        prepared_statement: *mut kuzu_prepared_statement,
        param_name: *const c_char,
        value: bool,
    ) -> kuzu_state;

    pub fn kuzu_prepared_statement_bind_float(
        prepared_statement: *mut kuzu_prepared_statement,
        param_name: *const c_char,
        value: f32,
    ) -> kuzu_state;

    // -- Query result ---------------------------------------------------------

    pub fn kuzu_query_result_destroy(query_result: *mut kuzu_query_result);

    pub fn kuzu_query_result_is_success(
        query_result: *mut kuzu_query_result,
    ) -> bool;

    pub fn kuzu_query_result_get_error_message(
        query_result: *mut kuzu_query_result,
    ) -> *mut c_char;

    pub fn kuzu_query_result_get_num_columns(
        query_result: *mut kuzu_query_result,
    ) -> u64;

    pub fn kuzu_query_result_get_column_name(
        query_result: *mut kuzu_query_result,
        index: u64,
        out_column_name: *mut *mut c_char,
    ) -> kuzu_state;

    pub fn kuzu_query_result_get_num_tuples(
        query_result: *mut kuzu_query_result,
    ) -> u64;

    pub fn kuzu_query_result_has_next(
        query_result: *mut kuzu_query_result,
    ) -> bool;

    pub fn kuzu_query_result_get_next(
        query_result: *mut kuzu_query_result,
        out_flat_tuple: *mut kuzu_flat_tuple,
    ) -> kuzu_state;

    pub fn kuzu_query_result_reset_iterator(
        query_result: *mut kuzu_query_result,
    );

    pub fn kuzu_query_result_to_string(
        query_result: *mut kuzu_query_result,
    ) -> *mut c_char;

    // -- Flat tuple -----------------------------------------------------------

    pub fn kuzu_flat_tuple_destroy(flat_tuple: *mut kuzu_flat_tuple);

    pub fn kuzu_flat_tuple_get_value(
        flat_tuple: *mut kuzu_flat_tuple,
        index: u64,
        out_value: *mut kuzu_value,
    ) -> kuzu_state;

    pub fn kuzu_flat_tuple_to_string(
        flat_tuple: *mut kuzu_flat_tuple,
    ) -> *mut c_char;

    // -- Value creation -------------------------------------------------------

    pub fn kuzu_value_create_string(val: *const c_char) -> *mut kuzu_value;
    pub fn kuzu_value_create_int64(val: i64) -> *mut kuzu_value;
    pub fn kuzu_value_create_double(val: f64) -> *mut kuzu_value;
    pub fn kuzu_value_create_float(val: f32) -> *mut kuzu_value;
    pub fn kuzu_value_create_bool(val: bool) -> *mut kuzu_value;
    pub fn kuzu_value_create_null() -> *mut kuzu_value;
    pub fn kuzu_value_clone(value: *mut kuzu_value) -> *mut kuzu_value;

    // -- Value access ---------------------------------------------------------

    pub fn kuzu_value_destroy(value: *mut kuzu_value);
    pub fn kuzu_value_is_null(value: *mut kuzu_value) -> bool;

    pub fn kuzu_value_get_bool(
        value: *mut kuzu_value,
        out_result: *mut bool,
    ) -> kuzu_state;

    pub fn kuzu_value_get_int64(
        value: *mut kuzu_value,
        out_result: *mut i64,
    ) -> kuzu_state;

    pub fn kuzu_value_get_float(
        value: *mut kuzu_value,
        out_result: *mut f32,
    ) -> kuzu_state;

    pub fn kuzu_value_get_double(
        value: *mut kuzu_value,
        out_result: *mut f64,
    ) -> kuzu_state;

    pub fn kuzu_value_get_string(
        value: *mut kuzu_value,
        out_result: *mut *mut c_char,
    ) -> kuzu_state;

    pub fn kuzu_value_to_string(value: *mut kuzu_value) -> *mut c_char;

    // -- Logical type ---------------------------------------------------------

    pub fn kuzu_value_get_data_type(
        value: *mut kuzu_value,
        out_type: *mut kuzu_logical_type,
    );

    pub fn kuzu_data_type_destroy(data_type: *mut kuzu_logical_type);
}

// ---------------------------------------------------------------------------
// Helper: free a C string returned by Kuzu
// ---------------------------------------------------------------------------

/// Free a C string allocated by Kuzu. The Kuzu C API documents that callers
/// must free returned `char*` with the standard C `free()`.
///
/// # Safety
/// The pointer must have been allocated by the Kuzu C library.
pub unsafe fn kuzu_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        libc::free(ptr as *mut c_void);
    }
}
