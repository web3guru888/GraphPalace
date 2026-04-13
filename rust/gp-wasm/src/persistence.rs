//! IndexedDB/OPFS persistence layer types and traits.
//!
//! Defines the abstraction for persisting palace state in browser storage.
//! Actual JavaScript interop (IndexedDB, OPFS) is stubbed since we can't
//! run WASM-specific code in native CI tests.

use serde::{Deserialize, Serialize};

/// Storage backend identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackend {
    /// In-memory storage (no persistence, for testing).
    Memory,
    /// Browser IndexedDB.
    IndexedDB,
    /// Origin Private File System (preferred for browsers).
    Opfs,
    /// Native file system (for server/CLI use).
    FileSystem,
}

impl std::fmt::Display for StorageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory => write!(f, "memory"),
            Self::IndexedDB => write!(f, "indexeddb"),
            Self::Opfs => write!(f, "opfs"),
            Self::FileSystem => write!(f, "filesystem"),
        }
    }
}

/// Import mode for loading palace data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportMode {
    /// Drop existing data, load new.
    Replace,
    /// Add new items, skip duplicates.
    Merge,
    /// Add new items, update existing.
    Overlay,
}

/// Metadata about a persisted palace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceMetadata {
    /// Palace name.
    pub name: String,
    /// Storage backend used.
    pub backend: StorageBackend,
    /// Total size in bytes (approximate).
    pub size_bytes: u64,
    /// Number of nodes.
    pub node_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Last saved timestamp (Unix millis).
    pub last_saved_ms: u64,
    /// Schema version for migration.
    pub schema_version: u32,
}

impl PalaceMetadata {
    /// Current schema version.
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    /// Create metadata for a new palace.
    pub fn new(name: &str, backend: StorageBackend) -> Self {
        Self {
            name: name.to_string(),
            backend,
            size_bytes: 0,
            node_count: 0,
            edge_count: 0,
            last_saved_ms: 0,
            schema_version: Self::CURRENT_SCHEMA_VERSION,
        }
    }
}

/// Trait for palace persistence backends.
///
/// Implementations handle serialization and storage of the palace
/// state to different browser/native storage mechanisms.
pub trait PalacePersistence {
    /// Save the palace data.
    fn save(&mut self, data: &str) -> Result<PalaceMetadata, PersistenceError>;

    /// Load the palace data. Returns None if no saved state exists.
    fn load(&self) -> Result<Option<String>, PersistenceError>;

    /// Delete saved palace data.
    fn delete(&mut self) -> Result<(), PersistenceError>;

    /// Check if a saved palace exists.
    fn exists(&self) -> bool;

    /// Get metadata about the saved palace.
    fn metadata(&self) -> Result<Option<PalaceMetadata>, PersistenceError>;
}

/// Persistence errors.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum PersistenceError {
    /// Storage is not available.
    #[error("Storage not available: {0}")]
    NotAvailable(String),
    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),
    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(String),
    /// Schema version mismatch.
    #[error("Schema version mismatch: expected {expected}, found {found}")]
    SchemaMismatch { expected: u32, found: u32 },
    /// Storage quota exceeded.
    #[error("Storage quota exceeded")]
    QuotaExceeded,
}

/// In-memory persistence backend (for testing).
#[derive(Debug, Clone, Default)]
pub struct MemoryPersistence {
    data: Option<String>,
    metadata: Option<PalaceMetadata>,
}

impl MemoryPersistence {
    /// Create a new memory persistence backend.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PalacePersistence for MemoryPersistence {
    fn save(&mut self, data: &str) -> Result<PalaceMetadata, PersistenceError> {
        let meta = PalaceMetadata {
            name: "memory".to_string(),
            backend: StorageBackend::Memory,
            size_bytes: data.len() as u64,
            node_count: 0,
            edge_count: 0,
            last_saved_ms: 0, // Would use js_sys::Date::now() in WASM
            schema_version: PalaceMetadata::CURRENT_SCHEMA_VERSION,
        };
        self.data = Some(data.to_string());
        self.metadata = Some(meta.clone());
        Ok(meta)
    }

    fn load(&self) -> Result<Option<String>, PersistenceError> {
        Ok(self.data.clone())
    }

    fn delete(&mut self) -> Result<(), PersistenceError> {
        self.data = None;
        self.metadata = None;
        Ok(())
    }

    fn exists(&self) -> bool {
        self.data.is_some()
    }

    fn metadata(&self) -> Result<Option<PalaceMetadata>, PersistenceError> {
        Ok(self.metadata.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_backend_display() {
        assert_eq!(StorageBackend::Memory.to_string(), "memory");
        assert_eq!(StorageBackend::IndexedDB.to_string(), "indexeddb");
        assert_eq!(StorageBackend::Opfs.to_string(), "opfs");
        assert_eq!(StorageBackend::FileSystem.to_string(), "filesystem");
    }

    #[test]
    fn palace_metadata_new() {
        let meta = PalaceMetadata::new("Test", StorageBackend::Memory);
        assert_eq!(meta.name, "Test");
        assert_eq!(meta.schema_version, 1);
        assert_eq!(meta.size_bytes, 0);
    }

    #[test]
    fn memory_persistence_save_load() {
        let mut store = MemoryPersistence::new();
        assert!(!store.exists());

        let meta = store.save("test data").unwrap();
        assert!(store.exists());
        assert_eq!(meta.size_bytes, 9); // "test data" = 9 bytes

        let data = store.load().unwrap();
        assert_eq!(data, Some("test data".to_string()));
    }

    #[test]
    fn memory_persistence_delete() {
        let mut store = MemoryPersistence::new();
        store.save("data").unwrap();
        assert!(store.exists());

        store.delete().unwrap();
        assert!(!store.exists());
        assert_eq!(store.load().unwrap(), None);
    }

    #[test]
    fn memory_persistence_metadata() {
        let mut store = MemoryPersistence::new();
        assert!(store.metadata().unwrap().is_none());

        store.save("some data").unwrap();
        let meta = store.metadata().unwrap().unwrap();
        assert_eq!(meta.backend, StorageBackend::Memory);
        assert_eq!(meta.size_bytes, 9);
    }

    #[test]
    fn memory_persistence_overwrite() {
        let mut store = MemoryPersistence::new();
        store.save("first").unwrap();
        store.save("second").unwrap();
        assert_eq!(store.load().unwrap(), Some("second".to_string()));
    }

    #[test]
    fn import_mode_variants() {
        let modes = vec![ImportMode::Replace, ImportMode::Merge, ImportMode::Overlay];
        for mode in &modes {
            let json = serde_json::to_string(mode).unwrap();
            let m2: ImportMode = serde_json::from_str(&json).unwrap();
            assert_eq!(&m2, mode);
        }
    }

    #[test]
    fn persistence_error_display() {
        let e = PersistenceError::NotAvailable("IndexedDB".into());
        assert!(e.to_string().contains("IndexedDB"));

        let e = PersistenceError::SchemaMismatch {
            expected: 2,
            found: 1,
        };
        assert!(e.to_string().contains("expected 2"));
    }

    #[test]
    fn persistence_error_serialization() {
        let e = PersistenceError::QuotaExceeded;
        let json = serde_json::to_string(&e).unwrap();
        let e2: PersistenceError = serde_json::from_str(&json).unwrap();
        assert!(matches!(e2, PersistenceError::QuotaExceeded));
    }

    #[test]
    fn metadata_serialization_roundtrip() {
        let meta = PalaceMetadata {
            name: "Test Palace".into(),
            backend: StorageBackend::Opfs,
            size_bytes: 1024,
            node_count: 100,
            edge_count: 150,
            last_saved_ms: 1700000000000,
            schema_version: 1,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let meta2: PalaceMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(meta2.name, "Test Palace");
        assert_eq!(meta2.backend, StorageBackend::Opfs);
        assert_eq!(meta2.node_count, 100);
    }

    #[test]
    fn memory_persistence_empty_data() {
        let mut store = MemoryPersistence::new();
        let meta = store.save("").unwrap();
        assert_eq!(meta.size_bytes, 0);
        assert!(store.exists());
        assert_eq!(store.load().unwrap(), Some(String::new()));
    }

    #[test]
    fn memory_persistence_large_data() {
        let mut store = MemoryPersistence::new();
        let big_data = "x".repeat(1_000_000);
        let meta = store.save(&big_data).unwrap();
        assert_eq!(meta.size_bytes, 1_000_000);
        assert_eq!(store.load().unwrap().unwrap().len(), 1_000_000);
    }

    #[test]
    fn storage_backend_equality() {
        assert_eq!(StorageBackend::Memory, StorageBackend::Memory);
        assert_ne!(StorageBackend::Memory, StorageBackend::IndexedDB);
        assert_ne!(StorageBackend::Opfs, StorageBackend::FileSystem);
    }

    #[test]
    fn persistence_error_io() {
        let e = PersistenceError::IoError("disk full".into());
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn persistence_error_serialization_msg() {
        let e = PersistenceError::SerializationError("invalid utf8".into());
        assert!(e.to_string().contains("invalid utf8"));
    }

    #[test]
    fn delete_nonexistent_is_ok() {
        let mut store = MemoryPersistence::new();
        assert!(!store.exists());
        store.delete().unwrap(); // Should not fail
        assert!(!store.exists());
    }
}
