//! A simple in-memory embedding cache.

use gp_core::{Embedding, EMBEDDING_DIM};
use std::collections::HashMap;

/// A `HashMap`-based cache that stores embeddings keyed by their source text.
///
/// This avoids recomputing (or re-fetching) embeddings for text that has
/// already been encoded.
pub struct EmbeddingCache {
    inner: HashMap<String, Embedding>,
}

impl EmbeddingCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Create a new cache pre-allocated for `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
        }
    }

    /// Look up a cached embedding by text.
    pub fn get(&self, text: &str) -> Option<&Embedding> {
        self.inner.get(text)
    }

    /// Insert (or overwrite) an embedding for the given text.
    pub fn insert(&mut self, text: &str, embedding: Embedding) {
        self.inner.insert(text.to_owned(), embedding);
    }

    /// Returns the number of cached embeddings.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Remove an entry from the cache. Returns `true` if the entry existed.
    pub fn remove(&mut self, text: &str) -> bool {
        self.inner.remove(text).is_some()
    }

    /// Clear all cached embeddings.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns an iterator over all cached `(text, embedding)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Embedding)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Return the dimension of cached embeddings (always `EMBEDDING_DIM`).
    pub fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::EMBEDDING_DIM;

    fn make_embedding(seed: f32) -> Embedding {
        let mut e = [0.0f32; EMBEDDING_DIM];
        for (i, v) in e.iter_mut().enumerate() {
            *v = seed + i as f32 * 0.001;
        }
        e
    }

    #[test]
    fn cache_miss_returns_none() {
        let cache = EmbeddingCache::new();
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn cache_insert_and_get() {
        let mut cache = EmbeddingCache::new();
        let emb = make_embedding(1.0);
        cache.insert("hello", emb);
        let retrieved = cache.get("hello").unwrap();
        assert_eq!(*retrieved, emb);
    }

    #[test]
    fn cache_overwrite() {
        let mut cache = EmbeddingCache::new();
        let emb1 = make_embedding(1.0);
        let emb2 = make_embedding(2.0);
        cache.insert("key", emb1);
        cache.insert("key", emb2);
        assert_eq!(cache.len(), 1);
        assert_eq!(*cache.get("key").unwrap(), emb2);
    }

    #[test]
    fn cache_len_and_is_empty() {
        let mut cache = EmbeddingCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.insert("a", make_embedding(0.0));
        cache.insert("b", make_embedding(1.0));
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_remove() {
        let mut cache = EmbeddingCache::new();
        cache.insert("x", make_embedding(0.5));
        assert!(cache.remove("x"));
        assert!(!cache.remove("x")); // second remove returns false
        assert!(cache.get("x").is_none());
    }

    #[test]
    fn cache_clear() {
        let mut cache = EmbeddingCache::new();
        cache.insert("a", make_embedding(0.0));
        cache.insert("b", make_embedding(1.0));
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_dimension() {
        let cache = EmbeddingCache::new();
        assert_eq!(cache.dimension(), 384);
    }
}
