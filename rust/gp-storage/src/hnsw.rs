//! HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index.
//!
//! Replaces the linear scan in `InMemoryBackend::search_drawers` with a
//! sub-linear time approximate search.  The index is not serialized — it is
//! rebuilt from drawer embeddings after deserialization / import.

use gp_core::Embedding;
use gp_embeddings::similarity::cosine_similarity;
use rand::Rng as _;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max bidirectional connections per node per layer.
    pub m: usize,
    /// Search width during construction.
    pub ef_construction: usize,
    /// Search width during query.
    pub ef_search: usize,
    /// Level generation multiplier: `1 / ln(m)`.
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (16.0f64).ln(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal node
// ---------------------------------------------------------------------------

/// An HNSW index entry.
#[derive(Clone)]
struct HnswNode {
    id: String,
    embedding: Embedding,
    /// `neighbors[level]` = set of neighbor IDs at that layer.
    neighbors: Vec<Vec<String>>,
    /// The highest layer this node lives on.
    level: usize,
}

// ---------------------------------------------------------------------------
// Heap helpers (we work in *similarity* space, higher = closer)
// ---------------------------------------------------------------------------

/// Element sorted by similarity — used as a *max-heap* entry (largest sim
/// first via the default `BinaryHeap`).
#[derive(Clone, PartialEq)]
struct MaxSim {
    sim: f32,
    id: String,
}

impl Eq for MaxSim {}

impl PartialOrd for MaxSim {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxSim {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sim
            .partial_cmp(&other.sim)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// HnswIndex
// ---------------------------------------------------------------------------

/// HNSW approximate nearest neighbor index.
#[derive(Clone)]
pub struct HnswIndex {
    config: HnswConfig,
    nodes: HashMap<String, HnswNode>,
    entry_point: Option<String>,
    max_level: usize,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("len", &self.nodes.len())
            .field("max_level", &self.max_level)
            .field("entry_point", &self.entry_point)
            .finish()
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new(HnswConfig::default())
    }
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            entry_point: None,
            max_level: 0,
        }
    }

    /// Number of points in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // -- Random level generation ---------------------------------------------

    /// Assign a random layer for a new node: `floor(-ln(uniform) * ml)`.
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rand::Rng::r#gen::<f64>(&mut rng); // (0, 1]
        // Clamp to avoid log(0).
        let r = r.max(1e-15);
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level
    }

    // -- Similarity helper ---------------------------------------------------

    fn similarity(&self, id: &str, query: &Embedding) -> f32 {
        match self.nodes.get(id) {
            Some(node) => cosine_similarity(query, &node.embedding),
            None => -1.0,
        }
    }

    // -- Core search_layer ---------------------------------------------------

    /// Search a single layer starting from `entry_ids`, returning up to `ef`
    /// nearest neighbors (by similarity, descending) at the given `level`.
    fn search_layer(
        &self,
        query: &Embedding,
        entry_ids: &[String],
        ef: usize,
        level: usize,
    ) -> Vec<(String, f32)> {
        let mut visited = HashSet::new();

        // candidates: max-heap — pop best (highest sim) candidate to expand.
        let mut candidates = BinaryHeap::<MaxSim>::new();

        // results: stored as a Vec; we track the worst sim separately.
        let mut results: Vec<(String, f32)> = Vec::new();

        for eid in entry_ids {
            if visited.insert(eid.clone()) {
                let sim = self.similarity(eid, query);
                candidates.push(MaxSim {
                    sim,
                    id: eid.clone(),
                });
                results.push((eid.clone(), sim));
            }
        }

        while let Some(best) = candidates.pop() {
            // Worst (lowest) similarity in the current result set.
            let worst_res_sim = results
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::INFINITY, f32::min);

            // If best remaining candidate is worse than the worst result and
            // we already have enough results, stop.
            if results.len() >= ef && best.sim < worst_res_sim {
                break;
            }

            // Expand neighbors of this candidate at the given level.
            if let Some(node) = self.nodes.get(&best.id) {
                if let Some(neighbors) = node.neighbors.get(level) {
                    for neighbor_id in neighbors {
                        if visited.insert(neighbor_id.clone()) {
                            let sim = self.similarity(neighbor_id, query);

                            let worst_res_sim = results
                                .iter()
                                .map(|(_, s)| *s)
                                .fold(f32::INFINITY, f32::min);

                            if results.len() < ef || sim > worst_res_sim {
                                candidates.push(MaxSim {
                                    sim,
                                    id: neighbor_id.clone(),
                                });
                                results.push((neighbor_id.clone(), sim));

                                // Evict worst if over capacity.
                                if results.len() > ef {
                                    if let Some(min_idx) = results
                                        .iter()
                                        .enumerate()
                                        .min_by(|a, b| {
                                            a.1 .1
                                                .partial_cmp(&b.1 .1)
                                                .unwrap_or(std::cmp::Ordering::Equal)
                                        })
                                        .map(|(i, _)| i)
                                    {
                                        results.swap_remove(min_idx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort descending by similarity.
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    // -- Greedy search (single best at a layer) ------------------------------

    /// Greedy walk at `level`, returning the single closest node to `query`.
    fn greedy_search(&self, query: &Embedding, entry_id: &str, level: usize) -> String {
        let mut current = entry_id.to_string();
        let mut current_sim = self.similarity(&current, query);

        loop {
            let mut changed = false;
            if let Some(node) = self.nodes.get(&current) {
                if let Some(neighbors) = node.neighbors.get(level) {
                    for nid in neighbors {
                        let sim = self.similarity(nid, query);
                        if sim > current_sim {
                            current_sim = sim;
                            current = nid.clone();
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    // -- Select neighbors (simple heuristic) ---------------------------------

    /// Select up to M best neighbors from candidates.
    fn select_neighbors(candidates: &[(String, f32)], m: usize) -> Vec<String> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(m);
        sorted.into_iter().map(|(id, _)| id).collect()
    }

    // -- Prune neighbors if over limit ---------------------------------------

    /// If a node has more than M neighbors at a given level, keep only the M
    /// closest to that node.
    fn prune_neighbors(&mut self, node_id: &str, level: usize) {
        let m = self.config.m;

        // Gather (neighbor_id, sim_to_node) pairs.
        let to_prune: Option<Vec<(String, f32)>> = {
            let node = self.nodes.get(node_id);
            node.and_then(|n| {
                n.neighbors.get(level).and_then(|nbrs| {
                    if nbrs.len() <= m {
                        None
                    } else {
                        Some(
                            nbrs.iter()
                                .map(|nid| {
                                    let sim = match (self.nodes.get(node_id), self.nodes.get(nid)) {
                                        (Some(a), Some(b)) => {
                                            cosine_similarity(&a.embedding, &b.embedding)
                                        }
                                        _ => -1.0,
                                    };
                                    (nid.clone(), sim)
                                })
                                .collect::<Vec<_>>(),
                        )
                    }
                })
            })
        };

        if let Some(candidates) = to_prune {
            let kept = Self::select_neighbors(&candidates, m);
            if let Some(node) = self.nodes.get_mut(node_id) {
                if let Some(nbrs) = node.neighbors.get_mut(level) {
                    *nbrs = kept;
                }
            }
        }
    }

    // -- Insert --------------------------------------------------------------

    /// Insert a new point into the index.
    pub fn insert(&mut self, id: &str, embedding: &Embedding) {
        // If the id already exists, remove it first to avoid duplicates.
        if self.nodes.contains_key(id) {
            self.remove(id);
        }

        let new_level = self.random_level();

        // Create the node.
        let new_node = HnswNode {
            id: id.to_string(),
            embedding: *embedding,
            neighbors: vec![Vec::new(); new_level + 1],
            level: new_level,
        };
        self.nodes.insert(id.to_string(), new_node);

        // First insertion — set as entry point and return.
        let entry_id = match &self.entry_point {
            Some(ep) => ep.clone(),
            None => {
                self.entry_point = Some(id.to_string());
                self.max_level = new_level;
                return;
            }
        };

        let mut current = entry_id;

        // Phase 1: greedy search from max_level down to new_level + 1.
        if self.max_level > new_level {
            for level in (new_level + 1..=self.max_level).rev() {
                current = self.greedy_search(embedding, &current, level);
            }
        }

        // Phase 2: search and connect at each level from min(new_level, max_level) down to 0.
        let connect_from = new_level.min(self.max_level);
        for level in (0..=connect_from).rev() {
            let neighbors =
                self.search_layer(embedding, &[current.clone()], self.config.ef_construction, level);

            let selected = Self::select_neighbors(&neighbors, self.config.m);

            // Connect new node → selected neighbors.
            if let Some(node) = self.nodes.get_mut(id) {
                if let Some(nbrs) = node.neighbors.get_mut(level) {
                    *nbrs = selected.clone();
                }
            }

            // Connect selected neighbors → new node (bidirectional).
            for neighbor_id in &selected {
                // Add new node to neighbor's neighbor list.
                if let Some(neighbor_node) = self.nodes.get_mut(neighbor_id) {
                    // Ensure the neighbor has enough levels.
                    while neighbor_node.neighbors.len() <= level {
                        neighbor_node.neighbors.push(Vec::new());
                    }
                    if let Some(nbrs) = neighbor_node.neighbors.get_mut(level) {
                        if !nbrs.contains(&id.to_string()) {
                            nbrs.push(id.to_string());
                        }
                    }
                }
                // Prune if over capacity.
                self.prune_neighbors(neighbor_id, level);
            }

            // Use the closest neighbor as entry for the next level down.
            if let Some(first) = neighbors.first() {
                current = first.0.clone();
            }
        }

        // Update entry point if new node is at a higher level.
        if new_level > self.max_level {
            self.max_level = new_level;
            self.entry_point = Some(id.to_string());
        }
    }

    // -- Remove --------------------------------------------------------------

    /// Remove a point from the index.
    ///
    /// This performs a "lazy" removal: the node is deleted and all references
    /// to it in its neighbors' adjacency lists are cleaned up.  No reconnection
    /// of orphaned neighbors is attempted (the index degrades gracefully and
    /// can be rebuilt if quality drops).
    pub fn remove(&mut self, id: &str) {
        let node = match self.nodes.remove(id) {
            Some(n) => n,
            None => return,
        };

        // Remove references to this node from all its neighbors.
        for (level, neighbors) in node.neighbors.iter().enumerate() {
            for neighbor_id in neighbors {
                if let Some(neighbor_node) = self.nodes.get_mut(neighbor_id) {
                    if let Some(nbrs) = neighbor_node.neighbors.get_mut(level) {
                        nbrs.retain(|n| n != id);
                    }
                }
            }
        }

        // If the removed node was the entry point, pick a replacement.
        if self.entry_point.as_deref() == Some(id) {
            if self.nodes.is_empty() {
                self.entry_point = None;
                self.max_level = 0;
            } else {
                // Pick the node with the highest level as the new entry point.
                let (best_id, best_level) = self
                    .nodes
                    .values()
                    .map(|n| (n.id.clone(), n.level))
                    .max_by_key(|(_, l)| *l)
                    .unwrap();
                self.entry_point = Some(best_id);
                self.max_level = best_level;
            }
        }
    }

    // -- Search --------------------------------------------------------------

    /// Find the k nearest neighbors of the query embedding.
    ///
    /// Returns `(id, similarity)` pairs sorted by descending similarity.
    pub fn search(&self, query: &Embedding, k: usize) -> Vec<(String, f32)> {
        let entry_id = match &self.entry_point {
            Some(ep) => ep.clone(),
            None => return Vec::new(),
        };

        let mut current = entry_id;

        // Phase 1: greedy descent from max_level down to level 1.
        if self.max_level > 0 {
            for level in (1..=self.max_level).rev() {
                current = self.greedy_search(query, &current, level);
            }
        }

        // Phase 2: search at level 0 with ef_search candidates.
        let ef = self.config.ef_search.max(k);
        let mut results = self.search_layer(query, &[current], ef, 0);

        // Return top k.
        results.truncate(k);
        results
    }

    // -- Rebuild -------------------------------------------------------------

    /// Rebuild the entire index from scratch.
    ///
    /// Call this after deserialization / import: collect all `(id, embedding)`
    /// pairs, clear the index, and re-insert them.
    pub fn rebuild(&mut self) {
        let items: Vec<(String, Embedding)> = self
            .nodes
            .drain()
            .map(|(id, node)| (id, node.embedding))
            .collect();

        self.entry_point = None;
        self.max_level = 0;

        for (id, emb) in items {
            self.insert(&id, &emb);
        }
    }

    /// Build the index from an iterator of `(id, embedding)` pairs.
    ///
    /// This is the preferred way to populate the index from existing data
    /// (e.g. after loading drawers from disk).
    pub fn build_from<'a>(&mut self, items: impl IntoIterator<Item = (&'a str, &'a Embedding)>) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_level = 0;

        for (id, emb) in items {
            self.insert(id, emb);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a normalized embedding seeded by `seed` (384-dim).
    fn make_emb(seed: u8) -> Embedding {
        let mut emb = [0.0f32; 384];
        for (i, v) in emb.iter_mut().enumerate() {
            *v = ((seed as f32 + i as f32) * 0.01).sin();
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in emb.iter_mut() {
                *v /= norm;
            }
        }
        emb
    }

    #[test]
    fn test_empty_index() {
        let idx = HnswIndex::default();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.search(&make_emb(0), 5).is_empty());
    }

    #[test]
    fn test_insert_single() {
        let mut idx = HnswIndex::default();
        idx.insert("a", &make_emb(1));
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());

        let results = idx.search(&make_emb(1), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 > 0.99, "self-similarity should be ~1.0");
    }

    #[test]
    fn test_insert_and_search_multiple() {
        let mut idx = HnswIndex::new(HnswConfig::default());

        // Insert 20 embeddings.
        for i in 0..20u8 {
            idx.insert(&format!("node_{i}"), &make_emb(i));
        }
        assert_eq!(idx.len(), 20);

        // Search for the exact embedding of node_5 — it should be the top result.
        let results = idx.search(&make_emb(5), 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "node_5");
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_search_returns_sorted_descending() {
        let mut idx = HnswIndex::new(HnswConfig::default());
        for i in 0..30u8 {
            idx.insert(&format!("n{i}"), &make_emb(i));
        }
        let results = idx.search(&make_emb(10), 10);
        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "results should be sorted descending by similarity: {} >= {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn test_remove() {
        let mut idx = HnswIndex::new(HnswConfig::default());
        idx.insert("a", &make_emb(1));
        idx.insert("b", &make_emb(2));
        idx.insert("c", &make_emb(3));
        assert_eq!(idx.len(), 3);

        idx.remove("b");
        assert_eq!(idx.len(), 2);

        // "b" should no longer appear in search results.
        let results = idx.search(&make_emb(2), 10);
        assert!(
            results.iter().all(|(id, _)| id != "b"),
            "removed node should not appear in results"
        );
    }

    #[test]
    fn test_remove_entry_point() {
        let mut idx = HnswIndex::new(HnswConfig::default());
        idx.insert("a", &make_emb(1));
        idx.insert("b", &make_emb(2));

        // Remove whatever the current entry point is.
        let ep = idx.entry_point.clone().unwrap();
        idx.remove(&ep);
        assert_eq!(idx.len(), 1);

        // Index should still be searchable.
        let results = idx.search(&make_emb(1), 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_remove_all() {
        let mut idx = HnswIndex::new(HnswConfig::default());
        idx.insert("a", &make_emb(1));
        idx.insert("b", &make_emb(2));
        idx.remove("a");
        idx.remove("b");
        assert!(idx.is_empty());
        assert!(idx.search(&make_emb(1), 5).is_empty());
    }

    #[test]
    fn test_rebuild() {
        let mut idx = HnswIndex::new(HnswConfig::default());
        for i in 0..15u8 {
            idx.insert(&format!("n{i}"), &make_emb(i));
        }

        idx.rebuild();
        assert_eq!(idx.len(), 15);

        // Should still find the right node.
        let results = idx.search(&make_emb(7), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "n7");
    }

    #[test]
    fn test_build_from() {
        let mut idx = HnswIndex::new(HnswConfig::default());
        let embeddings: Vec<(String, Embedding)> = (0..10u8)
            .map(|i| (format!("item_{i}"), make_emb(i)))
            .collect();

        let items: Vec<(&str, &Embedding)> = embeddings
            .iter()
            .map(|(id, emb)| (id.as_str(), emb))
            .collect();
        idx.build_from(items);

        assert_eq!(idx.len(), 10);

        let results = idx.search(&make_emb(3), 1);
        assert_eq!(results[0].0, "item_3");
    }

    #[test]
    fn test_duplicate_insert_replaces() {
        let mut idx = HnswIndex::new(HnswConfig::default());
        idx.insert("a", &make_emb(1));
        idx.insert("a", &make_emb(2)); // replace
        assert_eq!(idx.len(), 1);

        // Should find the new embedding.
        let results = idx.search(&make_emb(2), 1);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_larger_index_recall() {
        // Insert 100 points and verify that exact-match queries have high recall.
        let mut idx = HnswIndex::new(HnswConfig {
            ef_search: 100,
            ..HnswConfig::default()
        });

        let mut embeddings = Vec::new();
        for i in 0..100u8 {
            let emb = make_emb(i);
            embeddings.push((format!("p{i}"), emb));
            idx.insert(&format!("p{i}"), &emb);
        }

        // For each point, the top-1 result should be itself.
        let mut exact_hits = 0;
        for (id, emb) in &embeddings {
            let results = idx.search(emb, 1);
            if !results.is_empty() && results[0].0 == *id {
                exact_hits += 1;
            }
        }

        assert!(
            exact_hits >= 95,
            "expected at least 95/100 exact self-matches, got {exact_hits}"
        );
    }
}
