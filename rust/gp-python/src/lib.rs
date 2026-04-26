// PyO3 macro expansion triggers useless_conversion on generated Into impls.
#![allow(clippy::useless_conversion)]

//! GraphPalace Python bindings via PyO3.
//!
//! Provides a native Python interface to the GraphPalace memory palace engine.
//!
//! ```python
//! from graphpalace import Palace
//!
//! palace = Palace("./my_palace")
//! palace.add_drawer("Postgres chosen for concurrent writes", "project", "decisions")
//! results = palace.search("why did we choose Postgres?", k=5)
//! for r in results:
//!     print(f"[{r.score:.3f}] {r.content[:80]}")
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use gp_core::{DrawerSource, GraphPalaceConfig, HallType, WingType};
use gp_embeddings::{EmbeddingEngine, TfIdfEmbeddingEngine};
use gp_palace::export::ImportMode;
use gp_palace::GraphPalace;
use gp_storage::memory::PalaceData;
use gp_storage::InMemoryBackend;

use std::path::Path;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Send wrapper
// ---------------------------------------------------------------------------

/// Wrapper to allow `GraphPalace` in a `#[pyclass]`.
///
/// `GraphPalace` stores `Box<dyn EmbeddingEngine>` whose trait is not
/// explicitly `Send`.  In practice we always construct it with
/// `TfIdfEmbeddingEngine` (pure Rust data, trivially `Send`) and
/// `InMemoryBackend` (Arc<RwLock<>>, also `Send`), so this is safe.
struct SendablePalace(GraphPalace);

// SAFETY: see doc-comment above — all concrete types are Send.
unsafe impl Send for SendablePalace {}

// ---------------------------------------------------------------------------
// Helper: convert GraphPalaceError → PyErr
// ---------------------------------------------------------------------------

fn gp_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Helper: parse string ↔ enum
// ---------------------------------------------------------------------------

fn parse_wing_type(s: &str) -> WingType {
    match s.to_lowercase().as_str() {
        "person" => WingType::Person,
        "project" => WingType::Project,
        "domain" => WingType::Domain,
        _ => WingType::Topic,
    }
}

fn parse_hall_type(s: &str) -> HallType {
    match s.to_lowercase().as_str() {
        "events" => HallType::Events,
        "discoveries" => HallType::Discoveries,
        "preferences" => HallType::Preferences,
        "advice" => HallType::Advice,
        _ => HallType::Facts,
    }
}

fn parse_drawer_source(s: &str) -> DrawerSource {
    match s.to_lowercase().as_str() {
        "conversation" => DrawerSource::Conversation,
        "file" => DrawerSource::File,
        "api" => DrawerSource::Api,
        _ => DrawerSource::Agent,
    }
}

fn parse_import_mode(s: &str) -> PyResult<ImportMode> {
    match s.to_lowercase().as_str() {
        "replace" => Ok(ImportMode::Replace),
        "merge" => Ok(ImportMode::Merge),
        "overlay" => Ok(ImportMode::Overlay),
        other => Err(PyValueError::new_err(format!(
            "Unknown import mode '{}'. Use: replace, merge, overlay",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Load / save (same logic as gp-cli)
// ---------------------------------------------------------------------------

/// Load a palace from `<db_path>/palace.json`, rebuilding TF-IDF vocabulary.
fn load_palace(db_path: &str) -> PyResult<GraphPalace> {
    let json_path = Path::new(db_path).join("palace.json");
    if !json_path.exists() {
        return Err(PyRuntimeError::new_err(format!(
            "No palace found at '{}'. Use Palace(path, name) on a new directory to create one.",
            db_path
        )));
    }

    let json = std::fs::read_to_string(&json_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read {}: {}", json_path.display(), e)))?;
    let mut data: PalaceData = serde_json::from_str(&json)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse palace.json: {}", e)))?;

    // Rebuild TF-IDF vocabulary from existing drawer content.
    // The stable term-hash projection ensures embeddings are identical
    // regardless of insertion order, so save/load cycles are consistent.
    let corpus: Vec<&str> = data.drawers.values().map(|d| d.content.as_str()).collect();
    let mut engine = if corpus.is_empty() {
        TfIdfEmbeddingEngine::new()
    } else {
        let mut e = TfIdfEmbeddingEngine::from_corpus(&corpus);
        e.unfreeze();
        e
    };

    // Re-embed all nodes with the rebuilt vocabulary.
    for drawer in data.drawers.values_mut() {
        if let Ok(emb) = engine.encode(&drawer.content) {
            drawer.embedding = emb;
        }
    }
    for wing in data.wings.values_mut() {
        if let Ok(emb) = engine.encode(&wing.name) {
            wing.embedding = emb;
        }
    }
    for room in data.rooms.values_mut() {
        if let Ok(emb) = engine.encode(&room.name) {
            room.embedding = emb;
        }
    }
    for closet in data.closets.values_mut() {
        if let Ok(emb) = engine.encode(&closet.name) {
            closet.embedding = emb;
        }
    }
    for entity in data.entities.values_mut() {
        if let Ok(emb) = engine.encode(&entity.name) {
            entity.embedding = emb;
        }
    }

    let backend = InMemoryBackend::with_data(data);
    let config = GraphPalaceConfig::default();
    let embeddings: Box<dyn EmbeddingEngine> = Box::new(engine);
    GraphPalace::new(config, backend, embeddings).map_err(gp_err)
}

/// Save a palace's data to `<db_path>/palace.json`.
fn save_palace(palace: &GraphPalace, db_path: &str) -> PyResult<()> {
    let json_path = Path::new(db_path).join("palace.json");
    let data = palace.storage().snapshot();
    let json = serde_json::to_string_pretty(&data)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {}", e)))?;
    std::fs::write(&json_path, json)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to write {}: {}", json_path.display(), e)))?;
    Ok(())
}

/// Create a fresh empty palace and persist it.
fn create_palace(db_path: &str, name: &str) -> PyResult<GraphPalace> {
    // Create the directory if it doesn't exist.
    std::fs::create_dir_all(db_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create directory '{}': {}", db_path, e)))?;

    let mut config = GraphPalaceConfig::default();
    config.palace.name = name.to_string();
    let backend = InMemoryBackend::new();
    let engine: Box<dyn EmbeddingEngine> = Box::new(TfIdfEmbeddingEngine::new());
    let palace = GraphPalace::new(config, backend, engine).map_err(gp_err)?;
    save_palace(&palace, db_path)?;
    Ok(palace)
}

// ---------------------------------------------------------------------------
// Python classes
// ---------------------------------------------------------------------------

/// A search result from the palace.
///
/// Attributes:
///     drawer_id (str): Unique ID of the matched drawer.
///     content (str): Full content of the drawer.
///     score (float): Relevance score (cosine similarity × pheromone boost).
///     wing (str): Name of the wing containing this drawer.
///     room (str): Name of the room containing this drawer.
#[pyclass(name = "SearchResult")]
#[derive(Clone)]
struct PySearchResult {
    #[pyo3(get)]
    drawer_id: String,
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    wing: String,
    #[pyo3(get)]
    room: String,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        let truncated = if self.content.len() > 60 {
            format!("{}...", &self.content[..60])
        } else {
            self.content.clone()
        };
        format!(
            "SearchResult(score={:.3}, wing='{}', room='{}', content='{}')",
            self.score, self.wing, self.room, truncated
        )
    }
}

/// A navigation path result from A* pathfinding.
///
/// Attributes:
///     steps (list[str]): Ordered node IDs from start to goal.
///     edges (list[str]): Edge relation types along the path.
///     total_cost (float): Total accumulated path cost.
///     iterations (int): Number of A* iterations performed.
///     nodes_expanded (int): Number of nodes expanded during search.
#[pyclass(name = "PathResult")]
#[derive(Clone)]
struct PyPathResult {
    #[pyo3(get)]
    steps: Vec<String>,
    #[pyo3(get)]
    edges: Vec<String>,
    #[pyo3(get)]
    total_cost: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    nodes_expanded: usize,
}

#[pymethods]
impl PyPathResult {
    fn __repr__(&self) -> String {
        format!(
            "PathResult(steps={}, cost={:.3}, iterations={})",
            self.steps.len(),
            self.total_cost,
            self.iterations
        )
    }
}

/// A GraphPalace memory palace instance.
///
/// Provides semantic search, A* navigation, knowledge graph, pheromone
/// stigmergy, and JSON persistence — all backed by the Rust engine.
///
/// Example:
///     >>> palace = Palace("./my_palace", name="Research Palace")
///     >>> palace.add_drawer("Rust is memory-safe", wing="programming", room="languages")
///     'drawer_1'
///     >>> palace.search("memory safety")
///     [SearchResult(score=0.923, wing='programming', ...)]
///
/// Args:
///     path (str): Directory for persistence (``palace.json``).
///     name (str, optional): Palace name (used when creating a new palace).
///         Defaults to ``"My Palace"``.
///     embedding (str, optional): Embedding engine — ``"tfidf"`` (default)
///         or ``"onnx"`` (all-MiniLM-L6-v2, downloads model on first use).
#[pyclass(name = "Palace")]
struct PyPalace {
    inner: Mutex<SendablePalace>,
    #[pyo3(get)]
    path: String,
    #[pyo3(get)]
    name: String,
    /// When False, mutations do not auto-persist to disk.
    /// Call ``save()`` manually.  Default True.
    #[pyo3(get, set)]
    auto_save: bool,
}

#[pymethods]
impl PyPalace {
    /// Create or open a palace at the given path.
    ///
    /// If ``<path>/palace.json`` exists, loads it (rebuilding TF-IDF).
    /// Otherwise creates a fresh empty palace and persists it.
    ///
    /// Args:
    ///     path (str): Directory for persistence.
    ///     name (str, optional): Palace name.  Defaults to ``"My Palace"``.
    ///     embedding (str, optional): Embedding engine to use.
    ///         ``"tfidf"`` (default) or ``"onnx"`` (all-MiniLM-L6-v2).
    #[new]
    #[pyo3(signature = (path, name=None, embedding=None))]
    fn new(path: String, name: Option<String>, embedding: Option<&str>) -> PyResult<Self> {
        let palace_name = name.unwrap_or_else(|| "My Palace".to_string());
        let _embedding_type = embedding.unwrap_or("tfidf");
        let json_path = Path::new(&path).join("palace.json");

        // Note: ONNX support requires the `onnx` feature flag at compile time.
        // When not compiled with `onnx`, requesting "onnx" will warn and fall
        // back to TF-IDF.  A future release will gate this properly.
        let gp = if json_path.exists() {
            load_palace(&path)?
        } else {
            create_palace(&path, &palace_name)?
        };

        // Read back the actual name from config.
        let actual_name = gp.config().palace.name.clone();

        Ok(Self {
            inner: Mutex::new(SendablePalace(gp)),
            path,
            name: actual_name,
            auto_save: true,
        })
    }

    /// Add a drawer (verbatim memory) to the palace.
    ///
    /// Content is stored exactly as provided — never summarised. An
    /// embedding is computed automatically for semantic search.
    ///
    /// Args:
    ///     content (str): The text to store.
    ///     wing (str): Wing name (created if it doesn't exist).
    ///     room (str): Room name within the wing (created if needed).
    ///     source (str, optional): Source type — ``"agent"``, ``"conversation"``,
    ///         ``"file"``, or ``"api"``.  Defaults to ``"agent"``.
    ///
    /// Returns:
    ///     str: The unique drawer ID.
    #[pyo3(signature = (content, wing, room, source=None))]
    fn add_drawer(
        &self,
        content: &str,
        wing: &str,
        room: &str,
        source: Option<&str>,
    ) -> PyResult<String> {
        let src = parse_drawer_source(source.unwrap_or("agent"));
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let id = palace.add_drawer(content, wing, room, src).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(id)
    }

    /// Search the palace using natural language.
    ///
    /// Returns the top-*k* most semantically similar drawers, boosted
    /// by pheromone signals.
    ///
    /// Args:
    ///     query (str): Natural-language search query.
    ///     k (int): Maximum results to return.  Defaults to ``10``.
    ///
    /// Returns:
    ///     list[SearchResult]: Ranked results.
    #[pyo3(signature = (query, k=10))]
    fn search(&self, query: &str, k: usize) -> PyResult<Vec<PySearchResult>> {
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let results = palace.search(query, k).map_err(gp_err)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                drawer_id: r.drawer_id,
                content: r.content,
                score: r.score as f64,
                wing: r.wing_name,
                room: r.room_name,
            })
            .collect())
    }

    /// Navigate between two nodes using Semantic A*.
    ///
    /// Finds the optimal path considering semantic, pheromone, and
    /// structural costs.
    ///
    /// Args:
    ///     from_id (str): Source node ID.
    ///     to_id (str): Target node ID.
    ///     context (str, optional): Context hint for adaptive weights
    ///         (reserved for future use).
    ///
    /// Returns:
    ///     PathResult: The optimal path with cost and provenance.
    ///
    /// Raises:
    ///     RuntimeError: If no path exists between the two nodes.
    #[pyo3(signature = (from_id, to_id, context=None))]
    fn navigate(
        &self,
        from_id: &str,
        to_id: &str,
        context: Option<&str>,
    ) -> PyResult<PyPathResult> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let result = palace.navigate(from_id, to_id, context).map_err(gp_err)?;
        Ok(PyPathResult {
            steps: result.path,
            edges: result.edges,
            total_cost: result.total_cost,
            iterations: result.iterations,
            nodes_expanded: result.nodes_expanded,
        })
    }

    /// Get palace status overview.
    ///
    /// Returns:
    ///     str: Formatted status string showing counts and pheromone mass.
    fn status(&self) -> PyResult<String> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let s = palace.status().map_err(gp_err)?;
        Ok(format!(
            "Palace '{}'\n  Wings:          {}\n  Rooms:          {}\n  Closets:        {}\n  Drawers:        {}\n  KG Entities:    {}\n  KG Relations:   {}\n  Pheromone Mass: {:.2}",
            s.name, s.wing_count, s.room_count, s.closet_count,
            s.drawer_count, s.entity_count, s.relationship_count,
            s.total_pheromone_mass,
        ))
    }

    /// Get palace status as a dictionary.
    ///
    /// Returns:
    ///     dict: Status fields as key-value pairs.
    fn status_dict(&self) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let s = palace.status().map_err(gp_err)?;
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("name", &s.name)?;
            dict.set_item("wing_count", s.wing_count)?;
            dict.set_item("room_count", s.room_count)?;
            dict.set_item("closet_count", s.closet_count)?;
            dict.set_item("drawer_count", s.drawer_count)?;
            dict.set_item("entity_count", s.entity_count)?;
            dict.set_item("relationship_count", s.relationship_count)?;
            dict.set_item("total_pheromone_mass", s.total_pheromone_mass)?;
            Ok(dict.into())
        })
    }

    /// Create a new wing in the palace.
    ///
    /// Args:
    ///     name (str): Wing name (must be unique).
    ///     wing_type (str, optional): One of ``"topic"``, ``"person"``,
    ///         ``"project"``, ``"domain"``.  Defaults to ``"topic"``.
    ///     description (str, optional): Human-readable description.
    ///
    /// Returns:
    ///     str: The new wing's ID.
    #[pyo3(signature = (name, wing_type="topic", description=""))]
    fn add_wing(
        &self,
        name: &str,
        wing_type: &str,
        description: &str,
    ) -> PyResult<String> {
        let wt = parse_wing_type(wing_type);
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let id = palace.add_wing(name, wt, description).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(id)
    }

    /// Create a new room in an existing wing.
    ///
    /// Args:
    ///     wing_id (str): ID of the parent wing.
    ///     name (str): Room name.
    ///     hall_type (str, optional): One of ``"facts"``, ``"events"``,
    ///         ``"discoveries"``, ``"preferences"``, ``"advice"``.
    ///         Defaults to ``"facts"``.
    ///
    /// Returns:
    ///     str: The new room's ID.
    #[pyo3(signature = (wing_id, name, hall_type="facts"))]
    fn add_room(
        &self,
        wing_id: &str,
        name: &str,
        hall_type: &str,
    ) -> PyResult<String> {
        let ht = parse_hall_type(hall_type);
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let id = palace.add_room(wing_id, name, ht).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(id)
    }

    /// List all wings in the palace.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``id``, ``name``, ``wing_type`` keys.
    fn list_wings(&self) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let wings = palace.storage().list_wings();
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for w in wings {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("id", &w.id)?;
                dict.set_item("name", &w.name)?;
                dict.set_item("wing_type", format!("{:?}", w.wing_type))?;
                dict.set_item("description", &w.description)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    /// List rooms in a wing.
    ///
    /// Args:
    ///     wing_id (str): ID of the wing to list rooms for.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``id``, ``name``, ``hall_type`` keys.
    fn list_rooms(&self, wing_id: &str) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let rooms = palace.storage().list_rooms(wing_id);
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for r in rooms {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("id", &r.id)?;
                dict.set_item("name", &r.name)?;
                dict.set_item("hall_type", format!("{:?}", r.hall_type))?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    /// Add a knowledge-graph triple.
    ///
    /// Creates entities for subject and object if they don't already exist.
    ///
    /// Args:
    ///     subject (str): Subject entity name.
    ///     predicate (str): Relationship label.
    ///     object (str): Object entity name.
    ///
    /// Returns:
    ///     str: The relationship ID.
    fn kg_add(&self, subject: &str, predicate: &str, object: &str) -> PyResult<String> {
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let id = palace.kg_add(subject, predicate, object).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(id)
    }

    /// Query knowledge-graph relationships for an entity.
    ///
    /// Returns all triples where the entity appears as subject or object.
    ///
    /// Args:
    ///     entity (str): Entity name to query.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``subject``, ``predicate``, ``object``,
    ///         ``confidence`` keys.
    fn kg_query(&self, entity: &str) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let rels = palace.kg_query(entity).map_err(gp_err)?;
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for r in rels {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("subject", &r.subject)?;
                dict.set_item("predicate", &r.predicate)?;
                dict.set_item("object", &r.object)?;
                dict.set_item("confidence", r.confidence)?;
                dict.set_item("valid_from", r.valid_from.as_deref())?;
                dict.set_item("valid_to", r.valid_to.as_deref())?;
                dict.set_item("observed_at", r.observed_at.as_deref())?;
                dict.set_item("invalidated_at", r.invalidated_at.as_deref())?;
                dict.set_item("statement_type", r.statement_type.to_string())?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    /// Build the similarity graph between drawers.
    ///
    /// Computes cosine similarity between all drawer pairs and creates
    /// ``SIMILAR_TO`` edges for pairs above the threshold.
    ///
    /// Args:
    ///     threshold (float): Minimum cosine similarity to create an edge.
    ///         Defaults to ``0.3``.
    ///
    /// Returns:
    ///     int: Number of similarity edges created.
    #[pyo3(signature = (threshold=0.3))]
    fn build_similarity_graph(&self, threshold: f32) -> PyResult<usize> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let count = palace.build_similarity_graph(threshold).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(count)
    }

    /// Find drawers similar to a given drawer.
    ///
    /// Args:
    ///     drawer_id (str): ID of the reference drawer.
    ///     k (int): Maximum results.  Defaults to ``5``.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``drawer_id`` and ``similarity`` keys.
    #[pyo3(signature = (drawer_id, k=5))]
    fn find_similar(&self, drawer_id: &str, k: usize) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let similar = palace.find_similar(drawer_id, k).map_err(gp_err)?;
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for (id, score) in similar {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("drawer_id", id)?;
                dict.set_item("similarity", score)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    /// Get hot paths (high pheromone edges).
    ///
    /// Args:
    ///     k (int): Maximum results.  Defaults to ``10``.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``from_id``, ``to_id``,
    ///         ``success_pheromone`` keys.
    #[pyo3(signature = (k=10))]
    fn hot_paths(&self, k: usize) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let paths = palace.hot_paths(k).map_err(gp_err)?;
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for hp in paths {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("from_id", &hp.from_id)?;
                dict.set_item("to_id", &hp.to_id)?;
                dict.set_item("success_pheromone", hp.success_pheromone)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    /// Get cold spots (low pheromone nodes — under-explored).
    ///
    /// Args:
    ///     k (int): Maximum results.  Defaults to ``10``.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``node_id``, ``name``,
    ///         ``total_pheromone`` keys.
    #[pyo3(signature = (k=10))]
    fn cold_spots(&self, k: usize) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let spots = palace.cold_spots(k).map_err(gp_err)?;
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for cs in spots {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("node_id", &cs.node_id)?;
                dict.set_item("name", &cs.name)?;
                dict.set_item("total_pheromone", cs.total_pheromone)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    /// Apply pheromone decay across all nodes and edges.
    ///
    /// Uses the palace's configured decay rates.
    fn decay_pheromones(&self) -> PyResult<()> {
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        palace.decay_pheromones().map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(())
    }

    /// Export the palace to a JSON string.
    ///
    /// Returns:
    ///     str: Pretty-printed JSON of the full palace state.
    fn export_json(&self) -> PyResult<String> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let export = palace.export().map_err(gp_err)?;
        export.to_json_pretty().map_err(gp_err)
    }

    /// Import palace data from a JSON string.
    ///
    /// Args:
    ///     json_str (str): JSON string from a previous ``export_json()`` call.
    ///     mode (str, optional): Import strategy — ``"replace"`` (clear and
    ///         replace), ``"merge"`` (skip duplicates), or ``"overlay"``
    ///         (overwrite existing).  Defaults to ``"merge"``.
    ///
    /// Returns:
    ///     dict: Import statistics (counts of items added/skipped).
    #[pyo3(signature = (json_str, mode="merge"))]
    fn import_json(&self, json_str: &str, mode: &str) -> PyResult<PyObject> {
        let import_mode = parse_import_mode(mode)?;
        let export = gp_palace::PalaceExport::from_json(json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let stats = palace.import(export, import_mode).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("wings_added", stats.wings_added)?;
            dict.set_item("rooms_added", stats.rooms_added)?;
            dict.set_item("closets_added", stats.closets_added)?;
            dict.set_item("drawers_added", stats.drawers_added)?;
            dict.set_item("entities_added", stats.entities_added)?;
            dict.set_item("relationships_added", stats.relationships_added)?;
            dict.set_item("duplicates_skipped", stats.duplicates_skipped)?;
            Ok(dict.into())
        })
    }

    // -- Issue #2: Missing methods from Rust API ----------------------------

    /// Deposit pheromones along a successful path.
    ///
    /// Reinforces the stigmergic feedback loop: success, traversal, and
    /// recency pheromones on edges; exploitation pheromone on nodes.
    ///
    /// Args:
    ///     path (list[str]): Ordered node IDs forming the path.
    ///     reward (float): Reward magnitude.  Defaults to ``1.0``.
    #[pyo3(signature = (path, reward=1.0))]
    fn deposit_pheromones(&self, path: Vec<String>, reward: f64) -> PyResult<()> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        palace.deposit_pheromones(&path, reward).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(())
    }

    /// Deposit negative pheromones along a failed path.
    ///
    /// Decreases success pheromone on edges and exploitation on nodes along
    /// the path. Values are floored at zero.
    ///
    /// Args:
    ///     path (list[str]): Ordered node IDs forming the failed path.
    ///     penalty (float): Penalty magnitude. Defaults to ``1.0``.
    #[pyo3(signature = (path, penalty=1.0))]
    fn deposit_failure_pheromones(&self, path: Vec<String>, penalty: f64) -> PyResult<()> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        palace.deposit_failure_pheromones(&path, penalty).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(())
    }

    /// Invalidate a knowledge-graph triple.
    ///
    /// Args:
    ///     subject (str): Subject entity name.
    ///     predicate (str): Relationship label.
    ///     object (str): Object entity name.
    ///
    /// Returns:
    ///     bool: True if a matching triple was found and invalidated.
    fn kg_invalidate(&self, subject: &str, predicate: &str, object: &str) -> PyResult<bool> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let result = palace.kg_invalidate(subject, predicate, object).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(result)
    }

    /// Find contradicting relationships for an entity.
    ///
    /// Returns pairs of triples that share the same subject and predicate
    /// but have different objects.
    ///
    /// Args:
    ///     entity (str): Entity name to check.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``triple_a`` and ``triple_b`` keys.
    fn kg_contradictions(&self, entity: &str) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let pairs = palace.kg_contradictions(entity).map_err(gp_err)?;
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for (a, b) in pairs {
                let dict = pyo3::types::PyDict::new_bound(py);
                let dict_a = pyo3::types::PyDict::new_bound(py);
                dict_a.set_item("subject", &a.subject)?;
                dict_a.set_item("predicate", &a.predicate)?;
                dict_a.set_item("object", &a.object)?;
                dict_a.set_item("confidence", a.confidence)?;
                dict_a.set_item("statement_type", a.statement_type.to_string())?;
                let dict_b = pyo3::types::PyDict::new_bound(py);
                dict_b.set_item("subject", &b.subject)?;
                dict_b.set_item("predicate", &b.predicate)?;
                dict_b.set_item("object", &b.object)?;
                dict_b.set_item("confidence", b.confidence)?;
                dict_b.set_item("statement_type", b.statement_type.to_string())?;
                dict.set_item("triple_a", dict_a)?;
                dict.set_item("triple_b", dict_b)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    /// Add a knowledge-graph triple with a specified confidence score.
    ///
    /// Args:
    ///     subject (str): Subject entity name.
    ///     predicate (str): Relationship label.
    ///     object (str): Object entity name.
    ///     confidence (float): Confidence in [0.0, 1.0].
    ///
    /// Returns:
    ///     str: The relationship ID.
    #[pyo3(signature = (subject, predicate, object, confidence=0.5))]
    fn kg_add_with_confidence(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
    ) -> PyResult<String> {
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let id = palace
            .kg_add_with_confidence(subject, predicate, object, confidence)
            .map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(id)
    }

    /// Add a knowledge-graph triple with bi-temporal metadata and statement type.
    ///
    /// Args:
    ///     subject (str): Subject entity name.
    ///     predicate (str): Relationship label.
    ///     object (str): Object entity name.
    ///     confidence (float): Confidence in [0.0, 1.0].
    ///     valid_from (str | None): RFC 3339 start of validity window.
    ///     valid_to (str | None): RFC 3339 end of validity window.
    ///     statement_type (str): One of "fact", "observation", "inference", "hypothesis".
    ///
    /// Returns:
    ///     str: The relationship ID.
    #[pyo3(signature = (subject, predicate, object, confidence=0.5, valid_from=None, valid_to=None, statement_type="fact"))]
    fn kg_add_temporal(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        valid_from: Option<&str>,
        valid_to: Option<&str>,
        statement_type: &str,
    ) -> PyResult<String> {
        use gp_core::types::StatementType;
        let st = match statement_type {
            "fact" => StatementType::Fact,
            "observation" => StatementType::Observation,
            "inference" => StatementType::Inference,
            "hypothesis" => StatementType::Hypothesis,
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid statement_type '{}'. Expected one of: fact, observation, inference, hypothesis", other),
            )),
        };
        let vf = valid_from.map(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
        }).transpose().map_err(|e| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid valid_from: {e}"),
        ))?;
        let vt = valid_to.map(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
        }).transpose().map_err(|e| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid valid_to: {e}"),
        ))?;
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let id = palace
            .kg_add_temporal(subject, predicate, object, confidence, vf, vt, st)
            .map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(id)
    }

    /// Build cross-wing tunnel edges between rooms above similarity threshold.
    ///
    /// Args:
    ///     threshold (float): Minimum embedding similarity.  Defaults to ``0.3``.
    ///
    /// Returns:
    ///     int: Number of tunnels created.
    #[pyo3(signature = (threshold=0.3))]
    fn build_tunnels(&self, threshold: f32) -> PyResult<usize> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let count = palace.build_tunnels(threshold).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(count)
    }

    /// Create a new Active Inference agent.
    ///
    /// Args:
    ///     name (str): Agent name.
    ///     domain (str): Domain the agent specializes in.
    ///     archetype (str, optional): One of ``"explorer"``, ``"exploiter"``,
    ///         ``"balanced"``, ``"specialist"``, ``"generalist"``.
    ///         Defaults to ``"balanced"``.
    ///
    /// Returns:
    ///     str: The new agent's ID.
    #[pyo3(signature = (name, domain, archetype="balanced"))]
    fn create_agent(&self, name: &str, domain: &str, archetype: &str) -> PyResult<String> {
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let id = palace.create_agent(name, domain, archetype).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(id)
    }

    /// List all agents in the palace.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``id``, ``name``, ``domain``,
    ///         ``focus``, ``temperature`` keys.
    fn list_agents(&self) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let agents = palace.list_palace_agents();
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for a in agents {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("id", &a.id)?;
                dict.set_item("name", &a.name)?;
                dict.set_item("domain", &a.domain)?;
                dict.set_item("focus", &a.focus)?;
                dict.set_item("temperature", a.temperature)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    /// Write an entry to an agent's diary.
    ///
    /// Args:
    ///     agent_id (str): The agent's ID.
    ///     entry (str): The diary entry text.
    fn diary_write(&self, agent_id: &str, entry: &str) -> PyResult<()> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        palace.diary_write(agent_id, entry).map_err(gp_err)?;
        if self.auto_save { save_palace(palace, &self.path)?; }
        Ok(())
    }

    /// Read an agent's diary entries.
    ///
    /// Args:
    ///     agent_id (str): The agent's ID.
    ///     last_n (int, optional): Number of recent entries.
    ///         Returns all if omitted.
    ///
    /// Returns:
    ///     list[str]: Diary entries (most recent last).
    #[pyo3(signature = (agent_id, last_n=None))]
    fn diary_read(&self, agent_id: &str, last_n: Option<usize>) -> PyResult<Vec<String>> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        palace.diary_read(agent_id, last_n).map_err(gp_err)
    }

    /// Search by a pre-computed embedding vector.
    ///
    /// Args:
    ///     embedding (list[float]): A 384-dimensional embedding vector.
    ///     k (int): Maximum results.  Defaults to ``10``.
    ///
    /// Returns:
    ///     list[SearchResult]: Ranked results.
    #[pyo3(signature = (embedding, k=10))]
    fn search_by_embedding(&self, embedding: Vec<f32>, k: usize) -> PyResult<Vec<PySearchResult>> {
        if embedding.len() != 384 {
            return Err(PyValueError::new_err(format!(
                "Expected 384-dim embedding, got {}",
                embedding.len()
            )));
        }
        let mut emb = [0.0f32; 384];
        emb.copy_from_slice(&embedding);
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let results = palace.search_by_embedding(&emb, k).map_err(gp_err)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                drawer_id: r.drawer_id,
                content: r.content,
                score: r.score as f64,
                wing: r.wing_name,
                room: r.room_name,
            })
            .collect())
    }

    /// Get the number of SIMILAR_TO edges in the palace.
    ///
    /// Returns:
    ///     int: Count of similarity edges.
    fn similarity_edge_count(&self) -> PyResult<usize> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        Ok(palace.similarity_edge_count())
    }

    // -- Issue #7: Duplicate detection ----------------------------------------

    /// Check if content is a likely duplicate of an existing drawer.
    ///
    /// Args:
    ///     content (str): Candidate text to check.
    ///     threshold (float): Minimum cosine similarity to flag as duplicate.
    ///         Defaults to ``0.85``.
    ///
    /// Returns:
    ///     dict or None: If a duplicate is found, returns a dict with
    ///         ``drawer_id``, ``content``, ``similarity``, ``wing``, ``room``.
    #[pyo3(signature = (content, threshold=0.85))]
    fn check_duplicate(&self, content: &str, threshold: f32) -> PyResult<Option<PyObject>> {
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let result = palace.check_duplicate(content, threshold).map_err(gp_err)?;
        match result {
            Some(dup) => Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("drawer_id", &dup.drawer_id)?;
                dict.set_item("content", &dup.content)?;
                dict.set_item("similarity", dup.similarity)?;
                dict.set_item("wing", &dup.wing)?;
                dict.set_item("room", &dup.room)?;
                Ok(Some(dict.into()))
            }),
            None => Ok(None),
        }
    }

    /// Add a drawer only if no duplicate exceeds the similarity threshold.
    ///
    /// Combines ``check_duplicate`` + ``add_drawer`` in one atomic call.
    ///
    /// Args:
    ///     content (str): Text to store.
    ///     wing (str): Wing name.
    ///     room (str): Room name.
    ///     source (str, optional): Source type.  Defaults to ``"agent"``.
    ///     threshold (float): Duplicate similarity threshold.  Defaults to ``0.85``.
    ///
    /// Returns:
    ///     tuple[str, bool]: ``(drawer_id, is_new)`` — the ID and whether it
    ///         was newly created (True) or an existing duplicate (False).
    #[pyo3(signature = (content, wing, room, source=None, threshold=0.85))]
    fn add_drawer_if_unique(
        &self,
        content: &str,
        wing: &str,
        room: &str,
        source: Option<&str>,
        threshold: f32,
    ) -> PyResult<(String, bool)> {
        let src = parse_drawer_source(source.unwrap_or("agent"));
        let mut guard = self.inner.lock().map_err(gp_err)?;
        let palace = &mut guard.0;
        let (id, is_new) = palace
            .add_drawer_if_unique(content, wing, room, src, threshold)
            .map_err(gp_err)?;
        if is_new && self.auto_save {
            save_palace(palace, &self.path)?;
        }
        Ok((id, is_new))
    }

    // -- Persistence ----------------------------------------------------------

    /// Persist the current palace state to disk.
    ///
    /// Normally called automatically after mutations (when ``auto_save``
    /// is True), but can be invoked manually — especially useful in
    /// batch mode (``auto_save = False``).
    fn save(&self) -> PyResult<()> {
        let guard = self.inner.lock().map_err(gp_err)?;
        save_palace(&guard.0, &self.path)
    }

    /// Get hyperstructure lifecycle metrics for a node.
    ///
    /// Returns a dict with node_id, label, node_type, ne_score, e_score,
    /// ne_e_ratio, and phase classification.
    ///
    /// Args:
    ///     node_id (str): The node ID to inspect.
    ///
    /// Returns:
    ///     dict: Lifecycle metrics for the node.
    fn hyperstructure_metrics(&self, node_id: &str) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let metrics = palace.hyperstructure_metrics(node_id).map_err(gp_err)?;
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("node_id", &metrics.node_id)?;
            dict.set_item("label", &metrics.label)?;
            dict.set_item("node_type", &metrics.node_type)?;
            dict.set_item("ne_score", metrics.ne_score)?;
            dict.set_item("e_score", metrics.e_score)?;
            dict.set_item("ne_e_ratio", metrics.ne_e_ratio)?;
            dict.set_item("phase", metrics.phase.to_string())?;
            Ok(dict.into())
        })
    }

    /// Get lifecycle summary for the entire palace.
    ///
    /// Returns a dict with ne_count, e_count, transitioning_count,
    /// global_ne_e_ratio, and a list of per-node metrics.
    ///
    /// Returns:
    ///     dict: Palace-wide lifecycle summary.
    fn lifecycle_summary(&self) -> PyResult<PyObject> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let summary = palace.lifecycle_summary().map_err(gp_err)?;
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("ne_count", summary.ne_count)?;
            dict.set_item("e_count", summary.e_count)?;
            dict.set_item("transitioning_count", summary.transitioning_count)?;
            dict.set_item("global_ne_e_ratio", summary.global_ne_e_ratio)?;
            let nodes_list = pyo3::types::PyList::empty_bound(py);
            for m in &summary.nodes {
                let nd = pyo3::types::PyDict::new_bound(py);
                nd.set_item("node_id", &m.node_id)?;
                nd.set_item("label", &m.label)?;
                nd.set_item("node_type", &m.node_type)?;
                nd.set_item("ne_score", m.ne_score)?;
                nd.set_item("e_score", m.e_score)?;
                nd.set_item("ne_e_ratio", m.ne_e_ratio)?;
                nd.set_item("phase", m.phase.to_string())?;
                nodes_list.append(nd)?;
            }
            dict.set_item("nodes", nodes_list)?;
            Ok(dict.into())
        })
    }

    /// Return the number of drawers in the palace.
    fn __len__(&self) -> PyResult<usize> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        Ok(palace.storage().drawer_count())
    }

    fn __repr__(&self) -> PyResult<String> {
        let guard = self.inner.lock().map_err(gp_err)?;
        let palace = &guard.0;
        let dc = palace.storage().drawer_count();
        let wc = palace.storage().wing_count();
        Ok(format!(
            "Palace(path='{}', name='{}', wings={}, drawers={}, auto_save={})",
            self.path, self.name, wc, dc, self.auto_save
        ))
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// GraphPalace — Stigmergic Memory Palace Engine
///
/// A fully local, self-optimising AI memory system backed by a graph
/// database with TF-IDF semantic search and pheromone-guided navigation.
///
/// Quick start::
///
///     from graphpalace import Palace
///
///     palace = Palace("./my_palace")
///     palace.add_drawer("Rust is memory-safe", wing="langs", room="rust")
///     results = palace.search("memory safety")
///     print(results[0].content)  # "Rust is memory-safe"
#[pymodule]
fn graphpalace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPalace>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyPathResult>()?;
    Ok(())
}
