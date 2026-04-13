use pyo3::prelude::*;

/// A GraphPalace memory palace instance.
///
/// Example usage:
/// ```python
/// from graphpalace import Palace
///
/// palace = Palace("./my_palace")
/// palace.add_drawer(
///     content="The team decided to use Postgres for concurrent write support",
///     wing="project_orion",
///     room="database_decisions",
/// )
/// results = palace.search("why did we choose Postgres?", k=5)
/// for result in results:
///     print(f"[{result.score:.2f}] {result.content[:100]}")
/// ```
#[pyclass]
struct Palace {
    #[pyo3(get)]
    path: String,
    #[pyo3(get)]
    name: String,
}

/// A search result from the palace.
#[pyclass]
#[derive(Clone)]
struct SearchResult {
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

/// A navigation path result.
#[pyclass]
#[derive(Clone)]
struct PathResult {
    #[pyo3(get)]
    steps: Vec<String>,
    #[pyo3(get)]
    total_cost: f64,
    #[pyo3(get)]
    iterations: usize,
}

#[pymethods]
impl Palace {
    /// Create or open a palace at the given path.
    #[new]
    #[pyo3(signature = (path, name=None))]
    fn new(path: String, name: Option<String>) -> Self {
        Palace {
            path,
            name: name.unwrap_or_else(|| "My Palace".to_string()),
        }
    }

    /// Add a drawer (verbatim memory) to the palace.
    ///
    /// Content is stored exactly as provided — never summarized.
    /// An embedding is automatically computed for semantic search.
    #[pyo3(signature = (content, wing, room, source=None))]
    fn add_drawer(
        &self,
        content: &str,
        wing: &str,
        room: &str,
        source: Option<&str>,
    ) -> PyResult<String> {
        let _ = (content, wing, room, source);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Palace.add_drawer() is not yet implemented — requires Kuzu FFI backend",
        ))
    }

    /// Search the palace using natural language.
    ///
    /// Returns the top-k most semantically similar drawers,
    /// boosted by pheromone signals.
    #[pyo3(signature = (query, k=10, wing=None, room=None))]
    fn search(
        &self,
        query: &str,
        k: usize,
        wing: Option<&str>,
        room: Option<&str>,
    ) -> PyResult<Vec<SearchResult>> {
        let _ = (query, k, wing, room);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Palace.search() is not yet implemented — requires Kuzu FFI backend",
        ))
    }

    /// Navigate between two nodes using Semantic A*.
    ///
    /// Returns the optimal path with cost and provenance.
    #[pyo3(signature = (from_id, to_id, context=None))]
    fn navigate(
        &self,
        from_id: &str,
        to_id: &str,
        context: Option<&str>,
    ) -> PyResult<PathResult> {
        let _ = (from_id, to_id, context);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Palace.navigate() is not yet implemented — requires Kuzu FFI backend",
        ))
    }

    /// Get palace status overview.
    fn status(&self) -> PyResult<String> {
        Ok(format!(
            "Palace '{}' at {} (not yet connected to Kuzu backend)",
            self.name, self.path
        ))
    }

    fn __repr__(&self) -> String {
        format!("Palace(path='{}', name='{}')", self.path, self.name)
    }
}

/// GraphPalace — Stigmergic Memory Palace Engine
///
/// A fully local, self-optimizing AI memory system backed by a graph database.
///
/// Quick start:
/// ```python
/// from graphpalace import Palace
/// palace = Palace("./my_palace")
/// ```
#[pymodule]
fn graphpalace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Palace>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<PathResult>()?;
    Ok(())
}
