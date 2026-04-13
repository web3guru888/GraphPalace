//! JavaScript-facing API surface for GraphPalace WASM bindings.
//!
//! These types define the public API that will be exported via `wasm_bindgen`
//! in Phase 6. For now they are plain Rust structs with `todo!()` method bodies.

/// JavaScript-facing palace handle (will be `#[wasm_bindgen]` exported).
pub struct JsPalace {
    // Will hold a gp-core GraphPalaceConfig + runtime state.
    _private: (),
}

/// JavaScript-facing search result.
pub struct JsSearchResult {
    pub id: String,
    pub content: String,
    pub score: f64,
}

/// JavaScript-facing path result.
pub struct JsPathResult {
    pub path: Vec<String>,
    pub total_cost: f64,
}

impl JsPalace {
    /// Create a new in-memory palace instance.
    ///
    /// Phase 6: will accept a JS config object and initialise IndexedDB/OPFS storage.
    pub fn new() -> Self {
        todo!("Phase 6: initialise GraphPalace with WASM-compatible storage")
    }

    /// Semantic search across all drawers in the palace.
    ///
    /// Phase 6: will accept a JS string and return a `JsValue` array.
    pub fn search(&self, _query: &str, _k: usize) -> Vec<JsSearchResult> {
        todo!("Phase 6: delegate to gp-core semantic search")
    }

    /// Find the shortest path between two nodes in the knowledge graph.
    ///
    /// Phase 6: will return a JS-serialisable path or `undefined`.
    pub fn navigate(&self, _from: &str, _to: &str) -> Option<JsPathResult> {
        todo!("Phase 6: delegate to gp-core graph traversal")
    }

    /// Add a new drawer (memory item) to the palace.
    ///
    /// Returns the generated drawer ID.
    ///
    /// Phase 6: will accept JS strings and return a `JsValue` ID.
    pub fn add_drawer(&mut self, _content: &str, _wing: &str, _room: &str) -> String {
        todo!("Phase 6: delegate to gp-core drawer creation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_result_creation() {
        let result = JsSearchResult {
            id: "drawer-001".into(),
            content: "test content".into(),
            score: 0.95,
        };
        assert_eq!(result.id, "drawer-001");
        assert_eq!(result.content, "test content");
        assert!((result.score - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn path_result_creation() {
        let result = JsPathResult {
            path: vec!["a".into(), "b".into(), "c".into()],
            total_cost: 1.5,
        };
        assert_eq!(result.path.len(), 3);
        assert_eq!(result.path[0], "a");
        assert!((result.total_cost - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn path_result_empty_path() {
        let result = JsPathResult {
            path: vec![],
            total_cost: 0.0,
        };
        assert!(result.path.is_empty());
    }

    #[test]
    #[should_panic(expected = "Phase 6")]
    fn js_palace_new_is_stub() {
        let _palace = JsPalace::new();
    }
}
