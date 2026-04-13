//! Web Worker integration types for offloading heavy computation.
//!
//! Defines message types for communicating between the main thread
//! and a Web Worker that runs GraphPalace operations. The actual
//! JavaScript interop is handled in WASM builds only.

use serde::{Deserialize, Serialize};

/// Messages sent from the main thread to the worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerRequest {
    /// Initialize the palace in the worker.
    Init {
        /// Palace name.
        name: String,
    },
    /// Add a drawer (potentially expensive due to embedding generation).
    AddDrawer {
        closet_id: String,
        content: String,
        source: String,
    },
    /// Search drawers by query text.
    Search {
        query: String,
        k: usize,
    },
    /// Navigate between two nodes.
    Navigate {
        from_id: String,
        to_id: String,
    },
    /// Apply pheromone decay cycle.
    DecayCycle,
    /// Export the palace as JSON.
    Export,
    /// Import a palace from JSON.
    Import {
        json: String,
    },
    /// Get palace overview statistics.
    GetOverview,
    /// Deposit pheromones along a path.
    DepositPheromones {
        path: Vec<String>,
        base_reward: f64,
    },
}

/// Messages sent from the worker back to the main thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerResponse {
    /// Palace initialized successfully.
    Initialized {
        node_count: usize,
    },
    /// Drawer added.
    DrawerAdded {
        drawer_id: String,
    },
    /// Search results.
    SearchResults {
        results: Vec<WorkerSearchResult>,
    },
    /// Navigation result.
    NavigationResult {
        path: Vec<String>,
        total_cost: f64,
    },
    /// Decay cycle completed.
    DecayComplete {
        cycle: usize,
    },
    /// Palace exported as JSON.
    Exported {
        json: String,
    },
    /// Import result.
    Imported {
        success: bool,
        node_count: usize,
    },
    /// Palace overview.
    Overview {
        name: String,
        wing_count: usize,
        room_count: usize,
        drawer_count: usize,
        total_nodes: usize,
        total_edges: usize,
    },
    /// Pheromones deposited.
    PheromonesDeposited {
        path_length: usize,
    },
    /// An error occurred.
    Error {
        message: String,
    },
}

/// Search result in worker messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSearchResult {
    pub id: String,
    pub content: String,
    pub similarity: f64,
}

/// Serialize a worker request to JSON for postMessage.
pub fn serialize_request(request: &WorkerRequest) -> Result<String, serde_json::Error> {
    serde_json::to_string(request)
}

/// Deserialize a worker request from JSON.
pub fn deserialize_request(json: &str) -> Result<WorkerRequest, serde_json::Error> {
    serde_json::from_str(json)
}

/// Serialize a worker response to JSON for postMessage.
pub fn serialize_response(response: &WorkerResponse) -> Result<String, serde_json::Error> {
    serde_json::to_string(response)
}

/// Deserialize a worker response from JSON.
pub fn deserialize_response(json: &str) -> Result<WorkerResponse, serde_json::Error> {
    serde_json::from_str(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_init_roundtrip() {
        let req = WorkerRequest::Init {
            name: "Test".to_string(),
        };
        let json = serialize_request(&req).unwrap();
        let req2 = deserialize_request(&json).unwrap();
        match req2 {
            WorkerRequest::Init { name } => assert_eq!(name, "Test"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn request_add_drawer_roundtrip() {
        let req = WorkerRequest::AddDrawer {
            closet_id: "c1".into(),
            content: "Test content".into(),
            source: "conversation".into(),
        };
        let json = serialize_request(&req).unwrap();
        let req2 = deserialize_request(&json).unwrap();
        match req2 {
            WorkerRequest::AddDrawer {
                closet_id,
                content,
                source,
            } => {
                assert_eq!(closet_id, "c1");
                assert_eq!(content, "Test content");
                assert_eq!(source, "conversation");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn request_search_roundtrip() {
        let req = WorkerRequest::Search {
            query: "test query".into(),
            k: 5,
        };
        let json = serialize_request(&req).unwrap();
        assert!(json.contains("test query"));
        assert!(json.contains("5"));
    }

    #[test]
    fn response_initialized_roundtrip() {
        let resp = WorkerResponse::Initialized { node_count: 42 };
        let json = serialize_response(&resp).unwrap();
        let resp2 = deserialize_response(&json).unwrap();
        match resp2 {
            WorkerResponse::Initialized { node_count } => assert_eq!(node_count, 42),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn response_search_results() {
        let resp = WorkerResponse::SearchResults {
            results: vec![
                WorkerSearchResult {
                    id: "d1".into(),
                    content: "Result 1".into(),
                    similarity: 0.95,
                },
                WorkerSearchResult {
                    id: "d2".into(),
                    content: "Result 2".into(),
                    similarity: 0.80,
                },
            ],
        };
        let json = serialize_response(&resp).unwrap();
        assert!(json.contains("d1"));
        assert!(json.contains("0.95"));
    }

    #[test]
    fn response_error() {
        let resp = WorkerResponse::Error {
            message: "Something went wrong".into(),
        };
        let json = serialize_response(&resp).unwrap();
        assert!(json.contains("Something went wrong"));
    }

    #[test]
    fn all_request_variants_serialize() {
        let requests = vec![
            WorkerRequest::Init { name: "P".into() },
            WorkerRequest::AddDrawer {
                closet_id: "c".into(),
                content: "x".into(),
                source: "api".into(),
            },
            WorkerRequest::Search {
                query: "q".into(),
                k: 10,
            },
            WorkerRequest::Navigate {
                from_id: "a".into(),
                to_id: "b".into(),
            },
            WorkerRequest::DecayCycle,
            WorkerRequest::Export,
            WorkerRequest::Import {
                json: "{}".into(),
            },
            WorkerRequest::GetOverview,
            WorkerRequest::DepositPheromones {
                path: vec!["a".into(), "b".into()],
                base_reward: 1.0,
            },
        ];
        for req in &requests {
            let json = serialize_request(req).unwrap();
            assert!(!json.is_empty());
            let _: WorkerRequest = deserialize_request(&json).unwrap();
        }
    }

    #[test]
    fn all_response_variants_serialize() {
        let responses = vec![
            WorkerResponse::Initialized { node_count: 1 },
            WorkerResponse::DrawerAdded {
                drawer_id: "d1".into(),
            },
            WorkerResponse::SearchResults { results: vec![] },
            WorkerResponse::NavigationResult {
                path: vec![],
                total_cost: 0.0,
            },
            WorkerResponse::DecayComplete { cycle: 5 },
            WorkerResponse::Exported {
                json: "{}".into(),
            },
            WorkerResponse::Imported {
                success: true,
                node_count: 10,
            },
            WorkerResponse::Overview {
                name: "P".into(),
                wing_count: 1,
                room_count: 2,
                drawer_count: 3,
                total_nodes: 10,
                total_edges: 9,
            },
            WorkerResponse::PheromonesDeposited { path_length: 3 },
            WorkerResponse::Error {
                message: "err".into(),
            },
        ];
        for resp in &responses {
            let json = serialize_response(resp).unwrap();
            assert!(!json.is_empty());
            let _: WorkerResponse = deserialize_response(&json).unwrap();
        }
    }

    #[test]
    fn navigate_request_fields() {
        let req = WorkerRequest::Navigate {
            from_id: "node_a".into(),
            to_id: "node_b".into(),
        };
        let json = serialize_request(&req).unwrap();
        let parsed = deserialize_request(&json).unwrap();
        match parsed {
            WorkerRequest::Navigate { from_id, to_id } => {
                assert_eq!(from_id, "node_a");
                assert_eq!(to_id, "node_b");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn deposit_pheromones_request() {
        let req = WorkerRequest::DepositPheromones {
            path: vec!["w1".into(), "r1".into(), "c1".into()],
            base_reward: 2.5,
        };
        let json = serialize_request(&req).unwrap();
        let parsed = deserialize_request(&json).unwrap();
        match parsed {
            WorkerRequest::DepositPheromones { path, base_reward } => {
                assert_eq!(path.len(), 3);
                assert!((base_reward - 2.5).abs() < f64::EPSILON);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn overview_response_fields() {
        let resp = WorkerResponse::Overview {
            name: "My Palace".into(),
            wing_count: 3,
            room_count: 7,
            drawer_count: 42,
            total_nodes: 100,
            total_edges: 99,
        };
        let json = serialize_response(&resp).unwrap();
        let parsed = deserialize_response(&json).unwrap();
        match parsed {
            WorkerResponse::Overview {
                name,
                wing_count,
                total_nodes,
                ..
            } => {
                assert_eq!(name, "My Palace");
                assert_eq!(wing_count, 3);
                assert_eq!(total_nodes, 100);
            }
            _ => panic!("Wrong variant"),
        }
    }
}
