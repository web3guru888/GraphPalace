//! MCP tool schemas for GraphPalace.
//!
//! Each tool is a Rust struct with `serde` derives for JSON serialisation.
//! Tools are grouped into five categories:
//!
//! 1. **Palace Navigation** (read-only) — 8 tools
//! 2. **Palace Operations** (write) — 5 tools
//! 3. **Knowledge Graph** — 6 tools
//! 4. **Stigmergy** — 5 tools
//! 5. **Agent Diary** — 3 tools
//!
//! Plus a [`ToolDefinition`] descriptor and [`tool_catalog`] function
//! that returns metadata for all 28 tools.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 1. Palace Navigation (read) — 8 tools
// ---------------------------------------------------------------------------

/// Return high-level palace status (wings, rooms, drawer counts, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PalaceStatus {}

/// List every wing in the palace.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ListWings {}

/// List rooms within a specific wing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ListRooms {
    pub wing_id: String,
}

/// Retrieve the full palace taxonomy (wings → rooms → closets → drawers).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GetTaxonomy {}

/// Semantic search across all drawers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Search {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wing: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub room: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<usize>,
}

/// Navigate (shortest path) between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Navigate {
    pub from_id: String,
    pub to_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

/// Find cross-wing tunnels (bridge edges) between two wings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FindTunnels {
    pub wing_a: String,
    pub wing_b: String,
}

/// Return aggregate graph statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphStats {}

// ---------------------------------------------------------------------------
// 2. Palace Operations (write) — 5 tools
// ---------------------------------------------------------------------------

/// Store a new drawer (memory item) in a specific wing/room.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AddDrawer {
    pub content: String,
    pub wing: String,
    pub room: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// Delete a drawer by id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeleteDrawer {
    pub drawer_id: String,
}

/// Create a new wing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AddWing {
    pub name: String,
    pub wing_type: String,
    pub description: String,
}

/// Create a new room in an existing wing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AddRoom {
    pub wing_id: String,
    pub name: String,
    pub hall_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Check whether content is a near-duplicate of an existing drawer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckDuplicate {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
}

// ---------------------------------------------------------------------------
// 3. Knowledge Graph — 6 tools
// ---------------------------------------------------------------------------

/// Add a (subject, predicate, object) triple to the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KgAdd {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// RFC 3339 start of validity window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    /// RFC 3339 end of validity window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<String>,
    /// Classification: "fact", "observation", "inference", "hypothesis".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statement_type: Option<String>,
}

/// Query triples for an entity, optionally at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KgQuery {
    pub entity: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub as_of: Option<String>,
}

/// Invalidate (soft-delete) a triple.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KgInvalidate {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Get the temporal history of an entity's triples.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KgTimeline {
    pub entity: String,
}

/// Traverse the knowledge graph from a start node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KgTraverse {
    pub start: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
}

/// Find contradictory triples for an entity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KgContradictions {
    pub entity: String,
}

/// Return knowledge graph aggregate statistics (triple count, entity count, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct KgStats {}

// ---------------------------------------------------------------------------
// 4. Stigmergy — 5 tools
// ---------------------------------------------------------------------------

/// Query pheromone levels on nodes or edges.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PheromoneStatus {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge: Option<String>,
}

/// Deposit pheromone along a traversal path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PheromoneDeposit {
    pub path: Vec<String>,
    pub reward_type: String,
}

/// Return the hottest (most-traversed) paths.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HotPaths {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wing: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<usize>,
}

/// Return cold spots (least-accessed areas) in the palace.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ColdSpots {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wing: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<usize>,
}

/// Trigger an immediate pheromone decay pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DecayNow {}

// ---------------------------------------------------------------------------
// 5. Agent Diary — 3 tools
// ---------------------------------------------------------------------------

/// List all specialist agents.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ListAgents {}

/// Append an entry to an agent's diary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiaryWrite {
    pub agent_id: String,
    pub entry: String,
}

/// Read entries from an agent's diary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiaryRead {
    pub agent_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_n: Option<usize>,
}

// ---------------------------------------------------------------------------
// Tool catalog
// ---------------------------------------------------------------------------

/// Metadata descriptor for a single MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// Tool name (e.g. `"palace_status"`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema of the tool's input parameters (as a `serde_json::Value`).
    pub parameters: serde_json::Value,
}

/// Convenience macro to build a [`ToolDefinition`] from a tool struct's
/// name, description, and list of parameters.
macro_rules! tool_def {
    ($name:expr, $desc:expr, { $( $pname:expr => $ptype:expr $(, optional: $opt:expr)? );* $(;)? }) => {
        {
            #[allow(unused_mut)]
            let mut props = serde_json::Map::new();
            #[allow(unused_mut)]
            let mut required: Vec<String> = Vec::new();
            $(
                let mut prop = serde_json::Map::new();
                prop.insert("type".into(), serde_json::Value::String($ptype.into()));
                props.insert($pname.into(), serde_json::Value::Object(prop));
                // Default is required; if `optional: true` is specified, skip.
                #[allow(unused_mut, unused_assignments)]
                let mut is_optional = false;
                $( is_optional = $opt; )?
                if !is_optional {
                    required.push($pname.into());
                }
            )*
            ToolDefinition {
                name: $name.into(),
                description: $desc.into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": serde_json::Value::Object(props),
                    "required": required,
                }),
            }
        }
    };
}

/// Returns metadata for all 28 GraphPalace MCP tools.
pub fn tool_catalog() -> Vec<ToolDefinition> {
    vec![
        // ── Palace Navigation (read) ────────────────────────────────
        tool_def!("palace_status", "Return high-level palace status (wings, rooms, drawer counts).", {}),
        tool_def!("list_wings", "List every wing in the palace.", {}),
        tool_def!("list_rooms", "List rooms within a specific wing.", {
            "wing_id" => "string"
        }),
        tool_def!("get_taxonomy", "Retrieve the full palace taxonomy.", {}),
        tool_def!("search", "Semantic search across all drawers.", {
            "query"  => "string";
            "wing"   => "string", optional: true;
            "room"   => "string", optional: true;
            "k"      => "integer", optional: true
        }),
        tool_def!("navigate", "Find shortest path between two nodes.", {
            "from_id" => "string";
            "to_id"   => "string";
            "context"  => "string", optional: true
        }),
        tool_def!("find_tunnels", "Find cross-wing tunnels between two wings.", {
            "wing_a" => "string";
            "wing_b" => "string"
        }),
        tool_def!("graph_stats", "Return aggregate graph statistics.", {}),

        // ── Palace Operations (write) ───────────────────────────────
        tool_def!("add_drawer", "Store a new drawer (memory item) in a specific wing/room.", {
            "content" => "string";
            "wing"    => "string";
            "room"    => "string";
            "source"  => "string", optional: true
        }),
        tool_def!("delete_drawer", "Delete a drawer by id.", {
            "drawer_id" => "string"
        }),
        tool_def!("add_wing", "Create a new wing.", {
            "name"        => "string";
            "wing_type"   => "string";
            "description" => "string"
        }),
        tool_def!("add_room", "Create a new room in an existing wing.", {
            "wing_id"     => "string";
            "name"        => "string";
            "hall_type"   => "string";
            "description" => "string", optional: true
        }),
        tool_def!("check_duplicate", "Check whether content is a near-duplicate of an existing drawer.", {
            "content"   => "string";
            "threshold" => "number", optional: true
        }),

        // ── Knowledge Graph ─────────────────────────────────────────
        tool_def!("kg_add", "Add a (subject, predicate, object) triple to the knowledge graph.", {
            "subject"        => "string";
            "predicate"      => "string";
            "object"         => "string";
            "confidence"     => "number", optional: true;
            "valid_from"     => "string", optional: true;
            "valid_to"       => "string", optional: true;
            "statement_type" => "string", optional: true
        }),
        tool_def!("kg_query", "Query triples for an entity, optionally at a point in time.", {
            "entity" => "string";
            "as_of"  => "string", optional: true
        }),
        tool_def!("kg_invalidate", "Invalidate (soft-delete) a triple.", {
            "subject"   => "string";
            "predicate" => "string";
            "object"    => "string"
        }),
        tool_def!("kg_timeline", "Get the temporal history of an entity's triples.", {
            "entity" => "string"
        }),
        tool_def!("kg_traverse", "Traverse the knowledge graph from a start node.", {
            "start"     => "string";
            "depth"     => "integer", optional: true;
            "predicate" => "string", optional: true
        }),
        tool_def!("kg_contradictions", "Find contradictory triples for an entity.", {
            "entity" => "string"
        }),
        tool_def!("kg_stats", "Return knowledge graph aggregate statistics.", {}),

        // ── Stigmergy ───────────────────────────────────────────────
        tool_def!("pheromone_status", "Query pheromone levels on nodes or edges.", {
            "node_id" => "string", optional: true;
            "edge"    => "string", optional: true
        }),
        tool_def!("pheromone_deposit", "Deposit pheromone along a traversal path.", {
            "path"        => "array";
            "reward_type" => "string"
        }),
        tool_def!("hot_paths", "Return the hottest (most-traversed) paths.", {
            "wing" => "string", optional: true;
            "k"    => "integer", optional: true
        }),
        tool_def!("cold_spots", "Return cold spots (least-accessed areas) in the palace.", {
            "wing" => "string", optional: true;
            "k"    => "integer", optional: true
        }),
        tool_def!("decay_now", "Trigger an immediate pheromone decay pass.", {}),

        // ── Agent Diary ─────────────────────────────────────────────
        tool_def!("list_agents", "List all specialist agents.", {}),
        tool_def!("diary_write", "Append an entry to an agent's diary.", {
            "agent_id" => "string";
            "entry"    => "string"
        }),
        tool_def!("diary_read", "Read entries from an agent's diary.", {
            "agent_id" => "string";
            "last_n"   => "integer", optional: true
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_28_tools() {
        let cat = tool_catalog();
        assert_eq!(cat.len(), 28, "Expected 28 tools, got {}", cat.len());
    }

    #[test]
    fn tool_names_unique() {
        let cat = tool_catalog();
        let mut names: Vec<&str> = cat.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 28, "Duplicate tool names found");
    }

    #[test]
    fn tool_definitions_serialize_roundtrip() {
        let cat = tool_catalog();
        for tool in &cat {
            let json = serde_json::to_string(tool)
                .unwrap_or_else(|e| panic!("Failed to serialize tool '{}': {e}", tool.name));
            let back: ToolDefinition = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("Failed to deserialize tool '{}': {e}", tool.name));
            assert_eq!(tool, &back);
        }
    }

    // ── Palace Navigation round-trips ────────────────────────────

    #[test]
    fn palace_status_roundtrip() {
        let t = PalaceStatus {};
        let json = serde_json::to_string(&t).unwrap();
        let back: PalaceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn list_wings_roundtrip() {
        let t = ListWings {};
        let json = serde_json::to_string(&t).unwrap();
        let back: ListWings = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn list_rooms_roundtrip() {
        let t = ListRooms { wing_id: "w1".into() };
        let json = serde_json::to_string(&t).unwrap();
        let back: ListRooms = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn get_taxonomy_roundtrip() {
        let t = GetTaxonomy {};
        let json = serde_json::to_string(&t).unwrap();
        let back: GetTaxonomy = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn search_roundtrip_full() {
        let t = Search {
            query: "quantum gravity".into(),
            wing: Some("physics".into()),
            room: Some("theory".into()),
            k: Some(10),
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("quantum gravity"));
        let back: Search = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn search_roundtrip_minimal() {
        let t = Search { query: "test".into(), wing: None, room: None, k: None };
        let json = serde_json::to_string(&t).unwrap();
        assert!(!json.contains("wing"));
        let back: Search = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn navigate_roundtrip() {
        let t = Navigate {
            from_id: "a".into(),
            to_id: "b".into(),
            context: Some("research".into()),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Navigate = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn find_tunnels_roundtrip() {
        let t = FindTunnels { wing_a: "alpha".into(), wing_b: "beta".into() };
        let json = serde_json::to_string(&t).unwrap();
        let back: FindTunnels = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn graph_stats_roundtrip() {
        let t = GraphStats {};
        let json = serde_json::to_string(&t).unwrap();
        let back: GraphStats = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    // ── Palace Operations round-trips ────────────────────────────

    #[test]
    fn add_drawer_roundtrip() {
        let t = AddDrawer {
            content: "memory".into(),
            wing: "w".into(),
            room: "r".into(),
            source: Some("paper".into()),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: AddDrawer = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn delete_drawer_roundtrip() {
        let t = DeleteDrawer { drawer_id: "d123".into() };
        let json = serde_json::to_string(&t).unwrap();
        let back: DeleteDrawer = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn add_wing_roundtrip() {
        let t = AddWing {
            name: "Physics".into(),
            wing_type: "research".into(),
            description: "All physics-related memories".into(),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: AddWing = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn add_room_roundtrip() {
        let t = AddRoom {
            wing_id: "w1".into(),
            name: "Quantum".into(),
            hall_type: "topic".into(),
            description: None,
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: AddRoom = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn add_room_with_description_roundtrip() {
        let t = AddRoom {
            wing_id: "w1".into(),
            name: "Quantum".into(),
            hall_type: "topic".into(),
            description: Some("Quantum mechanics research".into()),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: AddRoom = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn check_duplicate_roundtrip() {
        let t = CheckDuplicate { content: "test".into(), threshold: Some(0.85) };
        let json = serde_json::to_string(&t).unwrap();
        let back: CheckDuplicate = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    // ── Knowledge Graph round-trips ──────────────────────────────

    #[test]
    fn kg_add_roundtrip() {
        let t = KgAdd {
            subject: "Einstein".into(),
            predicate: "discovered".into(),
            object: "relativity".into(),
            confidence: Some(0.99),
            valid_from: None,
            valid_to: None,
            statement_type: None,
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: KgAdd = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn kg_add_temporal_roundtrip() {
        let t = KgAdd {
            subject: "Earth".into(),
            predicate: "orbits".into(),
            object: "Sun".into(),
            confidence: Some(1.0),
            valid_from: Some("2026-01-01T00:00:00+00:00".into()),
            valid_to: None,
            statement_type: Some("fact".into()),
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("valid_from"));
        assert!(json.contains("statement_type"));
        let back: KgAdd = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn kg_query_roundtrip() {
        let t = KgQuery { entity: "Einstein".into(), as_of: Some("2025-01-01".into()) };
        let json = serde_json::to_string(&t).unwrap();
        let back: KgQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn kg_invalidate_roundtrip() {
        let t = KgInvalidate {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: KgInvalidate = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn kg_timeline_roundtrip() {
        let t = KgTimeline { entity: "Earth".into() };
        let json = serde_json::to_string(&t).unwrap();
        let back: KgTimeline = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn kg_traverse_roundtrip() {
        let t = KgTraverse {
            start: "node1".into(),
            depth: Some(3),
            predicate: Some("causes".into()),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: KgTraverse = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn kg_contradictions_roundtrip() {
        let t = KgContradictions { entity: "Pluto".into() };
        let json = serde_json::to_string(&t).unwrap();
        let back: KgContradictions = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn kg_stats_roundtrip() {
        let t = KgStats {};
        let json = serde_json::to_string(&t).unwrap();
        let back: KgStats = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    // ── Stigmergy round-trips ────────────────────────────────────

    #[test]
    fn pheromone_status_roundtrip() {
        let t = PheromoneStatus { node_id: Some("n1".into()), edge: None };
        let json = serde_json::to_string(&t).unwrap();
        let back: PheromoneStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn pheromone_deposit_roundtrip() {
        let t = PheromoneDeposit {
            path: vec!["a".into(), "b".into(), "c".into()],
            reward_type: "success".into(),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: PheromoneDeposit = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn hot_paths_roundtrip() {
        let t = HotPaths { wing: Some("w1".into()), k: Some(5) };
        let json = serde_json::to_string(&t).unwrap();
        let back: HotPaths = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn cold_spots_roundtrip() {
        let t = ColdSpots { wing: None, k: Some(10) };
        let json = serde_json::to_string(&t).unwrap();
        let back: ColdSpots = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn decay_now_roundtrip() {
        let t = DecayNow {};
        let json = serde_json::to_string(&t).unwrap();
        let back: DecayNow = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    // ── Agent Diary round-trips ──────────────────────────────────

    #[test]
    fn list_agents_roundtrip() {
        let t = ListAgents {};
        let json = serde_json::to_string(&t).unwrap();
        let back: ListAgents = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn diary_write_roundtrip() {
        let t = DiaryWrite { agent_id: "climate-scout".into(), entry: "New finding".into() };
        let json = serde_json::to_string(&t).unwrap();
        let back: DiaryWrite = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn diary_read_roundtrip() {
        let t = DiaryRead { agent_id: "astro-scout".into(), last_n: Some(5) };
        let json = serde_json::to_string(&t).unwrap();
        let back: DiaryRead = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    // ── Catalog content checks ───────────────────────────────────

    #[test]
    fn all_tools_have_non_empty_description() {
        for tool in tool_catalog() {
            assert!(!tool.description.is_empty(), "Tool '{}' has empty description", tool.name);
        }
    }

    #[test]
    fn all_tools_have_object_params() {
        for tool in tool_catalog() {
            let ty = tool.parameters.get("type").and_then(|v| v.as_str());
            assert_eq!(ty, Some("object"), "Tool '{}' params is not an object schema", tool.name);
        }
    }

    #[test]
    fn expected_tool_names_present() {
        let cat = tool_catalog();
        let names: Vec<&str> = cat.iter().map(|t| t.name.as_str()).collect();
        let expected = [
            "palace_status", "list_wings", "list_rooms", "get_taxonomy",
            "search", "navigate", "find_tunnels", "graph_stats",
            "add_drawer", "delete_drawer", "add_wing", "add_room", "check_duplicate",
            "kg_add", "kg_query", "kg_invalidate", "kg_timeline", "kg_traverse", "kg_contradictions", "kg_stats",
            "pheromone_status", "pheromone_deposit", "hot_paths", "cold_spots", "decay_now",
            "list_agents", "diary_write", "diary_read",
        ];
        for name in &expected {
            assert!(names.contains(name), "Missing tool: {name}");
        }
    }
}
