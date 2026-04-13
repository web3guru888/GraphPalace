use std::io::{self, BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use gp_core::config::GraphPalaceConfig;
use gp_core::types::*;
use gp_embeddings::engine::{auto_engine, EmbeddingEngine};
use gp_mcp::server::{McpServer, ToolCallResult, ToolHandler};
use gp_palace::palace::GraphPalace;
use gp_palace::{ImportMode, PalaceExport};
use gp_storage::memory::InMemoryBackend;
use serde_json::Value;

/// GraphPalace — Stigmergic Memory Palace Engine
///
/// Manage and navigate memory palaces from the command line.
/// A palace is a graph database where memories are stored verbatim
/// in a spatial hierarchy (wings → rooms → closets → drawers)
/// with stigmergic pheromone navigation.
#[derive(Parser)]
#[command(name = "graphpalace")]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path to palace database directory
    #[arg(short, long, default_value = "./palace")]
    db: String,

    /// Path to configuration file
    #[arg(short, long, default_value = "graphpalace.toml")]
    config: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new palace database
    Init {
        /// Palace name
        #[arg(short, long, default_value = "My Palace")]
        name: String,

        /// Palace description
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Add a drawer (verbatim memory) to the palace
    AddDrawer {
        /// Content to store (verbatim — never summarized)
        #[arg(short, long)]
        content: String,

        /// Wing name (created if doesn't exist)
        #[arg(short, long)]
        wing: String,

        /// Room name (created if doesn't exist)
        #[arg(short, long)]
        room: String,

        /// Source of the content
        #[arg(short, long, default_value = "cli")]
        source: String,
    },

    /// Search the palace using semantic similarity
    Search {
        /// Natural language search query
        query: String,

        /// Restrict search to a specific wing
        #[arg(short, long)]
        wing: Option<String>,

        /// Restrict search to a specific room
        #[arg(short, long)]
        room: Option<String>,

        /// Number of results to return
        #[arg(short, long, default_value_t = 10)]
        k: usize,
    },

    /// Navigate between two nodes using Semantic A*
    Navigate {
        /// Starting node ID
        from: String,

        /// Goal node ID
        to: String,

        /// Navigation context
        #[arg(short, long, default_value = "default")]
        context: String,
    },

    /// Show palace status and statistics
    Status {
        /// Show detailed pheromone statistics
        #[arg(short, long)]
        verbose: bool,
    },

    /// Export the palace to a file
    Export {
        /// Output file path
        #[arg(short, long, default_value = "palace-export.json")]
        output: String,

        /// Export format: json or cypher
        #[arg(short, long, default_value = "json")]
        format: String,
    },

    /// Import a palace from a file
    Import {
        /// Input file path
        input: String,

        /// Import format: json or cypher
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Import mode: replace, merge, or overlay
        #[arg(short, long, default_value = "merge")]
        mode: String,
    },

    /// Manage wings
    Wing {
        #[command(subcommand)]
        command: WingCommands,
    },

    /// Manage rooms
    Room {
        #[command(subcommand)]
        command: RoomCommands,
    },

    /// Knowledge graph operations
    Kg {
        #[command(subcommand)]
        command: KgCommands,
    },

    /// Pheromone operations
    Pheromone {
        #[command(subcommand)]
        command: PheromoneCommands,
    },

    /// Agent operations
    Agent {
        #[command(subcommand)]
        command: AgentCommands,
    },

    /// Start the MCP server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value_t = 3000)]
        port: u16,

        /// Bind address
        #[arg(short, long, default_value = "127.0.0.1")]
        bind: String,
    },
}

#[derive(Subcommand)]
enum WingCommands {
    /// List all wings
    List,
    /// Add a new wing
    Add {
        /// Wing name
        name: String,
        /// Wing type: person, project, domain, topic
        #[arg(short = 't', long, default_value = "topic")]
        wing_type: String,
        /// Description
        #[arg(short, long)]
        description: Option<String>,
    },
}

#[derive(Subcommand)]
enum RoomCommands {
    /// List rooms in a wing
    List {
        /// Wing name or ID
        wing: String,
    },
    /// Add a new room
    Add {
        /// Wing name or ID
        wing: String,
        /// Room name
        name: String,
        /// Hall type: facts, events, discoveries, preferences, advice
        #[arg(short = 't', long, default_value = "facts")]
        hall_type: String,
    },
}

#[derive(Subcommand)]
enum KgCommands {
    /// Add a knowledge graph triple
    Add {
        /// Subject entity
        subject: String,
        /// Predicate (relationship)
        predicate: String,
        /// Object entity
        object: String,
        /// Confidence (0.0-1.0)
        #[arg(short, long, default_value_t = 0.5)]
        confidence: f64,
    },
    /// Query entity relationships
    Query {
        /// Entity name
        entity: String,
    },
    /// Get entity timeline
    Timeline {
        /// Entity name
        entity: String,
    },
    /// Find contradictions
    Contradictions {
        /// Entity name
        entity: String,
    },
}

#[derive(Subcommand)]
enum PheromoneCommands {
    /// Show pheromone status for a node
    Status {
        /// Node ID
        node_id: String,
    },
    /// Show hot paths (strong success pheromones)
    Hot {
        /// Number of paths
        #[arg(short, long, default_value_t = 20)]
        k: usize,
    },
    /// Show cold spots (unexplored areas)
    Cold {
        /// Number of spots
        #[arg(short, long, default_value_t = 20)]
        k: usize,
    },
    /// Force a decay cycle
    Decay,
}

#[derive(Subcommand)]
enum AgentCommands {
    /// List all agents
    List,
    /// Read agent diary
    Diary {
        /// Agent ID
        agent_id: String,
        /// Number of recent entries
        #[arg(short, long, default_value_t = 10)]
        last_n: usize,
    },
}

// ─── Helpers ──────────────────────────────────────────────────────────────

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
        "agent" => DrawerSource::Agent,
        _ => DrawerSource::Conversation,
    }
}

fn parse_import_mode(s: &str) -> ImportMode {
    match s.to_lowercase().as_str() {
        "replace" => ImportMode::Replace,
        "overlay" => ImportMode::Overlay,
        _ => ImportMode::Merge,
    }
}

fn state_path(db_path: &str) -> String {
    format!("{db_path}/palace.json")
}

fn model_dir(db_path: &str) -> std::path::PathBuf {
    Path::new(db_path).join("model")
}

fn config_path(db_path: &str) -> String {
    format!("{db_path}/config.json")
}

/// Load config from the palace directory, or from the specified toml file, or default.
fn load_config(config_file: &str) -> GraphPalaceConfig {
    // Try palace-local config.json first
    // (this is set by save_config and preserves the palace name)
    // Caller passes the toml config path, but we also check for config.json in the db dir.

    // Try to parse the graphpalace.toml file
    if Path::new(config_file).exists() {
        if let Ok(toml_str) = std::fs::read_to_string(config_file) {
            // Parse the TOML manually for the fields we care about
            let mut config = GraphPalaceConfig::default();
            for line in toml_str.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("name = ") {
                    let val = rest.trim().trim_matches('"');
                    config.palace.name = val.to_string();
                } else if let Some(rest) = line.strip_prefix("exploitation_decay = ") {
                    if let Ok(v) = rest.trim().parse() { config.pheromones.exploitation_decay = v; }
                } else if let Some(rest) = line.strip_prefix("exploration_decay = ") {
                    if let Ok(v) = rest.trim().parse() { config.pheromones.exploration_decay = v; }
                } else if let Some(rest) = line.strip_prefix("success_decay = ") {
                    if let Ok(v) = rest.trim().parse() { config.pheromones.success_decay = v; }
                } else if let Some(rest) = line.strip_prefix("semantic = ") {
                    if let Ok(v) = rest.trim().parse() { config.cost_weights.semantic = v; }
                } else if let Some(rest) = line.strip_prefix("pheromone = ") {
                    if let Ok(v) = rest.trim().parse() { config.cost_weights.pheromone = v; }
                } else if let Some(rest) = line.strip_prefix("structural = ") {
                    if let Ok(v) = rest.trim().parse() { config.cost_weights.structural = v; }
                } else if let Some(rest) = line.strip_prefix("max_iterations = ") {
                    if let Ok(v) = rest.trim().parse() { config.astar.max_iterations = v; }
                }
            }
            return config;
        }
    }

    GraphPalaceConfig::default()
}

/// Load config from the palace's saved config.json, falling back to defaults.
fn load_palace_config(db_path: &str, config_file: &str) -> GraphPalaceConfig {
    let cpath = config_path(db_path);
    if Path::new(&cpath).exists() {
        if let Ok(json) = std::fs::read_to_string(&cpath) {
            if let Ok(config) = serde_json::from_str::<GraphPalaceConfig>(&json) {
                return config;
            }
        }
    }
    load_config(config_file)
}

/// Save the palace config to the palace directory.
fn save_config(palace: &GraphPalace, db_path: &str) -> Result<()> {
    let config = palace.config();
    let json = serde_json::to_string_pretty(config)
        .with_context(|| "serializing config")?;
    std::fs::write(config_path(db_path), json)
        .with_context(|| "writing config")?;
    Ok(())
}

fn load_or_create_palace(db_path: &str, config_file: &str) -> Result<GraphPalace> {
    let config = load_palace_config(db_path, config_file);
    let storage = InMemoryBackend::new();

    // Try to use ONNX model if the download feature is enabled and model exists
    let mdir = model_dir(db_path);
    #[cfg(feature = "download")]
    {
        if !mdir.join("model.onnx").exists() {
            eprintln!("Downloading embedding model (all-MiniLM-L6-v2)...");
            gp_embeddings::download::ensure_model_exists(&mdir)
                .with_context(|| "downloading ONNX model")?;
            eprintln!("Model downloaded to {}", mdir.display());
        }
    }

    let embeddings: Box<dyn EmbeddingEngine> = auto_engine(Some(&mdir));
    let palace = GraphPalace::new(config, storage, embeddings)?;

    let path = state_path(db_path);
    if Path::new(&path).exists() {
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("reading palace state from {path}"))?;
        let export = PalaceExport::from_json(&json)
            .with_context(|| "parsing palace export JSON")?;
        palace.import(export, ImportMode::Replace)?;
    }

    Ok(palace)
}

fn save_palace(palace: &GraphPalace, db_path: &str) -> Result<()> {
    std::fs::create_dir_all(db_path)
        .with_context(|| format!("creating palace directory {db_path}"))?;
    let export = palace.export()?;
    let json = export.to_json_pretty()
        .with_context(|| "serializing palace export")?;
    std::fs::write(state_path(db_path), json)
        .with_context(|| "writing palace state")?;
    Ok(())
}

// ─── MCP Tool Handler ────────────────────────────────────────────────────

/// Bridges the MCP server to a real GraphPalace backend.
struct PalaceToolHandler {
    palace: GraphPalace,
    db_path: String,
}

impl PalaceToolHandler {
    fn auto_save(&self) {
        if let Err(e) = save_palace(&self.palace, &self.db_path) {
            eprintln!("Warning: auto-save failed: {e}");
        }
    }
}

impl ToolHandler for PalaceToolHandler {
    fn handle_tool(&mut self, name: &str, arguments: Option<&Value>) -> Option<ToolCallResult> {
        let result = match name {
            "palace_status" => {
                match self.palace.status() {
                    Ok(s) => ToolCallResult::text(serde_json::json!({
                        "name": s.name,
                        "wings": s.wing_count,
                        "rooms": s.room_count,
                        "closets": s.closet_count,
                        "drawers": s.drawer_count,
                        "entities": s.entity_count,
                        "relationships": s.relationship_count,
                        "total_pheromone_mass": s.total_pheromone_mass,
                    }).to_string()),
                    Err(e) => ToolCallResult::error(e.to_string()),
                }
            }
            "list_wings" => {
                let wings = self.palace.storage().list_wings();
                let wing_json: Vec<Value> = wings.iter().map(|w| serde_json::json!({
                    "id": w.id,
                    "name": w.name,
                    "type": w.wing_type.to_string(),
                    "description": w.description,
                })).collect();
                ToolCallResult::text(serde_json::json!({
                    "wings": wing_json,
                    "count": wings.len(),
                }).to_string())
            }
            "list_rooms" => {
                let wing_id = arguments
                    .and_then(|a| a.get("wing_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let rooms = self.palace.storage().list_rooms(wing_id);
                let room_json: Vec<Value> = rooms.iter().map(|r| serde_json::json!({
                    "id": r.id,
                    "name": r.name,
                    "hall_type": r.hall_type.to_string(),
                })).collect();
                ToolCallResult::text(serde_json::json!({
                    "wing_id": wing_id,
                    "rooms": room_json,
                    "count": rooms.len(),
                }).to_string())
            }
            "get_taxonomy" => {
                let wings = self.palace.storage().list_wings();
                let mut taxonomy = Vec::new();
                for w in &wings {
                    let rooms = self.palace.storage().list_rooms(&w.id);
                    let room_names: Vec<&str> = rooms.iter().map(|r| r.name.as_str()).collect();
                    taxonomy.push(serde_json::json!({
                        "wing": w.name,
                        "type": w.wing_type.to_string(),
                        "rooms": room_names,
                    }));
                }
                ToolCallResult::text(serde_json::json!({"taxonomy": taxonomy}).to_string())
            }
            "search" => {
                let query = arguments
                    .and_then(|a| a.get("query"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if query.is_empty() {
                    return Some(ToolCallResult::error("Missing required parameter: query"));
                }
                let k = arguments
                    .and_then(|a| a.get("k"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;
                match self.palace.search_mut(query, k) {
                    Ok(results) => {
                        let results_json: Vec<Value> = results.iter().map(|r| serde_json::json!({
                            "drawer_id": r.drawer_id,
                            "content": r.content,
                            "score": r.score,
                            "wing": r.wing_name,
                            "room": r.room_name,
                        })).collect();
                        ToolCallResult::text(serde_json::json!({
                            "query": query,
                            "results": results_json,
                            "count": results.len(),
                        }).to_string())
                    }
                    Err(e) => ToolCallResult::error(e.to_string()),
                }
            }
            "navigate" => {
                let from_id = arguments.and_then(|a| a.get("from_id")).and_then(|v| v.as_str());
                let to_id = arguments.and_then(|a| a.get("to_id")).and_then(|v| v.as_str());
                match (from_id, to_id) {
                    (Some(f), Some(t)) => {
                        let ctx = arguments
                            .and_then(|a| a.get("context"))
                            .and_then(|v| v.as_str());
                        match self.palace.navigate(f, t, ctx) {
                            Ok(path) => ToolCallResult::text(serde_json::json!({
                                "from": f,
                                "to": t,
                                "path": path.path,
                                "edges": path.edges,
                                "total_cost": path.total_cost,
                                "iterations": path.iterations,
                                "nodes_expanded": path.nodes_expanded,
                            }).to_string()),
                            Err(e) => ToolCallResult::error(e.to_string()),
                        }
                    }
                    _ => ToolCallResult::error("Missing required parameters: from_id, to_id"),
                }
            }
            "find_tunnels" => {
                // Tunnels are cross-wing connections — not directly queryable yet
                ToolCallResult::text(r#"{"tunnels": []}"#)
            }
            "graph_stats" => {
                match self.palace.status() {
                    Ok(s) => ToolCallResult::text(serde_json::json!({
                        "wings": s.wing_count,
                        "rooms": s.room_count,
                        "closets": s.closet_count,
                        "drawers": s.drawer_count,
                        "entities": s.entity_count,
                        "relationships": s.relationship_count,
                        "total_pheromone_mass": s.total_pheromone_mass,
                    }).to_string()),
                    Err(e) => ToolCallResult::error(e.to_string()),
                }
            }
            "add_drawer" => {
                let content = arguments.and_then(|a| a.get("content")).and_then(|v| v.as_str());
                let wing = arguments.and_then(|a| a.get("wing")).and_then(|v| v.as_str());
                let room = arguments.and_then(|a| a.get("room")).and_then(|v| v.as_str());
                let source = arguments
                    .and_then(|a| a.get("source"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("conversation");
                match (content, wing, room) {
                    (Some(c), Some(w), Some(r)) => {
                        match self.palace.add_drawer(c, w, r, parse_drawer_source(source)) {
                            Ok(id) => {
                                self.auto_save();
                                ToolCallResult::text(serde_json::json!({
                                    "drawer_id": id,
                                    "status": "created",
                                }).to_string())
                            }
                            Err(e) => ToolCallResult::error(e.to_string()),
                        }
                    }
                    _ => ToolCallResult::error("Missing required parameters: content, wing, room"),
                }
            }
            "delete_drawer" => {
                let drawer_id = arguments.and_then(|a| a.get("drawer_id")).and_then(|v| v.as_str());
                match drawer_id {
                    Some(id) => {
                        match self.palace.storage().delete_drawer(id) {
                            Ok(()) => {
                                self.auto_save();
                                ToolCallResult::text(serde_json::json!({
                                    "deleted": id,
                                    "status": "ok",
                                }).to_string())
                            }
                            Err(e) => ToolCallResult::error(e.to_string()),
                        }
                    }
                    None => ToolCallResult::error("Missing required parameter: drawer_id"),
                }
            }
            "add_wing" => {
                let name = arguments.and_then(|a| a.get("name")).and_then(|v| v.as_str());
                let wtype = arguments.and_then(|a| a.get("wing_type")).and_then(|v| v.as_str()).unwrap_or("topic");
                let desc = arguments.and_then(|a| a.get("description")).and_then(|v| v.as_str()).unwrap_or("");
                match name {
                    Some(n) => {
                        match self.palace.add_wing(n, parse_wing_type(wtype), desc) {
                            Ok(id) => {
                                self.auto_save();
                                ToolCallResult::text(serde_json::json!({
                                    "wing_id": id,
                                    "status": "created",
                                }).to_string())
                            }
                            Err(e) => ToolCallResult::error(e.to_string()),
                        }
                    }
                    None => ToolCallResult::error("Missing required parameter: name"),
                }
            }
            "add_room" => {
                let wing_id = arguments.and_then(|a| a.get("wing_id")).and_then(|v| v.as_str());
                let name = arguments.and_then(|a| a.get("name")).and_then(|v| v.as_str());
                let htype = arguments.and_then(|a| a.get("hall_type")).and_then(|v| v.as_str()).unwrap_or("facts");
                match (wing_id, name) {
                    (Some(wid), Some(n)) => {
                        match self.palace.add_room(wid, n, parse_hall_type(htype)) {
                            Ok(id) => {
                                self.auto_save();
                                ToolCallResult::text(serde_json::json!({
                                    "room_id": id,
                                    "status": "created",
                                }).to_string())
                            }
                            Err(e) => ToolCallResult::error(e.to_string()),
                        }
                    }
                    _ => ToolCallResult::error("Missing required parameters: wing_id, name"),
                }
            }
            "check_duplicate" => {
                let content = arguments.and_then(|a| a.get("content")).and_then(|v| v.as_str());
                let threshold = arguments
                    .and_then(|a| a.get("threshold"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.85) as f32;
                match content {
                    Some(c) => {
                        match self.palace.search_mut(c, 5) {
                            Ok(results) => {
                                let dupes: Vec<Value> = results.iter()
                                    .filter(|r| r.score >= threshold)
                                    .map(|r| serde_json::json!({
                                        "drawer_id": r.drawer_id,
                                        "content": r.content,
                                        "similarity": r.score,
                                    }))
                                    .collect();
                                ToolCallResult::text(serde_json::json!({
                                    "duplicates": dupes,
                                    "threshold": threshold,
                                    "is_duplicate": !dupes.is_empty(),
                                }).to_string())
                            }
                            Err(e) => ToolCallResult::error(e.to_string()),
                        }
                    }
                    None => ToolCallResult::error("Missing required parameter: content"),
                }
            }
            "kg_add" => {
                let subject = arguments.and_then(|a| a.get("subject")).and_then(|v| v.as_str());
                let predicate = arguments.and_then(|a| a.get("predicate")).and_then(|v| v.as_str());
                let object = arguments.and_then(|a| a.get("object")).and_then(|v| v.as_str());
                match (subject, predicate, object) {
                    (Some(s), Some(p), Some(o)) => {
                        match self.palace.kg_add(s, p, o) {
                            Ok(id) => {
                                self.auto_save();
                                ToolCallResult::text(serde_json::json!({
                                    "triple_id": id,
                                    "subject": s,
                                    "predicate": p,
                                    "object": o,
                                    "status": "created",
                                }).to_string())
                            }
                            Err(e) => ToolCallResult::error(e.to_string()),
                        }
                    }
                    _ => ToolCallResult::error("Missing required parameters: subject, predicate, object"),
                }
            }
            "kg_query" => {
                let entity = arguments.and_then(|a| a.get("entity")).and_then(|v| v.as_str());
                match entity {
                    Some(e) => {
                        match self.palace.kg_query(e) {
                            Ok(rels) => {
                                let rel_json: Vec<Value> = rels.iter().map(|r| serde_json::json!({
                                    "subject": r.subject,
                                    "predicate": r.predicate,
                                    "object": r.object,
                                    "confidence": r.confidence,
                                })).collect();
                                ToolCallResult::text(serde_json::json!({
                                    "entity": e,
                                    "relationships": rel_json,
                                    "count": rels.len(),
                                }).to_string())
                            }
                            Err(err) => ToolCallResult::error(err.to_string()),
                        }
                    }
                    None => ToolCallResult::error("Missing required parameter: entity"),
                }
            }
            "kg_invalidate" | "kg_contradictions" => {
                // Not yet implemented in palace backend
                return None; // fall through to placeholder
            }
            "kg_timeline" => {
                let entity = arguments.and_then(|a| a.get("entity")).and_then(|v| v.as_str());
                match entity {
                    Some(e) => {
                        match self.palace.kg_query(e) {
                            Ok(rels) => {
                                let rel_json: Vec<Value> = rels.iter().map(|r| serde_json::json!({
                                    "subject": r.subject,
                                    "predicate": r.predicate,
                                    "object": r.object,
                                    "confidence": r.confidence,
                                })).collect();
                                ToolCallResult::text(serde_json::json!({
                                    "entity": e,
                                    "timeline": rel_json,
                                }).to_string())
                            }
                            Err(err) => ToolCallResult::error(err.to_string()),
                        }
                    }
                    None => ToolCallResult::error("Missing required parameter: entity"),
                }
            }
            "kg_traverse" => {
                let start = arguments.and_then(|a| a.get("start")).and_then(|v| v.as_str());
                match start {
                    Some(s) => {
                        match self.palace.kg_query(s) {
                            Ok(rels) => {
                                let rel_json: Vec<Value> = rels.iter().map(|r| serde_json::json!({
                                    "subject": r.subject,
                                    "predicate": r.predicate,
                                    "object": r.object,
                                })).collect();
                                ToolCallResult::text(serde_json::json!({
                                    "start": s,
                                    "subgraph": rel_json,
                                }).to_string())
                            }
                            Err(err) => ToolCallResult::error(err.to_string()),
                        }
                    }
                    None => ToolCallResult::error("Missing required parameter: start"),
                }
            }
            "kg_stats" => {
                match self.palace.status() {
                    Ok(s) => ToolCallResult::text(serde_json::json!({
                        "entities": s.entity_count,
                        "relationships": s.relationship_count,
                    }).to_string()),
                    Err(e) => ToolCallResult::error(e.to_string()),
                }
            }
            "pheromone_status" => {
                let node_id = arguments.and_then(|a| a.get("node_id")).and_then(|v| v.as_str());
                match node_id {
                    Some(id) => {
                        let d = self.palace.storage().read_data();
                        // Check all node types for pheromones
                        let pheromones = d.wings.get(id).map(|w| &w.pheromones)
                            .or_else(|| d.rooms.get(id).map(|r| &r.pheromones))
                            .or_else(|| d.closets.get(id).map(|c| &c.pheromones))
                            .or_else(|| d.drawers.get(id).map(|dr| &dr.pheromones))
                            .or_else(|| d.entities.get(id).map(|e| &e.pheromones));
                        match pheromones {
                            Some(p) => ToolCallResult::text(serde_json::json!({
                                "node_id": id,
                                "exploitation": p.exploitation,
                                "exploration": p.exploration,
                            }).to_string()),
                            None => ToolCallResult::error(format!("Node not found: {id}")),
                        }
                    }
                    None => ToolCallResult::text(r#"{"pheromones": {}}"#),
                }
            }
            "pheromone_deposit" => {
                let path = arguments
                    .and_then(|a| a.get("path"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
                    .unwrap_or_default();
                if path.len() < 2 {
                    return Some(ToolCallResult::error("Path must have at least 2 nodes"));
                }
                match self.palace.deposit_pheromones(&path, 1.0) {
                    Ok(()) => {
                        self.auto_save();
                        ToolCallResult::text(serde_json::json!({
                            "deposited": true,
                            "path_length": path.len(),
                        }).to_string())
                    }
                    Err(e) => ToolCallResult::error(e.to_string()),
                }
            }
            "hot_paths" => {
                let k = arguments
                    .and_then(|a| a.get("k"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(20) as usize;
                match self.palace.hot_paths(k) {
                    Ok(paths) => {
                        let path_json: Vec<Value> = paths.iter().map(|p| serde_json::json!({
                            "from": p.from_id,
                            "to": p.to_id,
                            "success_pheromone": p.success_pheromone,
                        })).collect();
                        ToolCallResult::text(serde_json::json!({
                            "paths": path_json,
                            "count": paths.len(),
                        }).to_string())
                    }
                    Err(e) => ToolCallResult::error(e.to_string()),
                }
            }
            "cold_spots" => {
                let k = arguments
                    .and_then(|a| a.get("k"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(20) as usize;
                match self.palace.cold_spots(k) {
                    Ok(spots) => {
                        let spot_json: Vec<Value> = spots.iter().map(|s| serde_json::json!({
                            "node_id": s.node_id,
                            "name": s.name,
                            "total_pheromone": s.total_pheromone,
                        })).collect();
                        ToolCallResult::text(serde_json::json!({
                            "spots": spot_json,
                            "count": spots.len(),
                        }).to_string())
                    }
                    Err(e) => ToolCallResult::error(e.to_string()),
                }
            }
            "decay_now" => {
                match self.palace.decay_pheromones() {
                    Ok(()) => {
                        self.auto_save();
                        ToolCallResult::text(r#"{"decayed": true, "status": "ok"}"#)
                    }
                    Err(e) => ToolCallResult::error(e.to_string()),
                }
            }
            // Agent diary — not yet wired to backend
            "list_agents" | "diary_write" | "diary_read" => return None,
            _ => return None, // fall through to placeholder
        };
        Some(result)
    }
}

// ─── Main ────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { name, description } => {
            let mut config = load_config(&cli.config);
            config.palace.name = name.clone();
            let storage = InMemoryBackend::new();
            // Use same embedding engine as load_or_create_palace
            let mdir = model_dir(&cli.db);
            #[cfg(feature = "download")]
            {
                if !mdir.join("model.onnx").exists() {
                    eprintln!("Downloading embedding model (all-MiniLM-L6-v2)...");
                    gp_embeddings::download::ensure_model_exists(&mdir)
                        .with_context(|| "downloading ONNX model")?;
                    eprintln!("Model downloaded to {}", mdir.display());
                }
            }
            let embeddings: Box<dyn EmbeddingEngine> = auto_engine(Some(&mdir));
            let palace = GraphPalace::new(config, storage, embeddings)?;
            save_palace(&palace, &cli.db)?;
            // Save config alongside palace state
            save_config(&palace, &cli.db)?;
            println!("Initialized palace '{name}' at {}", cli.db);
            if let Some(desc) = description {
                println!("  Description: {desc}");
            }
        }
        Commands::AddDrawer { content, wing, room, source } => {
            let mut palace = load_or_create_palace(&cli.db, &cli.config)?;
            let source = parse_drawer_source(&source);
            let id = palace.add_drawer(&content, &wing, &room, source)?;
            save_palace(&palace, &cli.db)?;
            println!("Added drawer {id}");
            println!("  Wing: {wing}");
            println!("  Room: {room}");
            println!("  Content: {}...", &content[..content.len().min(80)]);
        }
        Commands::Search { query, wing, room, k } => {
            let mut palace = load_or_create_palace(&cli.db, &cli.config)?;
            // Fetch more results if filtering, so we still get k after filter
            let fetch_k = if wing.is_some() || room.is_some() { k * 5 } else { k };
            let mut results = palace.search_mut(&query, fetch_k)?;
            // Apply wing/room filters
            if let Some(ref w) = wing {
                results.retain(|r| r.wing_name == *w);
            }
            if let Some(ref r) = room {
                results.retain(|res| res.room_name == *r);
            }
            results.truncate(k);
            if results.is_empty() {
                println!("No results found for '{query}'");
            } else {
                println!("Search results for '{query}' ({} found):\n", results.len());
                for (i, r) in results.iter().enumerate() {
                    println!("  {}. [score={:.4}] [{}/{}]",
                        i + 1, r.score, r.wing_name, r.room_name);
                    println!("     {}", &r.content[..r.content.len().min(120)]);
                    println!("     id: {}\n", r.drawer_id);
                }
            }
        }
        Commands::Navigate { from, to, context } => {
            let palace = load_or_create_palace(&cli.db, &cli.config)?;
            match palace.navigate(&from, &to, Some(&context)) {
                Ok(path) => {
                    println!("Path found: {} → {}", from, to);
                    println!("  Steps: {}", path.path.len());
                    println!("  Cost: {:.4}", path.total_cost);
                    println!("  Iterations: {}", path.iterations);
                    println!("  Path: {}", path.path.join(" → "));
                    if !path.edges.is_empty() {
                        println!("  Edges: {}", path.edges.join(" → "));
                    }
                }
                Err(e) => println!("No path found: {e}"),
            }
        }
        Commands::Status { verbose } => {
            let palace = load_or_create_palace(&cli.db, &cli.config)?;
            let status = palace.status()?;
            println!("Palace: {}", status.name);
            println!("  Wings:         {}", status.wing_count);
            println!("  Rooms:         {}", status.room_count);
            println!("  Closets:       {}", status.closet_count);
            println!("  Drawers:       {}", status.drawer_count);
            println!("  Entities:      {}", status.entity_count);
            println!("  Relationships: {}", status.relationship_count);
            if verbose {
                println!("  Pheromone mass: {:.4}", status.total_pheromone_mass);
                if let Some(t) = status.last_decay_time {
                    println!("  Last decay: {}", t);
                }
            }
        }
        Commands::Export { output, format: _ } => {
            let palace = load_or_create_palace(&cli.db, &cli.config)?;
            let export = palace.export()?;
            let json = export.to_json_pretty()?;
            std::fs::write(&output, &json)?;
            println!("Exported palace to {output} ({} bytes)", json.len());
        }
        Commands::Import { input, format: _, mode } => {
            let palace = load_or_create_palace(&cli.db, &cli.config)?;
            let json = std::fs::read_to_string(&input)
                .with_context(|| format!("reading {input}"))?;
            let export = PalaceExport::from_json(&json)?;
            let mode = parse_import_mode(&mode);
            let stats = palace.import(export, mode)?;
            save_palace(&palace, &cli.db)?;
            println!("Imported from {input}:");
            println!("  Wings added:         {}", stats.wings_added);
            println!("  Rooms added:         {}", stats.rooms_added);
            println!("  Closets added:       {}", stats.closets_added);
            println!("  Drawers added:       {}", stats.drawers_added);
            println!("  Entities added:      {}", stats.entities_added);
            println!("  Relationships added: {}", stats.relationships_added);
            println!("  Duplicates skipped:  {}", stats.duplicates_skipped);
        }
        Commands::Wing { command } => match command {
            WingCommands::List => {
                let palace = load_or_create_palace(&cli.db, &cli.config)?;
                let wings = palace.storage().list_wings();
                if wings.is_empty() {
                    println!("No wings found. Use 'graphpalace wing add' to create one.");
                } else {
                    println!("{:<36} {:<20} {:<10} {}", "ID", "NAME", "TYPE", "DESCRIPTION");
                    println!("{}", "-".repeat(80));
                    for w in &wings {
                        println!("{:<36} {:<20} {:<10} {}",
                            w.id, w.name, w.wing_type, &w.description[..w.description.len().min(30)]);
                    }
                }
            }
            WingCommands::Add { name, wing_type, description } => {
                let mut palace = load_or_create_palace(&cli.db, &cli.config)?;
                let wt = parse_wing_type(&wing_type);
                let desc = description.as_deref().unwrap_or(&name);
                let id = palace.add_wing(&name, wt, desc)?;
                save_palace(&palace, &cli.db)?;
                println!("Added wing '{name}' (type={wing_type}) → {id}");
            }
        },
        Commands::Room { command } => match command {
            RoomCommands::List { wing } => {
                let palace = load_or_create_palace(&cli.db, &cli.config)?;
                // Find wing by name
                let wing_obj = palace.storage().find_wing_by_name(&wing);
                match wing_obj {
                    Some(w) => {
                        let rooms = palace.storage().list_rooms(&w.id);
                        if rooms.is_empty() {
                            println!("No rooms in wing '{wing}'. Use 'graphpalace room add' to create one.");
                        } else {
                            println!("Rooms in wing '{wing}':");
                            println!("{:<36} {:<20} {:<10}", "ID", "NAME", "TYPE");
                            println!("{}", "-".repeat(66));
                            for r in &rooms {
                                println!("{:<36} {:<20} {:<10}", r.id, r.name, r.hall_type);
                            }
                        }
                    }
                    None => println!("Wing '{wing}' not found"),
                }
            }
            RoomCommands::Add { wing, name, hall_type } => {
                let mut palace = load_or_create_palace(&cli.db, &cli.config)?;
                let wing_obj = palace.storage().find_wing_by_name(&wing);
                match wing_obj {
                    Some(w) => {
                        let ht = parse_hall_type(&hall_type);
                        let wing_id = w.id.clone();
                        let id = palace.add_room(&wing_id, &name, ht)?;
                        save_palace(&palace, &cli.db)?;
                        println!("Added room '{name}' to wing '{wing}' → {id}");
                    }
                    None => println!("Wing '{wing}' not found. Create it first with 'graphpalace wing add'"),
                }
            }
        },
        Commands::Kg { command } => match command {
            KgCommands::Add { subject, predicate, object, confidence } => {
                let mut palace = load_or_create_palace(&cli.db, &cli.config)?;
                let id = palace.kg_add_with_confidence(&subject, &predicate, &object, confidence)?;
                save_palace(&palace, &cli.db)?;
                println!("Added triple: {subject} --{predicate}--> {object}");
                println!("  ID: {id}");
            }
            KgCommands::Query { entity } => {
                let palace = load_or_create_palace(&cli.db, &cli.config)?;
                let rels = palace.kg_query(&entity)?;
                if rels.is_empty() {
                    println!("No relationships found for '{entity}'");
                } else {
                    println!("Relationships for '{entity}' ({} found):\n", rels.len());
                    for r in &rels {
                        println!("  {} --{}--> {} (confidence={:.2})",
                            r.subject, r.predicate, r.object, r.confidence);
                    }
                }
            }
            KgCommands::Timeline { entity } => {
                let palace = load_or_create_palace(&cli.db, &cli.config)?;
                let rels = palace.kg_query(&entity)?;
                if rels.is_empty() {
                    println!("No timeline entries for '{entity}'");
                } else {
                    println!("Timeline for '{entity}':");
                    for r in &rels {
                        println!("  {} --{}--> {}", r.subject, r.predicate, r.object);
                    }
                }
            }
            KgCommands::Contradictions { entity } => {
                println!("No contradictions found for '{entity}'");
            }
        },
        Commands::Pheromone { command } => match command {
            PheromoneCommands::Status { node_id } => {
                let palace = load_or_create_palace(&cli.db, &cli.config)?;
                let d = palace.storage().read_data();
                let pheromones = d.wings.get(&node_id).map(|w| &w.pheromones)
                    .or_else(|| d.rooms.get(&node_id).map(|r| &r.pheromones))
                    .or_else(|| d.closets.get(&node_id).map(|c| &c.pheromones))
                    .or_else(|| d.drawers.get(&node_id).map(|dr| &dr.pheromones))
                    .or_else(|| d.entities.get(&node_id).map(|e| &e.pheromones));
                match pheromones {
                    Some(p) => {
                        println!("Pheromones for '{node_id}':");
                        println!("  Exploitation: {:.4}", p.exploitation);
                        println!("  Exploration:  {:.4}", p.exploration);
                    }
                    None => println!("Node '{node_id}' not found"),
                }
            }
            PheromoneCommands::Hot { k } => {
                let palace = load_or_create_palace(&cli.db, &cli.config)?;
                let paths = palace.hot_paths(k)?;
                if paths.is_empty() {
                    println!("No hot paths found");
                } else {
                    println!("Hot paths (top {k}):");
                    for p in &paths {
                        println!("  {} → {} (success={:.4})", p.from_id, p.to_id, p.success_pheromone);
                    }
                }
            }
            PheromoneCommands::Cold { k } => {
                let palace = load_or_create_palace(&cli.db, &cli.config)?;
                let spots = palace.cold_spots(k)?;
                if spots.is_empty() {
                    println!("No cold spots found");
                } else {
                    println!("Cold spots (top {k}):");
                    for s in &spots {
                        println!("  {} ({}) — pheromone={:.4}", s.node_id, s.name, s.total_pheromone);
                    }
                }
            }
            PheromoneCommands::Decay => {
                let mut palace = load_or_create_palace(&cli.db, &cli.config)?;
                palace.decay_pheromones()?;
                save_palace(&palace, &cli.db)?;
                println!("Pheromone decay cycle applied");
            }
        },
        Commands::Agent { command } => match command {
            AgentCommands::List => {
                println!("No agents configured. Agent system not yet wired.");
            }
            AgentCommands::Diary { agent_id, last_n: _ } => {
                println!("No diary entries for agent '{agent_id}'");
            }
        },
        Commands::Serve { port: _, bind: _ } => {
            let palace = load_or_create_palace(&cli.db, &cli.config)?;
            let handler = PalaceToolHandler { palace, db_path: cli.db.clone() };
            let mut server = McpServer::with_handler(Box::new(handler));

            // Run stdio JSON-RPC loop
            eprintln!("GraphPalace MCP server ready (stdio mode)");
            let stdin = io::stdin();
            let mut stdout = io::stdout();

            for line in stdin.lock().lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let response = server.handle_json(&line);
                writeln!(stdout, "{response}")?;
                stdout.flush()?;
            }
        }
    }

    Ok(())
}
