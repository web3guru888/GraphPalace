use std::path::Path;

use anyhow::Context;
use clap::{Parser, Subcommand};

use gp_core::{DrawerSource, GraphPalaceConfig, HallType, WingType};
use gp_embeddings::{EmbeddingEngine, TfIdfEmbeddingEngine};
use gp_palace::{GraphPalace, ImportMode, PalaceExport};
use gp_storage::memory::PalaceData;
use gp_storage::InMemoryBackend;

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
    #[command(name = "add-drawer")]
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

        /// Source of the content: conversation, file, api, agent, cli
        #[arg(short, long, default_value = "cli")]
        source: String,
    },

    /// Search the palace using semantic similarity
    Search {
        /// Natural language search query
        query: String,

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
        #[arg(short, long)]
        context: Option<String>,
    },

    /// Show palace status and statistics
    Status,

    /// Export the palace to a file
    Export {
        /// Output file path
        #[arg(short, long, default_value = "palace-export.json")]
        output: String,
    },

    /// Import a palace from a file
    Import {
        /// Input file path
        input: String,

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
        /// Wing name
        wing: String,
    },
    /// Add a new room
    Add {
        /// Wing name
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
    /// List all agent archetypes
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_palace(db_path: &str) -> anyhow::Result<GraphPalace> {
    let json_path = Path::new(db_path).join("palace.json");
    let config = GraphPalaceConfig::default();
    if json_path.exists() {
        let json = std::fs::read_to_string(&json_path)
            .with_context(|| format!("Failed to read {}", json_path.display()))?;
        let mut data: PalaceData = serde_json::from_str(&json)
            .with_context(|| "Failed to parse palace.json")?;

        // Rebuild TF-IDF vocabulary from existing drawer content and
        // re-embed all nodes.  The stable term-hash projection ensures
        // embeddings are identical regardless of insertion order, so
        // palace save/load cycles are perfectly consistent.
        let corpus: Vec<&str> = data.drawers.values().map(|d| d.content.as_str()).collect();
        let mut engine = if corpus.is_empty() {
            TfIdfEmbeddingEngine::new()
        } else {
            let mut e = TfIdfEmbeddingEngine::from_corpus(&corpus);
            e.unfreeze();
            e
        };

        // Re-embed drawers, wings, rooms, closets, entities with
        // the rebuilt vocabulary so stored vectors match query vectors.
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
        let embeddings: Box<dyn gp_embeddings::EmbeddingEngine> = Box::new(engine);
        GraphPalace::new(config, backend, embeddings).map_err(Into::into)
    } else {
        anyhow::bail!(
            "No palace found at '{}'. Use `graphpalace init` first.",
            db_path
        )
    }
}

fn save_palace(palace: &GraphPalace, db_path: &str) -> anyhow::Result<()> {
    let json_path = Path::new(db_path).join("palace.json");
    let data = palace.storage().snapshot();
    let json = serde_json::to_string_pretty(&data)?;
    std::fs::write(&json_path, json)
        .with_context(|| format!("Failed to write {}", json_path.display()))?;
    Ok(())
}

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
        _ => DrawerSource::Agent, // "agent", "cli", or anything else
    }
}

fn parse_import_mode(s: &str) -> anyhow::Result<ImportMode> {
    match s.to_lowercase().as_str() {
        "replace" => Ok(ImportMode::Replace),
        "merge" => Ok(ImportMode::Merge),
        "overlay" => Ok(ImportMode::Overlay),
        other => anyhow::bail!(
            "Unknown import mode '{}'. Use: replace, merge, overlay",
            other
        ),
    }
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

fn cmd_init(db_path: &str, name: &str, description: Option<&str>) -> anyhow::Result<()> {
    let dir = Path::new(db_path);
    let json_path = dir.join("palace.json");

    if json_path.exists() {
        anyhow::bail!("Palace already exists at '{}'", db_path);
    }

    std::fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create directory '{}'", db_path))?;

    let config = GraphPalaceConfig::default();
    let embeddings: Box<dyn gp_embeddings::EmbeddingEngine> =
        Box::new(TfIdfEmbeddingEngine::new());
    let backend = InMemoryBackend::new();
    let palace = GraphPalace::new(config, backend, embeddings)?;

    save_palace(&palace, db_path)?;

    println!("✓ Palace '{}' initialized at {}", name, db_path);
    if let Some(desc) = description {
        println!("  Description: {}", desc);
    }
    println!("  Storage: {}/palace.json", db_path);
    Ok(())
}

fn cmd_add_drawer(
    db_path: &str,
    content: &str,
    wing: &str,
    room: &str,
    source: &str,
) -> anyhow::Result<()> {
    let mut palace = load_palace(db_path)?;
    let drawer_source = parse_drawer_source(source);
    let drawer_id = palace.add_drawer(content, wing, room, drawer_source)?;
    save_palace(&palace, db_path)?;

    println!("✓ Drawer added: {}", drawer_id);
    println!("  Wing: {}", wing);
    println!("  Room: {}", room);
    println!(
        "  Content: {}{}",
        &content[..content.len().min(80)],
        if content.len() > 80 { "…" } else { "" }
    );
    Ok(())
}

fn cmd_search(db_path: &str, query: &str, k: usize) -> anyhow::Result<()> {
    let mut palace = load_palace(db_path)?;
    let results = palace.search_mut(query, k)?;

    if results.is_empty() {
        println!("No results found for '{}'", query);
        return Ok(());
    }

    println!(
        "Search results for '{}' (top {}):\n",
        query,
        results.len()
    );
    println!(
        "  {:<4} {:<10} {:<18} {:<18} Content",
        "#", "Score", "Wing", "Room"
    );
    println!("  {}", "─".repeat(78));

    for (i, r) in results.iter().enumerate() {
        let snippet: String = r
            .content
            .chars()
            .take(40)
            .collect::<String>()
            .replace('\n', " ");
        println!(
            "  {:<4} {:<10.4} {:<18} {:<18} {}{}",
            i + 1,
            r.score,
            truncate(&r.wing_name, 16),
            truncate(&r.room_name, 16),
            snippet,
            if r.content.len() > 40 { "…" } else { "" }
        );
    }
    println!();
    Ok(())
}

fn cmd_navigate(
    db_path: &str,
    from: &str,
    to: &str,
    context: Option<&str>,
) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let result = palace.navigate(from, to, context)?;

    println!("Path found ({} steps, cost {:.4}):\n", result.path.len(), result.total_cost);

    for (i, step) in result.provenance.iter().enumerate() {
        let connector = if i == result.provenance.len() - 1 {
            "└──"
        } else {
            "├──"
        };
        let edge_label = if step.edge_type.is_empty() {
            "START".to_string()
        } else {
            step.edge_type.clone()
        };
        println!(
            "  {} [{}] {} (g={:.3} h={:.3} f={:.3})",
            connector, edge_label, step.node_id, step.g_cost, step.h_cost, step.f_cost
        );
    }
    println!(
        "\n  Iterations: {} | Nodes expanded: {}",
        result.iterations, result.nodes_expanded
    );
    Ok(())
}

fn cmd_status(db_path: &str) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let s = palace.status()?;

    println!("Palace: {}\n", s.name);
    println!("  Structure");
    println!("  ├── Wings:         {}", s.wing_count);
    println!("  ├── Rooms:         {}", s.room_count);
    println!("  ├── Closets:       {}", s.closet_count);
    println!("  └── Drawers:       {}", s.drawer_count);
    println!();
    println!("  Knowledge Graph");
    println!("  ├── Entities:      {}", s.entity_count);
    println!("  └── Relationships: {}", s.relationship_count);
    println!();
    println!("  Stigmergy");
    println!(
        "  ├── Pheromone mass: {:.4}",
        s.total_pheromone_mass
    );
    println!(
        "  └── Last decay:     {}",
        s.last_decay_time
            .map(|t| t.to_rfc3339())
            .unwrap_or_else(|| "never".into())
    );
    Ok(())
}

fn cmd_export(db_path: &str, output: &str) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let export = palace.export()?;
    let json = export.to_json_pretty().context("Failed to serialize export")?;
    std::fs::write(output, &json)
        .with_context(|| format!("Failed to write '{}'", output))?;
    println!("✓ Palace exported to {} ({} bytes)", output, json.len());
    Ok(())
}

fn cmd_import(db_path: &str, input: &str, mode_str: &str) -> anyhow::Result<()> {
    let mode = parse_import_mode(mode_str)?;
    let json = std::fs::read_to_string(input)
        .with_context(|| format!("Failed to read '{}'", input))?;
    let export = PalaceExport::from_json(&json).context("Failed to parse export file")?;

    let palace = load_palace(db_path)?;
    let stats = palace.import(export, mode)?;
    save_palace(&palace, db_path)?;

    println!("✓ Import complete (mode: {})", mode_str);
    println!("  Wings added:         {}", stats.wings_added);
    println!("  Rooms added:         {}", stats.rooms_added);
    println!("  Closets added:       {}", stats.closets_added);
    println!("  Drawers added:       {}", stats.drawers_added);
    println!("  Entities added:      {}", stats.entities_added);
    println!("  Relationships added: {}", stats.relationships_added);
    println!("  Duplicates skipped:  {}", stats.duplicates_skipped);
    Ok(())
}

fn cmd_wing_list(db_path: &str) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let wings = palace.storage().list_wings();

    if wings.is_empty() {
        println!("No wings in this palace.");
        return Ok(());
    }

    println!("Wings ({}):\n", wings.len());
    for (i, w) in wings.iter().enumerate() {
        let connector = if i == wings.len() - 1 { "└──" } else { "├──" };
        let rooms = palace.storage().list_rooms(&w.id);
        println!(
            "  {} {} [{}] ({} rooms)",
            connector,
            w.name,
            w.wing_type,
            rooms.len()
        );
        println!("  {}   ID: {}", if i == wings.len() - 1 { " " } else { "│" }, w.id);
    }
    Ok(())
}

fn cmd_wing_add(
    db_path: &str,
    name: &str,
    wing_type_str: &str,
    description: Option<&str>,
) -> anyhow::Result<()> {
    let mut palace = load_palace(db_path)?;
    let wt = parse_wing_type(wing_type_str);
    let desc = description.unwrap_or("");
    let wing_id = palace.add_wing(name, wt, desc)?;
    save_palace(&palace, db_path)?;

    println!("✓ Wing '{}' added (id: {})", name, wing_id);
    Ok(())
}

fn cmd_room_list(db_path: &str, wing_name: &str) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let wing = palace
        .storage()
        .find_wing_by_name(wing_name)
        .ok_or_else(|| anyhow::anyhow!("Wing '{}' not found", wing_name))?;

    let rooms = palace.storage().list_rooms(&wing.id);

    if rooms.is_empty() {
        println!("No rooms in wing '{}'.", wing_name);
        return Ok(());
    }

    println!("Rooms in '{}' ({}):\n", wing_name, rooms.len());
    for (i, r) in rooms.iter().enumerate() {
        let connector = if i == rooms.len() - 1 { "└──" } else { "├──" };
        println!(
            "  {} {} [{}]",
            connector, r.name, r.hall_type
        );
        println!(
            "  {}   ID: {}",
            if i == rooms.len() - 1 { " " } else { "│" },
            r.id
        );
    }
    Ok(())
}

fn cmd_room_add(
    db_path: &str,
    wing_name: &str,
    room_name: &str,
    hall_type_str: &str,
) -> anyhow::Result<()> {
    let mut palace = load_palace(db_path)?;
    let wing = palace
        .storage()
        .find_wing_by_name(wing_name)
        .ok_or_else(|| anyhow::anyhow!("Wing '{}' not found", wing_name))?;

    let ht = parse_hall_type(hall_type_str);
    let room_id = palace.add_room(&wing.id, room_name, ht)?;
    save_palace(&palace, db_path)?;

    println!(
        "✓ Room '{}' added to wing '{}' (id: {})",
        room_name, wing_name, room_id
    );
    Ok(())
}

fn cmd_kg_add(
    db_path: &str,
    subject: &str,
    predicate: &str,
    object: &str,
) -> anyhow::Result<()> {
    let mut palace = load_palace(db_path)?;
    let rel_id = palace.kg_add(subject, predicate, object)?;
    save_palace(&palace, db_path)?;

    println!(
        "✓ Triple added: {} --[{}]--> {} (id: {})",
        subject, predicate, object, rel_id
    );
    Ok(())
}

fn cmd_kg_query(db_path: &str, entity: &str) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let rels = palace.kg_query(entity)?;

    if rels.is_empty() {
        println!("No relationships found for '{}'", entity);
        return Ok(());
    }

    println!("Knowledge graph for '{}' ({} relationships):\n", entity, rels.len());
    for (i, r) in rels.iter().enumerate() {
        let connector = if i == rels.len() - 1 { "└──" } else { "├──" };
        println!(
            "  {} {} --[{}]--> {} (confidence: {:.2})",
            connector, r.subject, r.predicate, r.object, r.confidence
        );
    }
    Ok(())
}

fn cmd_pheromone_status(db_path: &str, node_id: &str) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let data = palace.storage().read_data();

    // Check all node types for pheromone data
    if let Some(w) = data.wings.get(node_id) {
        let p = &w.pheromones;
        println!("Pheromones for wing '{}' ({}):", w.name, node_id);
        println!("  Exploitation: {:.4}", p.exploitation);
        println!("  Exploration:  {:.4}", p.exploration);
        return Ok(());
    }
    if let Some(r) = data.rooms.get(node_id) {
        let p = &r.pheromones;
        println!("Pheromones for room '{}' ({}):", r.name, node_id);
        println!("  Exploitation: {:.4}", p.exploitation);
        println!("  Exploration:  {:.4}", p.exploration);
        return Ok(());
    }
    if let Some(c) = data.closets.get(node_id) {
        let p = &c.pheromones;
        println!("Pheromones for closet '{}' ({}):", c.name, node_id);
        println!("  Exploitation: {:.4}", p.exploitation);
        println!("  Exploration:  {:.4}", p.exploration);
        return Ok(());
    }
    if let Some(d) = data.drawers.get(node_id) {
        let p = &d.pheromones;
        let snippet: String = d.content.chars().take(40).collect();
        println!("Pheromones for drawer '{}…' ({}):", snippet, node_id);
        println!("  Exploitation: {:.4}", p.exploitation);
        println!("  Exploration:  {:.4}", p.exploration);
        return Ok(());
    }

    anyhow::bail!("Node '{}' not found", node_id);
}

fn cmd_pheromone_hot(db_path: &str, k: usize) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let paths = palace.hot_paths(k)?;

    if paths.is_empty() {
        println!("No hot paths — palace has no pheromone deposits yet.");
        return Ok(());
    }

    println!("Hot paths (top {}):\n", paths.len());
    println!(
        "  {:<4} {:<7} {:<24} To",
        "#", "Score", "From"
    );
    println!("  {}", "─".repeat(60));
    for (i, p) in paths.iter().enumerate() {
        println!(
            "  {:<4} {:<7.3} {:<24} {}",
            i + 1,
            p.success_pheromone,
            truncate(&p.from_id, 22),
            truncate(&p.to_id, 22)
        );
    }
    Ok(())
}

fn cmd_pheromone_cold(db_path: &str, k: usize) -> anyhow::Result<()> {
    let palace = load_palace(db_path)?;
    let spots = palace.cold_spots(k)?;

    if spots.is_empty() {
        println!("No cold spots detected.");
        return Ok(());
    }

    println!("Cold spots (top {}):\n", spots.len());
    println!(
        "  {:<4} {:<10} {:<24} Name",
        "#", "Pheromone", "Node ID"
    );
    println!("  {}", "─".repeat(60));
    for (i, s) in spots.iter().enumerate() {
        println!(
            "  {:<4} {:<10.4} {:<24} {}",
            i + 1,
            s.total_pheromone,
            truncate(&s.node_id, 22),
            truncate(&s.name, 30)
        );
    }
    Ok(())
}

fn cmd_pheromone_decay(db_path: &str) -> anyhow::Result<()> {
    let mut palace = load_palace(db_path)?;
    palace.decay_pheromones()?;
    save_palace(&palace, db_path)?;
    println!("✓ Pheromone decay cycle applied.");
    Ok(())
}

fn cmd_agent_list() {
    println!("Agent archetypes:\n");
    println!("  ├── explorer     — Exploration-biased pathfinding agent");
    println!("  ├── exploiter    — Exploitation-biased, follows hot paths");
    println!("  ├── researcher   — Hypothesis-testing with Bayesian updates");
    println!("  ├── curator      — Knowledge graph maintenance and linking");
    println!("  └── sentinel     — Anomaly detection and contradiction finding");
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Init { name, description } => {
            cmd_init(&cli.db, name, description.as_deref())?;
        }
        Commands::AddDrawer {
            content,
            wing,
            room,
            source,
        } => {
            cmd_add_drawer(&cli.db, content, wing, room, source)?;
        }
        Commands::Search { query, k } => {
            cmd_search(&cli.db, query, *k)?;
        }
        Commands::Navigate { from, to, context } => {
            cmd_navigate(&cli.db, from, to, context.as_deref())?;
        }
        Commands::Status => {
            cmd_status(&cli.db)?;
        }
        Commands::Export { output } => {
            cmd_export(&cli.db, output)?;
        }
        Commands::Import { input, mode } => {
            cmd_import(&cli.db, input, mode)?;
        }
        Commands::Wing { command } => match command {
            WingCommands::List => cmd_wing_list(&cli.db)?,
            WingCommands::Add {
                name,
                wing_type,
                description,
            } => cmd_wing_add(&cli.db, name, wing_type, description.as_deref())?,
        },
        Commands::Room { command } => match command {
            RoomCommands::List { wing } => cmd_room_list(&cli.db, wing)?,
            RoomCommands::Add {
                wing,
                name,
                hall_type,
            } => cmd_room_add(&cli.db, wing, name, hall_type)?,
        },
        Commands::Kg { command } => match command {
            KgCommands::Add {
                subject,
                predicate,
                object,
            } => cmd_kg_add(&cli.db, subject, predicate, object)?,
            KgCommands::Query { entity } => cmd_kg_query(&cli.db, entity)?,
            KgCommands::Timeline { entity: _ } => {
                println!("Feature requires temporal store — not yet implemented.");
            }
            KgCommands::Contradictions { entity: _ } => {
                println!("Feature requires temporal store — not yet implemented.");
            }
        },
        Commands::Pheromone { command } => match command {
            PheromoneCommands::Status { node_id } => cmd_pheromone_status(&cli.db, node_id)?,
            PheromoneCommands::Hot { k } => cmd_pheromone_hot(&cli.db, *k)?,
            PheromoneCommands::Cold { k } => cmd_pheromone_cold(&cli.db, *k)?,
            PheromoneCommands::Decay => cmd_pheromone_decay(&cli.db)?,
        },
        Commands::Agent { command } => match command {
            AgentCommands::List => cmd_agent_list(),
            AgentCommands::Diary {
                agent_id: _,
                last_n: _,
            } => {
                println!("Agent diary requires persistent agent runtime — not yet implemented.");
            }
        },
        Commands::Serve { port, bind } => {
            println!(
                "MCP server requires async runtime — use `graphpalace-server` binary.\n\
                 (Would bind to {}:{})",
                bind, port
            );
        }
    }

    Ok(())
}
