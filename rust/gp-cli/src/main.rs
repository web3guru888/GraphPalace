use clap::{Parser, Subcommand};

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

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Init { name, description } => {
            println!("Not yet implemented: init palace '{}' at '{}'", name, cli.db);
            if let Some(desc) = description {
                println!("  Description: {}", desc);
            }
        }
        Commands::AddDrawer { content, wing, room, source } => {
            println!("Not yet implemented: add drawer to {}/{}", wing, room);
            println!("  Content: {}...", &content[..content.len().min(80)]);
            println!("  Source: {}", source);
        }
        Commands::Search { query, wing, room, k } => {
            println!("Not yet implemented: search '{}' (k={})", query, k);
            if let Some(w) = wing { println!("  Wing filter: {}", w); }
            if let Some(r) = room { println!("  Room filter: {}", r); }
        }
        Commands::Navigate { from, to, context } => {
            println!("Not yet implemented: navigate {} → {} (context={})", from, to, context);
        }
        Commands::Status { verbose } => {
            println!("Not yet implemented: palace status (verbose={})", verbose);
        }
        Commands::Export { output, format } => {
            println!("Not yet implemented: export to {} (format={})", output, format);
        }
        Commands::Import { input, format, mode } => {
            println!("Not yet implemented: import from {} (format={}, mode={})", input, format, mode);
        }
        Commands::Wing { command } => match command {
            WingCommands::List => println!("Not yet implemented: list wings"),
            WingCommands::Add { name, wing_type, description } => {
                println!("Not yet implemented: add wing '{}' (type={})", name, wing_type);
                if let Some(desc) = description { println!("  Description: {}", desc); }
            }
        },
        Commands::Room { command } => match command {
            RoomCommands::List { wing } => println!("Not yet implemented: list rooms in '{}'", wing),
            RoomCommands::Add { wing, name, hall_type } => {
                println!("Not yet implemented: add room '{}' to wing '{}' (type={})", name, wing, hall_type);
            }
        },
        Commands::Kg { command } => match command {
            KgCommands::Add { subject, predicate, object, confidence } => {
                println!("Not yet implemented: kg add ({} --{}-> {}, conf={})", subject, predicate, object, confidence);
            }
            KgCommands::Query { entity } => println!("Not yet implemented: kg query '{}'", entity),
            KgCommands::Timeline { entity } => println!("Not yet implemented: kg timeline '{}'", entity),
            KgCommands::Contradictions { entity } => println!("Not yet implemented: kg contradictions '{}'", entity),
        },
        Commands::Pheromone { command } => match command {
            PheromoneCommands::Status { node_id } => println!("Not yet implemented: pheromone status '{}'", node_id),
            PheromoneCommands::Hot { k } => println!("Not yet implemented: hot paths (k={})", k),
            PheromoneCommands::Cold { k } => println!("Not yet implemented: cold spots (k={})", k),
            PheromoneCommands::Decay => println!("Not yet implemented: force decay"),
        },
        Commands::Agent { command } => match command {
            AgentCommands::List => println!("Not yet implemented: list agents"),
            AgentCommands::Diary { agent_id, last_n } => {
                println!("Not yet implemented: diary for '{}' (last {})", agent_id, last_n);
            }
        },
        Commands::Serve { port, bind } => {
            println!("Not yet implemented: MCP server on {}:{}", bind, port);
        }
    }
}
