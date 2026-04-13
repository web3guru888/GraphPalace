//! Continuous swarm soak test runner for GraphPalace.
//!
//! Creates a palace of configurable size, populates it with agents
//! (one per archetype), and runs continuous swarm cycles logging
//! per-cycle NDJSON metrics until convergence or `max_cycles`.

use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;

use gp_agents::active_inference::ActiveInferenceAgent;
use gp_bench::generators::generate_palace;
use gp_core::config::PheromoneConfig;
use gp_core::types::{GraphNode, NodePheromones, EdgePheromones, zero_embedding};
use gp_swarm::convergence::ConvergenceDetector;
use gp_swarm::coordinator::SwarmCoordinator;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Continuous swarm soak test: runs agent cycles until convergence.
#[derive(Parser, Debug)]
#[command(name = "soak-test", about = "Continuous agent swarm soak test")]
struct Cli {
    /// Total number of drawers in the palace.
    #[arg(long, default_value_t = 500)]
    drawers: usize,

    /// Number of wings.
    #[arg(long, default_value_t = 4)]
    wings: usize,

    /// Number of rooms per wing.
    #[arg(long, default_value_t = 6)]
    rooms: usize,

    /// Maximum cycles before stopping.
    #[arg(long, default_value_t = 1000)]
    max_cycles: usize,

    /// Random seed for determinism.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// ---------------------------------------------------------------------------
// Agent archetypes
// ---------------------------------------------------------------------------

/// Five canonical archetypes with distinct temperature / goal bias.
///
/// - **Explorer**: high temperature (1.0) — wide, stochastic search
/// - **Exploiter**: low temperature (0.1) — greedy, follows pheromones
/// - **Balanced**: medium temperature (0.5) — equal explore/exploit
/// - **Specialist**: medium-low temperature (0.3) — narrow goal focus
///   (goal embedding has a spike in dimension 0)
/// - **Generalist**: medium-high temperature (0.7) — broad coverage
fn make_agents() -> Vec<ActiveInferenceAgent> {
    let zero = zero_embedding();

    // Specialist gets a non-zero goal embedding (spike in dim 0).
    let mut specialist_goal = zero;
    specialist_goal[0] = 1.0;

    vec![
        ActiveInferenceAgent::new(
            "explorer".into(),
            "Explorer".into(),
            zero,
            1.0,
        ),
        ActiveInferenceAgent::new(
            "exploiter".into(),
            "Exploiter".into(),
            zero,
            0.1,
        ),
        ActiveInferenceAgent::new(
            "balanced".into(),
            "Balanced".into(),
            zero,
            0.5,
        ),
        ActiveInferenceAgent::new(
            "specialist".into(),
            "Specialist".into(),
            specialist_goal,
            0.3,
        ),
        ActiveInferenceAgent::new(
            "generalist".into(),
            "Generalist".into(),
            zero,
            0.7,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Frontier builder
// ---------------------------------------------------------------------------

/// Build a frontier of `GraphNode`s from the palace's drawers.
///
/// Uses the palace's storage to extract all drawers, converting them
/// into `GraphNode` values suitable for the swarm coordinator.
fn build_frontier(palace: &gp_palace::GraphPalace) -> Vec<GraphNode> {
    let data = palace.storage().read_data();
    data.drawers
        .values()
        .map(|d| GraphNode {
            id: d.id.clone(),
            label: d.content.chars().take(40).collect(),
            embedding: d.embedding,
            pheromones: d.pheromones.clone(),
            degree: 0, // drawers are leaf nodes
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║       GraphPalace Soak Test — Continuous Swarm          ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Wings:      {}", cli.wings);
    eprintln!("  Rooms/wing: {}", cli.rooms);
    eprintln!("  Drawers:    {}", cli.drawers);
    eprintln!("  Max cycles: {}", cli.max_cycles);
    eprintln!("  Seed:       {}", cli.seed);
    eprintln!();

    // Compute drawers per room (distribute evenly, minimum 1).
    let total_rooms = cli.wings * cli.rooms;
    let drawers_per_room = if total_rooms == 0 {
        cli.drawers
    } else {
        (cli.drawers / total_rooms).max(1)
    };
    let actual_drawers = cli.wings * cli.rooms * drawers_per_room;

    eprintln!("  Drawers/room: {} (actual total: {})", drawers_per_room, actual_drawers);
    eprintln!();

    // ── 1. Generate palace ────────────────────────────────────────────
    eprintln!("  Generating palace...");
    let (palace, _contents) = generate_palace(
        cli.wings,
        cli.rooms,
        drawers_per_room,
        cli.seed,
    );
    let status = palace.status().unwrap();
    eprintln!(
        "  Palace: {} wings, {} rooms, {} closets, {} drawers",
        status.wing_count, status.room_count, status.closet_count, status.drawer_count,
    );

    // ── 2. Build initial frontier ────────────────────────────────────
    let frontier = build_frontier(&palace);
    eprintln!("  Frontier size: {}", frontier.len());

    // ── 3. Create node & edge pheromone arrays ───────────────────────
    let mut node_pheromones: Vec<NodePheromones> =
        vec![NodePheromones::default(); frontier.len()];
    let mut edge_pheromones: Vec<EdgePheromones> =
        vec![EdgePheromones::default(); frontier.len().saturating_sub(1)];

    // ── 4. Create agents ─────────────────────────────────────────────
    let agents = make_agents();
    eprintln!("  Agents: {}", agents.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", "));

    // ── 5. Create coordinator ────────────────────────────────────────
    let convergence_detector = ConvergenceDetector::from_config(
        20,    // history_window
        5.0,   // growth_threshold
        0.05,  // variance_threshold
        10,    // frontier_threshold
    );

    let mut coordinator = SwarmCoordinator::new(
        agents,
        10,               // decay_interval
        cli.max_cycles,
        PheromoneConfig::default(),
        convergence_detector,
    );

    let mut rng = StdRng::seed_from_u64(cli.seed);

    eprintln!();
    eprintln!("  Running swarm cycles...");
    eprintln!();

    // ── 6. Run cycles ────────────────────────────────────────────────
    let mut prev_pheromone_mass: f64 = 0.0;

    loop {
        let cycle_num = coordinator.cycle_count();
        if coordinator.should_stop() {
            break;
        }

        let result = coordinator.run_cycle(
            &frontier,
            &mut node_pheromones,
            &mut edge_pheromones,
            &mut rng,
        );

        // Compute metrics.
        let node_mass: f64 = node_pheromones
            .iter()
            .map(|np| np.exploitation + np.exploration)
            .sum();
        let edge_mass: f64 = edge_pheromones
            .iter()
            .map(|ep| ep.success + ep.traversal + ep.recency)
            .sum();
        let total_mass = node_mass + edge_mass;
        let growth_rate = total_mass - prev_pheromone_mass;
        prev_pheromone_mass = total_mass;

        let pheromone_variance = if !edge_pheromones.is_empty() {
            let values: Vec<f64> = edge_pheromones
                .iter()
                .map(|e| e.success + e.traversal + e.recency)
                .collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
        } else {
            0.0
        };

        // Build per-agent action summaries.
        let agent_actions: Vec<serde_json::Value> = result.actions.iter().map(|a| {
            json!({
                "agent": a.agent_id,
                "target": a.target_node_id,
                "productive": a.productive,
                "efe": (a.efe_value * 1000.0).round() / 1000.0,
            })
        }).collect();

        // Convergence criteria snapshot.
        let criteria = coordinator.convergence_detector.evaluate_criteria(&coordinator.history);
        let criteria_met = coordinator.convergence_detector.criteria_met_count(&coordinator.history);

        // Emit NDJSON line to stdout.
        let line = json!({
            "cycle": cycle_num,
            "frontier_size": frontier.len(),
            "productive_count": result.productive_count,
            "pheromone_mass": (total_mass * 1000.0).round() / 1000.0,
            "growth_rate": (growth_rate * 1000.0).round() / 1000.0,
            "pheromone_variance": (pheromone_variance * 100_000.0).round() / 100_000.0,
            "decay_applied": result.decay_applied,
            "converged": result.converged,
            "criteria": {
                "growth_converged": criteria[0],
                "variance_converged": criteria[1],
                "frontier_converged": criteria[2],
                "met": criteria_met,
            },
            "agents": agent_actions,
        });
        println!("{}", line);

        // Progress every 100 cycles on stderr.
        if (cycle_num + 1).is_multiple_of(100) {
            eprintln!(
                "  cycle {:>5} | mass={:.2} | Δ={:.4} | var={:.6} | criteria={}/3 | converged={}",
                cycle_num + 1,
                total_mass,
                growth_rate,
                pheromone_variance,
                criteria_met,
                result.converged,
            );
        }

        if result.converged {
            eprintln!("  ✅ Convergence detected at cycle {}!", cycle_num);
            break;
        }
    }

    // ── 7. Summary ───────────────────────────────────────────────────
    let total_cycles = coordinator.cycle_count();
    let converged = coordinator.convergence_detector.is_converged(&coordinator.history);

    let node_mass: f64 = node_pheromones
        .iter()
        .map(|np| np.exploitation + np.exploration)
        .sum();
    let edge_mass: f64 = edge_pheromones
        .iter()
        .map(|ep| ep.success + ep.traversal + ep.recency)
        .sum();

    // Per-agent belief summaries.
    let mut agent_summaries: Vec<serde_json::Value> = Vec::new();
    for agent in &coordinator.agents {
        let num_beliefs = agent.beliefs.len();
        let avg_precision = if num_beliefs > 0 {
            agent.beliefs.values().map(|b| b.precision).sum::<f64>() / num_beliefs as f64
        } else {
            0.0
        };
        let avg_mean = if num_beliefs > 0 {
            agent.beliefs.values().map(|b| b.mean).sum::<f64>() / num_beliefs as f64
        } else {
            0.0
        };
        agent_summaries.push(json!({
            "id": agent.id,
            "name": agent.name,
            "temperature": agent.temperature,
            "beliefs_count": num_beliefs,
            "avg_precision": (avg_precision * 1000.0).round() / 1000.0,
            "avg_mean": (avg_mean * 1000.0).round() / 1000.0,
        }));
    }

    // Pheromone distribution across nodes.
    let node_exploitation: Vec<f64> = node_pheromones.iter().map(|np| np.exploitation).collect();
    let node_exploration: Vec<f64> = node_pheromones.iter().map(|np| np.exploration).collect();

    let exploit_mean = if !node_exploitation.is_empty() {
        node_exploitation.iter().sum::<f64>() / node_exploitation.len() as f64
    } else {
        0.0
    };
    let explore_mean = if !node_exploration.is_empty() {
        node_exploration.iter().sum::<f64>() / node_exploration.len() as f64
    } else {
        0.0
    };
    let exploit_max = node_exploitation.iter().cloned().fold(0.0_f64, f64::max);
    let explore_max = node_exploration.iter().cloned().fold(0.0_f64, f64::max);

    // History stats.
    let avg_growth = coordinator.history.avg_growth(20);
    let avg_variance = coordinator.history.avg_pheromone_variance(20);

    let summary = json!({
        "type": "summary",
        "total_cycles": total_cycles,
        "converged": converged,
        "palace": {
            "wings": cli.wings,
            "rooms_per_wing": cli.rooms,
            "drawers": actual_drawers,
            "frontier_size": frontier.len(),
        },
        "pheromones": {
            "node_mass": (node_mass * 1000.0).round() / 1000.0,
            "edge_mass": (edge_mass * 1000.0).round() / 1000.0,
            "total_mass": ((node_mass + edge_mass) * 1000.0).round() / 1000.0,
            "node_exploitation_mean": (exploit_mean * 1000.0).round() / 1000.0,
            "node_exploitation_max": (exploit_max * 1000.0).round() / 1000.0,
            "node_exploration_mean": (explore_mean * 1000.0).round() / 1000.0,
            "node_exploration_max": (explore_max * 1000.0).round() / 1000.0,
        },
        "convergence": {
            "avg_growth_rate_20": (avg_growth * 1000.0).round() / 1000.0,
            "avg_variance_20": (avg_variance * 100_000.0).round() / 100_000.0,
            "frontier_size": frontier.len(),
            "criteria_met": coordinator.convergence_detector.criteria_met_count(&coordinator.history),
        },
        "agents": agent_summaries,
    });
    println!("{}", summary);

    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║                    SOAK TEST COMPLETE                   ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!("  Total cycles: {}", total_cycles);
    eprintln!("  Converged:    {}", converged);
    eprintln!("  Node mass:    {:.4}", node_mass);
    eprintln!("  Edge mass:    {:.4}", edge_mass);
    eprintln!("  Agents:       {}", coordinator.agent_count());
    eprintln!();
}
