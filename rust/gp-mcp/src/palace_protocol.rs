//! Dynamic PALACE_PROTOCOL prompt generation (spec §8.2).
//!
//! Extends the static PALACE_PROTOCOL prompt with live palace statistics
//! (wing count, room count, drawer count, etc.) injected at runtime.

use serde::{Deserialize, Serialize};

/// Live palace statistics for prompt injection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PalaceStats {
    /// Total number of wings.
    pub wing_count: usize,
    /// Total number of rooms.
    pub room_count: usize,
    /// Total number of closets.
    pub closet_count: usize,
    /// Total number of drawers.
    pub drawer_count: usize,
    /// Total number of entities in the knowledge graph.
    pub entity_count: usize,
    /// Total number of relationships in the knowledge graph.
    pub relationship_count: usize,
    /// Total number of specialist agents.
    pub agent_count: usize,
    /// Top 3 hottest paths (most traversed).
    pub hot_paths: Vec<String>,
    /// Top 3 coldest spots (least explored).
    pub cold_spots: Vec<String>,
}

/// Header for the PALACE_PROTOCOL prompt.
pub const PALACE_PROTOCOL_HEADER: &str = "PALACE_PROTOCOL v1.0";

/// Generate the complete PALACE_PROTOCOL prompt with live statistics.
///
/// This is returned by the `palace_status` tool and teaches the LLM
/// how to interact with the memory palace.
pub fn generate_palace_protocol(stats: &PalaceStats) -> String {
    let mut prompt = String::with_capacity(2048);

    prompt.push_str(PALACE_PROTOCOL_HEADER);
    prompt.push_str("\n\n");

    // Stats section
    prompt.push_str("PALACE STATUS:\n");
    prompt.push_str(&format!(
        "  Wings: {}  |  Rooms: {}  |  Closets: {}  |  Drawers: {}\n",
        stats.wing_count, stats.room_count, stats.closet_count, stats.drawer_count
    ));
    prompt.push_str(&format!(
        "  Entities: {}  |  Relationships: {}  |  Agents: {}\n",
        stats.entity_count, stats.relationship_count, stats.agent_count
    ));

    if !stats.hot_paths.is_empty() {
        prompt.push_str("  Hot paths: ");
        prompt.push_str(&stats.hot_paths.join(", "));
        prompt.push('\n');
    }
    if !stats.cold_spots.is_empty() {
        prompt.push_str("  Cold spots: ");
        prompt.push_str(&stats.cold_spots.join(", "));
        prompt.push('\n');
    }

    prompt.push_str("\nYou have a memory palace. It's a real graph — not a metaphor.\n\n");

    prompt.push_str("RULES:\n");
    prompt.push_str("1. SEARCH before claiming you don't know something\n");
    prompt.push_str("2. NAVIGATE to follow connections between ideas\n");
    prompt.push_str("3. ADD important information to appropriate rooms\n");
    prompt.push_str("4. CHECK for duplicates before adding\n");
    prompt.push_str("5. DEPOSIT pheromones when a path is useful (reinforcement)\n");
    prompt.push_str("6. READ agent diaries for specialist knowledge\n");
    prompt.push_str("7. TRAVERSE the knowledge graph for causal chains\n\n");

    prompt.push_str("NAVIGATION:\n");
    prompt.push_str("- Wings = domains (projects, people, topics)\n");
    prompt.push_str("- Rooms = specific subjects within a wing\n");
    prompt.push_str("- Halls = corridors connecting rooms in the same wing\n");
    prompt.push_str("- Tunnels = cross-wing connections (same topic, different domain)\n");
    prompt.push_str("- Closets = summaries pointing to drawers\n");
    prompt.push_str("- Drawers = verbatim original content (never summarized)\n\n");

    prompt.push_str("PHEROMONES:\n");
    prompt.push_str("- Exploitation (node): \"Come here — this is valuable\"\n");
    prompt.push_str("- Exploration (node): \"Already checked — try elsewhere\"\n");
    prompt.push_str("- Success (edge): \"This connection led to good results\"\n");
    prompt.push_str("- Traversal (edge): \"This path is frequently used\"\n");
    prompt.push_str("- Recency (edge): \"This was used recently\"\n\n");

    prompt.push_str("Hot paths = strong success pheromones. Follow them.\n");
    prompt.push_str("Cold spots = low exploration pheromones. Investigate them.\n");

    prompt
}

/// Generate a minimal protocol prompt without stats (for init/bootstrap).
pub fn generate_palace_protocol_minimal() -> String {
    generate_palace_protocol(&PalaceStats::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_contains_header() {
        let prompt = generate_palace_protocol_minimal();
        assert!(prompt.starts_with("PALACE_PROTOCOL v1.0"));
    }

    #[test]
    fn protocol_contains_rules() {
        let prompt = generate_palace_protocol_minimal();
        assert!(prompt.contains("RULES:"));
        assert!(prompt.contains("SEARCH before"));
        assert!(prompt.contains("NAVIGATE to"));
        assert!(prompt.contains("DEPOSIT pheromones"));
    }

    #[test]
    fn protocol_contains_navigation() {
        let prompt = generate_palace_protocol_minimal();
        assert!(prompt.contains("NAVIGATION:"));
        assert!(prompt.contains("Wings = domains"));
        assert!(prompt.contains("Drawers = verbatim"));
    }

    #[test]
    fn protocol_contains_pheromones() {
        let prompt = generate_palace_protocol_minimal();
        assert!(prompt.contains("PHEROMONES:"));
        assert!(prompt.contains("Exploitation"));
        assert!(prompt.contains("Exploration"));
        assert!(prompt.contains("Success"));
        assert!(prompt.contains("Hot paths"));
    }

    #[test]
    fn protocol_with_stats() {
        let stats = PalaceStats {
            wing_count: 5,
            room_count: 20,
            closet_count: 40,
            drawer_count: 200,
            entity_count: 50,
            relationship_count: 120,
            agent_count: 3,
            hot_paths: vec![
                "wing_a \u{2192} room_b".into(),
                "room_c \u{2192} room_d".into(),
            ],
            cold_spots: vec!["wing_x".into()],
        };
        let prompt = generate_palace_protocol(&stats);
        assert!(prompt.contains("Wings: 5"));
        assert!(prompt.contains("Drawers: 200"));
        assert!(prompt.contains("Entities: 50"));
        assert!(prompt.contains("Agents: 3"));
        assert!(prompt.contains("wing_a"));
        assert!(prompt.contains("room_b"));
        assert!(prompt.contains("wing_x"));
    }

    #[test]
    fn protocol_with_empty_lists() {
        let stats = PalaceStats {
            wing_count: 1,
            room_count: 2,
            hot_paths: vec![],
            cold_spots: vec![],
            ..PalaceStats::default()
        };
        let prompt = generate_palace_protocol(&stats);
        assert!(!prompt.contains("Hot paths:"));
        assert!(!prompt.contains("Cold spots:"));
        assert!(prompt.contains("Wings: 1"));
    }

    #[test]
    fn protocol_stats_default() {
        let stats = PalaceStats::default();
        assert_eq!(stats.wing_count, 0);
        assert_eq!(stats.drawer_count, 0);
        assert!(stats.hot_paths.is_empty());
    }

    #[test]
    fn protocol_stats_serialization() {
        let stats = PalaceStats {
            wing_count: 3,
            room_count: 10,
            ..PalaceStats::default()
        };
        let json = serde_json::to_string(&stats).unwrap();
        let stats2: PalaceStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats2.wing_count, 3);
        assert_eq!(stats2.room_count, 10);
    }
}
