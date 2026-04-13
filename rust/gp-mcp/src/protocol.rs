//! PALACE_PROTOCOL — the system prompt preamble injected into every
//! MCP session so that LLM clients understand the palace metaphor,
//! tool naming conventions, and expected interaction patterns.

/// The full PALACE_PROTOCOL v1.0 prompt text (§8.2).
///
/// This constant is sent to the client during MCP session initialisation
/// so that the LLM understands how to interact with GraphPalace tools.
pub const PALACE_PROTOCOL: &str = r#"PALACE_PROTOCOL v1.0

You have access to a **Memory Palace** — a spatial, graph-backed knowledge
store that organises information hierarchically and lets you search, navigate,
and reason over everything you have ever learned.

## Palace Structure

The palace is organised as a four-level hierarchy:

  Wing  →  Room  →  Closet  →  Drawer
  (domain)  (topic)  (subtopic)  (individual memory)

Each drawer holds a single piece of knowledge (text, source, metadata) and is
embedded in a vector space for semantic search. Drawers are also connected
through a **knowledge graph** of (subject, predicate, object) triples with
timestamps and confidence scores.

## Stigmergy

The palace tracks how you traverse it. Frequently-visited paths accumulate
**pheromone** that strengthens them; rarely-used paths decay. Use stigmergy
tools (`hot_paths`, `cold_spots`, `pheromone_status`) to discover what
knowledge is most — or least — utilised.

## Knowledge Graph

Beyond the hierarchy, knowledge is linked via temporal triples. You can:
- **Add** new relationships (`kg_add`)
- **Query** an entity's relationships (`kg_query`)
- **Traverse** the graph (`kg_traverse`)
- **Find contradictions** (`kg_contradictions`)
- **View history** (`kg_timeline`)
- **Invalidate** stale facts (`kg_invalidate`)

Triples are never deleted — only invalidated — so you can always reconstruct
the full history of what was believed and when.

## Specialist Agents

Domain-specific agents maintain **diaries** of their ongoing work.
Use `list_agents` to see who is available, `diary_read` to review their
recent findings, and `diary_write` to record new observations.

## Tool Naming Conventions

| Prefix            | Category                  | Side-Effects |
|-------------------|---------------------------|-------------|
| `palace_*`        | Palace-level status       | None        |
| `list_*` / `get_*`| Navigation & read         | None        |
| `search`          | Semantic vector search    | None        |
| `navigate`        | Shortest-path routing     | None        |
| `find_tunnels`    | Cross-wing bridge lookup  | None        |
| `graph_stats`     | Aggregate statistics      | None        |
| `add_*`           | Create new structures     | Write       |
| `delete_*`        | Remove structures         | Write       |
| `check_duplicate` | Near-duplicate detection  | None        |
| `kg_*`            | Knowledge graph operations| Mixed       |
| `pheromone_*`     | Stigmergy read/write      | Mixed       |
| `hot_paths`       | High-traffic paths        | None        |
| `cold_spots`      | Low-traffic areas         | None        |
| `decay_now`       | Force pheromone decay     | Write       |
| `diary_*`         | Agent diary operations    | Mixed       |

## Interaction Guidelines

1. **Search before you store.** Always use `search` or `check_duplicate`
   before `add_drawer` to avoid redundant memories.
2. **Navigate, don't enumerate.** Use `navigate` and `find_tunnels` to
   discover connections rather than manually listing everything.
3. **Maintain the graph.** When you learn a new relationship, add it with
   `kg_add`. When a fact becomes outdated, invalidate it with `kg_invalidate`.
4. **Respect confidence.** Attach confidence scores to triples. Use
   `kg_contradictions` to find and resolve conflicting beliefs.
5. **Use stigmergy.** Check `hot_paths` to see what knowledge you rely on
   most, and `cold_spots` to find neglected areas that may need review.
6. **Write diary entries.** After completing a research task, record findings
   with `diary_write` so future sessions benefit from your work.
7. **Spatial thinking.** Think of the palace as a physical space. Wings are
   buildings, rooms are chambers, closets are cabinets, drawers are files.
   This spatial metaphor aids retrieval and organisation.
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_is_non_empty() {
        assert!(!PALACE_PROTOCOL.is_empty());
    }

    #[test]
    fn protocol_contains_version() {
        assert!(PALACE_PROTOCOL.contains("PALACE_PROTOCOL v1.0"));
    }

    #[test]
    fn protocol_contains_hierarchy() {
        assert!(PALACE_PROTOCOL.contains("Wing"));
        assert!(PALACE_PROTOCOL.contains("Room"));
        assert!(PALACE_PROTOCOL.contains("Closet"));
        assert!(PALACE_PROTOCOL.contains("Drawer"));
    }

    #[test]
    fn protocol_contains_stigmergy() {
        assert!(PALACE_PROTOCOL.contains("pheromone"));
        assert!(PALACE_PROTOCOL.contains("stigmergy") || PALACE_PROTOCOL.contains("Stigmergy"));
    }

    #[test]
    fn protocol_contains_knowledge_graph() {
        assert!(PALACE_PROTOCOL.contains("Knowledge Graph") || PALACE_PROTOCOL.contains("knowledge graph"));
        assert!(PALACE_PROTOCOL.contains("subject"));
        assert!(PALACE_PROTOCOL.contains("predicate"));
        assert!(PALACE_PROTOCOL.contains("object"));
    }

    #[test]
    fn protocol_contains_guidelines() {
        assert!(PALACE_PROTOCOL.contains("Search before you store"));
        assert!(PALACE_PROTOCOL.contains("Spatial thinking"));
    }

    #[test]
    fn protocol_mentions_key_tools() {
        assert!(PALACE_PROTOCOL.contains("kg_add"));
        assert!(PALACE_PROTOCOL.contains("hot_paths"));
        assert!(PALACE_PROTOCOL.contains("diary_write"));
        assert!(PALACE_PROTOCOL.contains("search"));
    }
}
