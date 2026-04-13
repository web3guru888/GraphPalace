# MCP Tools Reference

GraphPalace exposes 28 tools via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing any LLM to interact with the memory palace. Tools are organized into six categories.

## PALACE_PROTOCOL

When an LLM first connects, the `palace_status` tool returns the PALACE_PROTOCOL prompt that teaches the agent how to use the palace:

```
PALACE_PROTOCOL v1.0

You have a memory palace. It's a real graph — not a metaphor.

RULES:
1. SEARCH before claiming you don't know something
2. NAVIGATE to follow connections between ideas
3. ADD important information to appropriate rooms
4. CHECK for duplicates before adding
5. DEPOSIT pheromones when a path is useful (reinforcement)
6. READ agent diaries for specialist knowledge
7. TRAVERSE the knowledge graph for causal chains

NAVIGATION:
- Wings = domains (projects, people, topics)
- Rooms = specific subjects within a wing
- Halls = corridors connecting rooms in the same wing
- Tunnels = cross-wing connections (same topic, different domain)
- Closets = summaries pointing to drawers
- Drawers = verbatim original content (never summarized)

PHEROMONES:
- Exploitation (node): "Come here — this is valuable"
- Exploration (node): "Already checked — try elsewhere"
- Success (edge): "This connection led to good results"
- Traversal (edge): "This path is frequently used"
- Recency (edge): "This was used recently"

Hot paths = strong success pheromones. Follow them.
Cold spots = low exploration pheromones. Investigate them.
```

## 1. Palace Navigation (Read)

### `palace_status`

Get an overview of the palace including PALACE_PROTOCOL prompt.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| — | — | — | No parameters |

**Returns**: Palace name, wing count, room count, drawer count, entity count, agent count, PALACE_PROTOCOL prompt.

### `list_wings`

List all wings in the palace with room counts and pheromone levels.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| — | — | — | No parameters |

**Returns**: Array of wings with `id`, `name`, `wing_type`, `room_count`, `exploitation_pheromone`, `exploration_pheromone`.

### `list_rooms`

List rooms within a specific wing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing_id` | string | ✅ | Wing to list rooms for |

**Returns**: Array of rooms with `id`, `name`, `hall_type`, `closet_count`, `drawer_count`, pheromone levels.

### `get_taxonomy`

Get the full palace hierarchy tree.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| — | — | — | No parameters |

**Returns**: Nested tree: Palace → Wings → Rooms → Closets → Drawers with counts at each level.

### `search`

Semantic search across the palace with pheromone-boosted ranking.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | ✅ | Natural language search query |
| `wing` | string | ❌ | Restrict to a specific wing |
| `room` | string | ❌ | Restrict to a specific room |
| `k` | integer | ❌ | Number of results (default: 10) |

**Returns**: Array of drawers with `id`, `content`, `similarity_score`, `pheromone_boost`, `final_score`, `path` (wing → room → closet).

### `navigate`

Find the optimal path between two nodes using Semantic A*.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `from_id` | string | ✅ | Starting node ID |
| `to_id` | string | ✅ | Goal node ID |
| `context` | string | ❌ | Navigation context: "default", "hypothesis_testing", "exploratory_research", "evidence_gathering", "memory_recall" |

**Returns**: Path with edges, total cost, iterations, nodes expanded, provenance steps.

### `find_tunnels`

Find rooms that connect two wings (cross-domain connections).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing_a` | string | ✅ | First wing name or ID |
| `wing_b` | string | ✅ | Second wing name or ID |

**Returns**: Array of tunnel connections with room pairs and relationship strength.

### `graph_stats`

Get graph connectivity overview.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| — | — | — | No parameters |

**Returns**: Node counts by type, edge counts by type, average degree, clustering coefficient.

## 2. Palace Operations (Write)

### `add_drawer`

Store new content in the palace. Content is stored verbatim and auto-embedded.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | ✅ | Verbatim text to store |
| `wing` | string | ✅ | Wing name (created if doesn't exist) |
| `room` | string | ✅ | Room name (created if doesn't exist) |
| `source` | string | ❌ | Origin: "conversation", "file", "api", "agent" |

**Returns**: New drawer ID, embedding computed, closet assigned, duplicate check result.

### `delete_drawer`

Remove a drawer from the palace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `drawer_id` | string | ✅ | Drawer ID to delete |

**Returns**: Confirmation with deleted drawer info.

### `add_wing`

Create a new wing in the palace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | ✅ | Wing name |
| `type` | string | ✅ | Wing type: "person", "project", "domain", "topic" |
| `description` | string | ❌ | Wing description |

**Returns**: New wing ID.

### `add_room`

Create a new room in a wing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing_id` | string | ✅ | Parent wing ID |
| `name` | string | ✅ | Room name |
| `hall_type` | string | ✅ | Type: "facts", "events", "discoveries", "preferences", "advice" |

**Returns**: New room ID.

### `check_duplicate`

Check if similar content already exists in the palace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | ✅ | Content to check |
| `threshold` | float | ❌ | Similarity threshold (default: 0.9) |

**Returns**: Array of similar existing drawers with similarity scores.

## 3. Knowledge Graph

### `kg_add`

Add a temporal entity-relationship triple to the knowledge graph.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | ✅ | Subject entity name |
| `predicate` | string | ✅ | Relationship: "causes", "inhibits", "correlates_with", etc. |
| `object` | string | ✅ | Object entity name |
| `confidence` | float | ❌ | Confidence level (default: 0.5) |

**Returns**: Triple ID, subject entity ID, object entity ID.

### `kg_query`

Query relationships for an entity, optionally at a point in time.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity` | string | ✅ | Entity name to query |
| `as_of` | timestamp | ❌ | Query at a specific point in time |

**Returns**: Array of relationships with subject, predicate, object, confidence, valid_from, valid_to.

### `kg_invalidate`

Mark a knowledge graph triple as no longer valid (sets `valid_to` to now).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | ✅ | Subject entity name |
| `predicate` | string | ✅ | Relationship predicate |
| `object` | string | ✅ | Object entity name |

**Returns**: Confirmation with invalidated triple info.

### `kg_timeline`

Get the chronological story of an entity — all relationships ordered by time.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity` | string | ✅ | Entity name |

**Returns**: Chronological array of events/relationships.

### `kg_traverse`

Multi-hop subgraph traversal from a starting entity.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start` | string | ✅ | Starting entity name |
| `depth` | integer | ❌ | Maximum hops (default: 3) |
| `predicate` | string | ❌ | Filter by predicate type |

**Returns**: Subgraph of entities and relationships within depth.

### `kg_contradictions`

Find conflicting relationships for an entity.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity` | string | ✅ | Entity to check for contradictions |

**Returns**: Array of contradicting relationship pairs with explanation.

## 4. Stigmergy

### `pheromone_status`

Get current pheromone levels for a node or edge.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | ❌ | Node ID (provide node_id OR edge) |
| `edge` | string | ❌ | Edge identifier ("from_id:to_id") |

**Returns**: All pheromone levels with interpretation.

### `pheromone_deposit`

Manually deposit pheromones along a path.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string[] | ✅ | Array of node IDs defining the path |
| `reward_type` | string | ✅ | "success", "traversal", or "exploration" |

**Returns**: Confirmation with updated pheromone levels.

### `hot_paths`

Find the most-traversed paths (strong success pheromones).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing` | string | ❌ | Restrict to a specific wing |
| `k` | integer | ❌ | Number of paths (default: 20) |

**Returns**: Array of edges sorted by success pheromone strength.

### `cold_spots`

Find unexplored areas (low exploration pheromones).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing` | string | ❌ | Restrict to a specific wing |
| `k` | integer | ❌ | Number of spots (default: 20) |

**Returns**: Array of nodes sorted by lowest exploration pheromone.

### `decay_now`

Force an immediate pheromone decay cycle.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| — | — | — | No parameters |

**Returns**: Confirmation with decay statistics (nodes/edges affected).

## 5. Agent Diary

### `list_agents`

List all specialist agents in the palace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| — | — | — | No parameters |

**Returns**: Array of agents with `id`, `name`, `domain`, `focus`, `temperature`.

### `diary_write`

Append an entry to an agent's persistent diary.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | ✅ | Agent ID |
| `entry` | string | ✅ | Diary entry text |

**Returns**: Confirmation with updated diary length.

### `diary_read`

Read recent entries from an agent's diary.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | ✅ | Agent ID |
| `last_n` | integer | ❌ | Number of recent entries (default: 10) |

**Returns**: Array of diary entries with timestamps.

## 6. System

### `export`

Export the entire palace as a portable file.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `format` | string | ❌ | "json" (default) or "cypher" |

**Returns**: Full palace data in the requested format.

### `import`

Import palace data from an export file.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | string | ✅ | Palace data (JSON or Cypher) |
| `format` | string | ❌ | "json" (default) or "cypher" |
| `mode` | string | ❌ | "replace", "merge" (default), or "overlay" |

**Returns**: Import statistics (nodes created, edges created, duplicates skipped).

## Crate: `gp-mcp`

Source: `rust/gp-mcp/src/`

| File | Purpose |
|------|---------|
| `lib.rs` | Public API, tool registry |
| `server.rs` | MCP server implementation |
| `tools.rs` | Tool implementations (28 tools) |
| `protocol.rs` | MCP protocol handling |
| `palace_protocol.rs` | PALACE_PROTOCOL prompt generation |
