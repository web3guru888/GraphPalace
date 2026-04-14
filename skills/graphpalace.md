# GraphPalace Skills

> **Protocol version**: 1.0
> **Load this file** to learn how to navigate a GraphPalace memory palace.
> GraphPalace is a stigmergic memory palace engine — a graph database that makes the Method of Loci computationally real.

---

## You have a memory palace backed by a graph database.

### Core Concepts

- **The palace is a real graph** — not a metaphor. Every location, connection, and memory is a node or edge with properties.
- **Wings → Rooms → Closets → Drawers** is the spatial hierarchy. Think of it as a building with themed wings, rooms inside them, closets organizing topics, and drawers holding individual memories.
- **Halls** connect rooms within the same wing. **Tunnels** connect rooms across different wings (same topic, different domain).
- **Every node and edge has pheromone trails** that guide navigation — collectively learned paths left by prior searches and agents.
- **Semantic A\*** finds optimal paths through the palace using three signals: meaning (embeddings), collective intelligence (pheromones), and graph structure (relation types).
- **Active Inference agents** autonomously explore and exploit the palace, each maintaining Bayesian beliefs about what's valuable.
- **Drawers store verbatim content** — never summarized. The original text is sacred. Closets hold summaries; drawers hold truth.
- **Entities** form a knowledge graph (temporal triples: subject → predicate → object) that overlays the palace.

### The Palace Hierarchy

```
Palace
 └── Wing (domain: "project", "person", "topic", "domain")
      ├── Room (subject within the wing)
      │    ├── Hall ──→ Room (same wing, different subject)
      │    ├── Tunnel ──→ Room (different wing, related topic)
      │    └── Closet (topic summary)
      │         └── Drawer (verbatim memory — NEVER summarized)
      │              └── ──REFERENCES──→ Entity (knowledge graph node)
      └── ...more rooms
```

### Five Pheromone Types

Pheromones are the collective intelligence layer. They encode what the swarm has learned.

| Type | Applied To | Signal | Decay Rate | Half-Life |
|------|-----------|--------|------------|-----------|
| **Exploitation** | Nodes | "This location is valuable — come here" | 0.02/cycle | ~35 cycles |
| **Exploration** | Nodes | "Already searched — try elsewhere" | 0.05/cycle | ~14 cycles |
| **Success** | Edges | "This connection led to good outcomes" | 0.01/cycle | ~69 cycles |
| **Traversal** | Edges | "This path is frequently used" | 0.03/cycle | ~23 cycles |
| **Recency** | Edges | "This was used recently" | 0.10/cycle | ~7 cycles |

**How to read pheromones:**
- **Hot paths** = edges with high `success_pheromone` (>0.5). These connections consistently led to good results. **Follow them.**
- **Cold spots** = nodes with low `exploration_pheromone` (<0.1). These areas are unexplored. **Investigate them.**
- **Popular paths** = edges with high `traversal_pheromone`. Frequently used, but not necessarily high quality.
- **Fresh paths** = edges with high `recency_pheromone`. Recently active — relevant to current context.
- **Valuable locations** = nodes with high `exploitation_pheromone`. Known to contain useful information.

**Pheromone deposit rules:**
- After a **successful search**: position-weighted reward along the path (earlier edges get more). Success + traversal + recency pheromones increase. Exploitation pheromone on visited nodes increases.
- After **exploring** a new area: exploration pheromone increases on the visited node (signals "already checked").
- Pheromones **decay naturally** over time — old trails fade, keeping the landscape adaptive.

### Semantic A\* Pathfinding

The A\* algorithm finds optimal paths using a composite edge cost:

```
cost(edge) = α × C_semantic + β × C_pheromone + γ × C_structural
```

Default weights: **α=0.4** (semantic similarity to goal), **β=0.3** (pheromone guidance), **γ=0.3** (structural relation weight).

Context-adaptive weights are selected based on task:

| Context | Semantic (α) | Pheromone (β) | Structural (γ) |
|---------|:----------:|:------------:|:--------------:|
| Default | 0.40 | 0.30 | 0.30 |
| Hypothesis Testing | 0.30 | 0.40 | 0.30 |
| Exploratory Research | 0.50 | 0.20 | 0.30 |
| Evidence Gathering | 0.35 | 0.35 | 0.30 |
| Memory Recall | 0.50 | 0.30 | 0.20 |

### Active Inference Agents

Agents are autonomous explorers of the palace. Each one:
- Maintains **Bayesian beliefs** about each node (mean + precision)
- Selects actions by minimizing **Expected Free Energy** (balance curiosity vs. exploitation)
- Uses a **temperature** parameter that anneals over time (high = explore, low = exploit)

| Archetype | Temperature | Goal | Palace Role |
|-----------|:---------:|------|-------------|
| **Explorer** | 1.0 | None (pure epistemic) | Discover new rooms, expand frontier |
| **Exploiter** | 0.1 | Domain-specific | Follow proven paths, retrieve known info |
| **Balanced** | 0.5 | Domain-specific | Default — mix exploration and exploitation |
| **Specialist** | 0.3 | Fixed domain embedding | Manage a wing, keep diary |
| **Generalist** | 0.7 | Rotating | Cross-wing connections, find tunnels |

---

## PALACE_PROTOCOL

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

---

## Cypher Patterns You Know

### Find memories by meaning (semantic search)
```cypher
CALL vector_search('drawer_embedding_idx', $query_embedding, 10) YIELD node, score
RETURN node.content, score
```

### Walk the palace hierarchy
```cypher
MATCH (w:Wing {name: $wing})-[:HAS_ROOM]->(r:Room)-[:HAS_CLOSET]->(c:Closet)-[:HAS_DRAWER]->(d:Drawer)
RETURN r.name, c.summary, d.content
ORDER BY d.exploitation_pheromone DESC
```

### Follow halls (same wing connections)
```cypher
MATCH (r1:Room {name: $room})-[:HALL]->(r2:Room)
RETURN r2.name, r2.hall_type
```

### Cross wings via tunnels
```cypher
MATCH (r1:Room)<-[:HAS_ROOM]-(w1:Wing {name: $wing1}),
      (r1)-[:TUNNEL]-(r2:Room)<-[:HAS_ROOM]-(w2:Wing)
WHERE w2.name <> $wing1
RETURN w2.name, r2.name
```

### Knowledge graph: causal chains
```cypher
MATCH path = (e1:Entity {name: $start})-[:RELATES_TO*1..5]->(e2:Entity {name: $end})
RETURN [n IN nodes(path) | n.name] AS chain,
       [r IN relationships(path) | r.predicate] AS predicates
```

### Find contradictions
```cypher
MATCH (e:Entity)<-[:REFERENCES]-(d1:Drawer),
      (e)<-[:REFERENCES]-(d2:Drawer)
WHERE d1.id <> d2.id
AND d1.content CONTAINS $claim_a
AND d2.content CONTAINS $claim_b
RETURN e.name, d1.content, d2.content
```

### Hot paths (follow proven trails)
```cypher
MATCH ()-[e]->()
WHERE e.success_pheromone > 0.5
RETURN startNode(e).id, endNode(e).id, e.success_pheromone
ORDER BY e.success_pheromone DESC
LIMIT 20
```

### Cold spots (unexplored areas)
```cypher
MATCH (n)
WHERE n.exploration_pheromone < 0.1
AND (n:Room OR n:Entity)
RETURN n.id, n.name, labels(n)[0] AS type
ORDER BY n.exploitation_pheromone DESC
LIMIT 20
```

### Find all entities a drawer references
```cypher
MATCH (d:Drawer {id: $drawer_id})-[:REFERENCES]->(e:Entity)
RETURN e.name, e.entity_type, e.description
```

### Entity timeline (temporal knowledge graph)
```cypher
MATCH (e:Entity {name: $entity})-[r:RELATES_TO]->(other:Entity)
WHERE r.valid_to IS NULL  -- Currently valid
RETURN e.name, r.predicate, other.name, r.confidence, r.observed_at
ORDER BY r.observed_at DESC
```

### Full-text keyword search
```cypher
CALL fts_search('drawer_content_idx', $keyword, 10) YIELD node, score
RETURN node.content, node.source, score
```

### Find similar drawers
```cypher
MATCH (d:Drawer {id: $drawer_id})-[s:SIMILAR_TO]-(other:Drawer)
RETURN other.content, s.similarity
ORDER BY s.similarity DESC
LIMIT 5
```

### Agent diary entries
```cypher
MATCH (a:Agent {name: $agent_name})
RETURN a.diary, a.domain, a.focus
```

### Graph connectivity overview
```cypher
MATCH (n)
WITH labels(n)[0] AS type, count(n) AS cnt
RETURN type, cnt
ORDER BY cnt DESC
```

---

## 28 MCP Tools Reference

### Palace Navigation (read-only)

| Tool | Parameters | Returns | When to Use |
|------|-----------|---------|-------------|
| `palace_status` | — | Palace overview + PALACE_PROTOCOL prompt | **Always call first** — orients you in the palace |
| `list_wings` | — | All wings with room counts | Discovering what domains exist |
| `list_rooms` | `wing_id` | Rooms within a wing | Drilling into a specific domain |
| `get_taxonomy` | — | Full wing→room→closet→drawer tree | Understanding palace structure at a glance |
| `search` | `query`, `wing?`, `room?`, `k?` | Semantic search with pheromone-boosted ranking | **Primary recall** — find memories by meaning |
| `navigate` | `from_id`, `to_id`, `context?` | A\* path with provenance | Finding connections between two ideas |
| `find_tunnels` | `wing_a`, `wing_b` | Rooms connecting two wings | Discovering cross-domain relationships |
| `graph_stats` | — | Graph connectivity overview | Health check — is the palace well-connected? |

### Palace Operations (write)

| Tool | Parameters | Returns | When to Use |
|------|-----------|---------|-------------|
| `add_drawer` | `content`, `wing`, `room`, `source?` | New drawer ID (auto-embedded) | Storing a new memory (always `check_duplicate` first!) |
| `delete_drawer` | `drawer_id` | Confirmation | Removing incorrect or duplicate content |
| `add_wing` | `name`, `type`, `description` | New wing ID | Creating a new domain/project/person area |
| `add_room` | `wing_id`, `name`, `hall_type`, `description?` | New room ID | Adding a subject area within a wing |
| `check_duplicate` | `content`, `threshold?` | Similar existing drawers | **Call before `add_drawer`** — prevent duplicates |

### Knowledge Graph

| Tool | Parameters | Returns | When to Use |
|------|-----------|---------|-------------|
| `kg_add` | `subject`, `predicate`, `object`, `confidence?` | Triple ID | Recording a causal/relational fact |
| `kg_query` | `entity`, `as_of?` | Entity relationships | Looking up what's known about an entity |
| `kg_invalidate` | `subject`, `predicate`, `object` | Confirmation | Marking a relationship as no longer valid |
| `kg_timeline` | `entity` | Chronological story | Understanding an entity's history |
| `kg_traverse` | `start`, `depth?`, `predicate?` | Multi-hop subgraph | Exploring causal chains and networks |
| `kg_contradictions` | `entity` | Conflicting relationships | Fact-checking — finding inconsistencies |

### Stigmergy

| Tool | Parameters | Returns | When to Use |
|------|-----------|---------|-------------|
| `pheromone_status` | `node_id` or `edge` | Current pheromone levels | Checking trail strength on a specific element |
| `pheromone_deposit` | `path`, `reward_type` | Confirmation | **After a useful search** — reinforce the path |
| `hot_paths` | `wing?`, `k?` | Most-traversed/successful paths | Finding well-known connections |
| `cold_spots` | `wing?`, `k?` | Unexplored areas | Finding gaps in knowledge to investigate |
| `decay_now` | — | Force pheromone decay cycle | Manually aging trails (usually automatic) |

### Agent Diary

| Tool | Parameters | Returns | When to Use |
|------|-----------|---------|-------------|
| `list_agents` | — | All specialist agents | Discovering which specialists exist |
| `diary_write` | `agent_id`, `entry` | Confirmation | Recording a specialist's observation |
| `diary_read` | `agent_id`, `last_n?` | Recent diary entries | Getting specialist knowledge and context |

### System

| Tool | Parameters | Returns | When to Use |
|------|-----------|---------|-------------|
| `export` | `format?` | Full palace as JSON/Cypher | Backup, sharing, migration |
| `import` | `data`, `format?` | Load confirmation | Restoring or merging palace data |

---

## Example Workflows

### 🔍 Workflow 1: Recall a Memory

```
1. search(query="how to configure nginx reverse proxy")
   → Returns top-10 drawers by semantic similarity, boosted by pheromones
2. (Read the results)
3. pheromone_deposit(path=[result_path], reward_type="success")
   → Reinforces the path so future searches find it faster
```

### 📝 Workflow 2: Store New Information

```
1. check_duplicate(content="New information about X...")
   → Check if similar content already exists (threshold 0.85 default)
2. IF no duplicates:
   add_drawer(content="Full verbatim text...", wing="projects", room="my-project", source="conversation")
   → Stores with auto-generated embedding
3. kg_add(subject="X", predicate="causes", object="Y", confidence=0.8)
   → Record the relationship in the knowledge graph
```

### 🧭 Workflow 3: Navigate Between Ideas

```
1. navigate(from_id="drawer_123", to_id="entity_456", context="looking for causal link")
   → A* finds optimal path considering semantics, pheromones, and structure
2. (Examine the path: drawer → closet → room → tunnel → room → entity)
3. pheromone_deposit(path=result.path, reward_type="success")
   → Reinforce if the path was useful
```

### 🔬 Workflow 4: Explore Unknown Territory

```
1. cold_spots(wing="science", k=10)
   → Find rooms and entities with low exploration pheromone
2. navigate(from_id=current_location, to_id=cold_spot_id)
   → Chart a path to the unexplored area
3. search(query="related concepts", room=cold_spot_room)
   → Investigate what's there
4. diary_write(agent_id="science-specialist", entry="Explored X, found...")
   → Record findings
```

### ✅ Workflow 5: Verify a Fact

```
1. kg_query(entity="climate change")
   → See all known relationships
2. kg_contradictions(entity="climate change")
   → Find conflicting claims
3. kg_timeline(entity="climate change")
   → See how understanding evolved over time
4. kg_traverse(start="climate change", depth=3, predicate="causes")
   → Trace causal chains
```

### 🌐 Workflow 6: Cross-Domain Discovery

```
1. find_tunnels(wing_a="biology", wing_b="economics")
   → See which rooms bridge these domains
2. navigate(from_id=bio_room, to_id=econ_room, context="cross-domain")
   → A* uses higher semantic weight for cross-domain paths
3. hot_paths(k=5)
   → Check which cross-domain connections are well-established
```

### 🏗️ Workflow 7: Build Palace Structure

```
1. palace_status()
   → Get overview and PALACE_PROTOCOL
2. add_wing(name="quantum-computing", type="domain", description="Quantum computing research")
3. add_room(wing_id=new_wing_id, name="algorithms", hall_type="facts")
4. add_room(wing_id=new_wing_id, name="hardware", hall_type="facts")
5. (Add drawers with content to each room)
```

---

## CLI Quick Reference

The `graphpalace` CLI mirrors the MCP tools above. Global flags go **before** the subcommand:

```
graphpalace -d <palace_dir> [-c config.toml] [-q] <command>
```

| Flag | Short | Default | Purpose |
|------|:-----:|---------|---------|
| `--db` | `-d` | `./palace` | Path to palace database directory |
| `--config` | `-c` | `graphpalace.toml` | Config file path |
| `--quiet` | `-q` | off | Suppress informational messages (e.g. tunnel build counts) |

### Common commands

```bash
# Initialize a new palace
graphpalace -d .palace init --name "My Palace"

# Wing operations
graphpalace -d .palace wing list
graphpalace -d .palace wing add <name> -t domain -d "description"
#   -t/--wing-type: person | project | domain | topic (default: topic)
#   -d/--description: optional wing description

# Room operations
graphpalace -d .palace room list <wing>
graphpalace -d .palace room add <wing> <name> -t facts -d "description"
#   -t/--hall-type: facts | events | discoveries | preferences | advice (default: facts)
#   -d/--description: optional room description

# Add a memory (drawer)
graphpalace -d .palace add-drawer -c "content" -w wing -r room -s conversation
#   -s/--source: conversation | file | api | agent

# Semantic search
graphpalace -d .palace search "query" -k 5

# Knowledge graph
graphpalace -d .palace kg add "subject" "predicate" "object" --confidence 0.8
graphpalace -d .palace kg query "entity"
graphpalace -d .palace kg timeline "entity"

# Navigation & structure
graphpalace -d .palace navigate <from_id> <to_id>
graphpalace -d .palace status --verbose
graphpalace -d .palace export -o backup.json

# Start MCP server (stdin/stdout JSON-RPC)
graphpalace -d .palace serve
```

> **Note:** The MCP tools table above documents the **server-side tool interface** (used via `serve`). The CLI subcommands listed here provide equivalent functionality for direct terminal use.

---

## Key Principles

1. **Search first, ask second.** Always `search` before telling a user you don't know something.
2. **Never summarize drawer content.** Drawers hold verbatim text. Closets hold summaries. Respect the hierarchy.
3. **Check before adding.** Always `check_duplicate` before `add_drawer`. Duplicate content degrades search quality.
4. **Reinforce what works.** After a successful retrieval, `pheromone_deposit` on the path. This is how the palace learns.
5. **Explore cold spots.** Unexplored areas may contain valuable connections. Use `cold_spots` periodically.
6. **Use the knowledge graph.** `kg_add` for facts, `kg_contradictions` for verification, `kg_timeline` for history.
7. **Trust the pheromones.** Hot paths exist for a reason — the swarm has validated them.
8. **Read agent diaries.** Specialists accumulate domain knowledge. Their diaries are condensed expertise.
9. **Context matters for A\*.** Set `context` in `navigate` to get weight-tuned pathfinding.
10. **The palace is alive.** Pheromones decay. Fresh trails matter. The palace reflects what's currently important.
