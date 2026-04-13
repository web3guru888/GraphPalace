# Skills Protocol

The `skills.md` protocol is GraphPalace's approach to teaching any LLM agent how to navigate a memory palace. Instead of relying on tool descriptions alone, GraphPalace provides a structured skills file that can be loaded into an agent's context, giving it immediate fluency in palace navigation.

## Concept

Traditional MCP integrations rely on tool names and parameter descriptions for the LLM to figure out how to use them. This works for simple tools but fails for a system as complex as a memory palace, where effective use requires understanding:

- The spatial hierarchy (wings → rooms → closets → drawers)
- When to use semantic search vs. graph navigation
- How pheromone trails work and when to deposit them
- Cypher query patterns for direct graph access
- The difference between halls and tunnels
- When to check for duplicates before adding content

The skills file solves this by providing **executable knowledge** — Cypher patterns, decision rules, and navigation strategies that the LLM can immediately apply.

## File Location

```
skills/
└── graphpalace.md     # The main skills file
```

## How It Works

### 1. Load at Context Start

The skills file is loaded into the LLM's context at the beginning of a conversation or session:

```python
# Python example
with open("skills/graphpalace.md") as f:
    skills = f.read()

messages = [
    {"role": "system", "content": f"You are a helpful assistant.\n\n{skills}"},
    {"role": "user", "content": user_message},
]
```

### 2. Agent Learns Cypher Patterns

The skills file includes ready-to-use Cypher patterns:

```cypher
-- Find memories by meaning (semantic search)
CALL vector_search('drawer_embedding_idx', $query_embedding, 10) YIELD node, score
RETURN node.content, score

-- Walk the palace hierarchy
MATCH (w:Wing {name: $wing})-[:HAS_ROOM]->(r:Room)-[:HAS_CLOSET]->(c:Closet)-[:HAS_DRAWER]->(d:Drawer)
RETURN r.name, c.summary, d.content
ORDER BY d.exploitation_pheromone DESC

-- Follow halls (same wing connections)
MATCH (r1:Room {name: $room})-[:HALL]->(r2:Room)
RETURN r2.name, r2.hall_type

-- Cross wings via tunnels
MATCH (r1:Room)<-[:HAS_ROOM]-(w1:Wing {name: $wing1}),
      (r1)-[:TUNNEL]-(r2:Room)<-[:HAS_ROOM]-(w2:Wing)
WHERE w2.name <> $wing1
RETURN w2.name, r2.name
```

### 3. Agent Learns Decision Rules

The skills file teaches when to use each tool:

| Task | Recommended Tool |
|------|-----------------|
| Recall a memory | `search` (semantic) or vector search Cypher |
| Find a connection | `navigate` (A*) or multi-hop Cypher |
| Store something | `add_drawer` (always check duplicate first) |
| Reinforce a path | `pheromone_deposit` after successful use |
| Explore unknown | `cold_spots` → navigate → investigate |
| Check a fact | `kg_query` + `kg_contradictions` |

### 4. Agent Understands Pheromones

The skills file explains how to read and use pheromone signals:

- **Hot paths** = edges with `success_pheromone > 0.5` → Follow them
- **Cold spots** = nodes with `exploration_pheromone < 0.1` → Investigate them
- **Valuable areas** = nodes with high `exploitation_pheromone` → Important content
- **Stale areas** = low `recency_pheromone` everywhere → Palace needs refreshing

## Skills File Structure

The skills file follows a consistent structure:

```markdown
# GraphPalace Skills

## You have a memory palace backed by a graph database.

### Core Concepts
[Palace hierarchy, pheromones, A*, agents, verbatim storage]

### The Palace Hierarchy
[ASCII diagram of Palace → Wing → Room → Closet → Drawer]

### Five Pheromone Types
[Table of pheromone types with signals and decay rates]

### Cypher Patterns You Know
[Ready-to-use Cypher queries for common operations]

### When to Use Each Tool
[Decision guide mapping tasks to tools]
```

## Customization

The skills file can be customized for specific use cases:

### Adding Domain-Specific Patterns

If your palace has a specific wing structure, add Cypher patterns for it:

```markdown
### Your Project Patterns

#### Find all decisions about databases
\```cypher
MATCH (w:Wing {name: "project_orion"})-[:HAS_ROOM]->(r:Room {hall_type: "decisions"})
      -[:HAS_CLOSET]->(c)-[:HAS_DRAWER]->(d)
WHERE d.content CONTAINS "database"
RETURN d.content, d.created_at
ORDER BY d.created_at DESC
\```
```

### Adding Agent-Specific Instructions

For specialist agents:

```markdown
### You are the Security Specialist

Your wing: "security_review"
Your focus: vulnerabilities, threat models, audit findings
After every search, deposit exploitation pheromones on useful results.
Keep your diary updated with findings.
```

## Integration with MCP

The `palace_status` MCP tool returns the PALACE_PROTOCOL prompt (a compact version of the skills file) every time it's called. This ensures the LLM always has navigation instructions, even without the full skills file loaded.

The full skills file provides deeper knowledge (Cypher patterns, advanced strategies) while PALACE_PROTOCOL provides essential rules for basic operation.

## Versioning

The skills file includes a version number:

```markdown
> **Protocol version**: 1.0
```

This allows applications to check compatibility and update the skills file when GraphPalace is upgraded.
