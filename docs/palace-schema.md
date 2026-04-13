# Palace Graph Schema

GraphPalace uses a property graph schema with 7 node types and 11 edge types, designed to represent the memory palace hierarchy, knowledge graph, and agent system. All schema definitions are in `gp-core/src/schema.rs`.

## Node Types

### Palace

The top-level container. A database has exactly one Palace node.

```cypher
CREATE NODE TABLE Palace(
    id STRING PRIMARY KEY,
    name STRING,
    description STRING,
    created_at TIMESTAMP DEFAULT current_timestamp()
)
```

### Wing

Top-level domain groupings within the palace. Each wing represents a major category of knowledge.

```cypher
CREATE NODE TABLE Wing(
    id STRING PRIMARY KEY,
    name STRING,
    wing_type STRING,          -- "person", "project", "domain", "topic"
    description STRING,
    embedding FLOAT[384],      -- Wing-level semantic embedding
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT current_timestamp()
)
```

### Room

Subjects within a wing. Rooms connect to each other via Halls (same wing) and Tunnels (cross wing).

```cypher
CREATE NODE TABLE Room(
    id STRING PRIMARY KEY,
    name STRING,
    hall_type STRING,          -- "facts", "events", "discoveries", "preferences", "advice"
    description STRING,
    embedding FLOAT[384],
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT current_timestamp()
)
```

### Closet

Summary containers that organize drawers within a room. Closets hold compressed summaries of their contained drawers.

```cypher
CREATE NODE TABLE Closet(
    id STRING PRIMARY KEY,
    name STRING,
    summary STRING,            -- Compressed summary of contained drawers
    embedding FLOAT[384],
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    drawer_count INT64 DEFAULT 0,
    created_at TIMESTAMP DEFAULT current_timestamp()
)
```

### Drawer

**The fundamental memory unit.** Drawers store verbatim original content — never summarized. This is MemPalace's key insight: raw storage with semantic search beats extraction-based approaches for recall (96.6% on LongMemEval).

```cypher
CREATE NODE TABLE Drawer(
    id STRING PRIMARY KEY,
    content STRING,            -- Verbatim text (NEVER summarized)
    embedding FLOAT[384],      -- all-MiniLM-L6-v2 (384-dim)
    source STRING,             -- "conversation", "file", "api", "agent"
    source_file STRING,        -- Original file path or URL
    importance FLOAT DEFAULT 0.5,
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT current_timestamp(),
    accessed_at TIMESTAMP DEFAULT current_timestamp()
)
```

### Entity

Knowledge graph nodes representing real-world concepts, people, events, and things.

```cypher
CREATE NODE TABLE Entity(
    id STRING PRIMARY KEY,
    name STRING,
    entity_type STRING,        -- "person", "concept", "event", "place", "organization"
    description STRING,
    embedding FLOAT[384],
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT current_timestamp()
)
```

### Agent

Specialist navigators with persistent diaries and domain expertise.

```cypher
CREATE NODE TABLE Agent(
    id STRING PRIMARY KEY,
    name STRING,
    domain STRING,
    focus STRING,              -- What this agent pays attention to
    diary STRING,              -- AAAK-compressed persistent diary
    goal_embedding FLOAT[384], -- Agent's domain embedding
    temperature FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT current_timestamp()
)
```

## Edge Types

### Palace Hierarchy (Structural)

```cypher
CREATE REL TABLE CONTAINS(FROM Palace TO Wing)

CREATE REL TABLE HAS_ROOM(FROM Wing TO Room,
    base_cost FLOAT DEFAULT 0.3,
    current_cost FLOAT DEFAULT 0.3,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)

CREATE REL TABLE HAS_CLOSET(FROM Room TO Closet,
    base_cost FLOAT DEFAULT 0.3,
    current_cost FLOAT DEFAULT 0.3,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)

CREATE REL TABLE HAS_DRAWER(FROM Closet TO Drawer,
    base_cost FLOAT DEFAULT 0.3,
    current_cost FLOAT DEFAULT 0.3,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)
```

### Palace Cross-Connections (Navigational)

```cypher
-- Within same wing (cheap to traverse)
CREATE REL TABLE HALL(FROM Room TO Room,
    hall_type STRING,          -- "facts", "events", "discoveries", etc.
    base_cost FLOAT DEFAULT 0.5,
    current_cost FLOAT DEFAULT 0.5,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)

-- Across wings (more expensive, but enables cross-domain discovery)
CREATE REL TABLE TUNNEL(FROM Room TO Room,
    base_cost FLOAT DEFAULT 0.7,
    current_cost FLOAT DEFAULT 0.7,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)
```

### Knowledge Graph (Semantic)

```cypher
CREATE REL TABLE RELATES_TO(FROM Entity TO Entity,
    predicate STRING,          -- "causes", "inhibits", "correlates_with", etc.
    confidence FLOAT DEFAULT 0.5,
    base_cost FLOAT DEFAULT 1.0,
    current_cost FLOAT DEFAULT 1.0,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,        -- NULL = currently valid
    observed_at TIMESTAMP DEFAULT current_timestamp()
)
```

### Memory ↔ Entity Connections

```cypher
CREATE REL TABLE REFERENCES(FROM Drawer TO Entity,
    relevance FLOAT DEFAULT 1.0,
    base_cost FLOAT DEFAULT 0.5,
    current_cost FLOAT DEFAULT 0.5,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)
```

### Semantic Similarity (Auto-Computed)

```cypher
CREATE REL TABLE SIMILAR_TO(FROM Drawer TO Drawer,
    similarity FLOAT,
    base_cost FLOAT,           -- 1.0 - similarity
    current_cost FLOAT,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)
```

### Agent ↔ Palace Connections

```cypher
CREATE REL TABLE MANAGES(FROM Agent TO Wing)

CREATE REL TABLE INVESTIGATED(FROM Agent TO Drawer,
    result STRING,             -- "useful", "irrelevant", "contradicts"
    investigated_at TIMESTAMP DEFAULT current_timestamp()
)
```

## Indexes

```cypher
-- Vector indexes (HNSW) for semantic search
CREATE VECTOR INDEX drawer_embedding_idx ON Drawer(embedding)
    WITH (metric='cosine', M=16, ef_construction=200)
CREATE VECTOR INDEX entity_embedding_idx ON Entity(embedding)
    WITH (metric='cosine', M=16, ef_construction=200)
CREATE VECTOR INDEX room_embedding_idx ON Room(embedding)
    WITH (metric='cosine', M=16, ef_construction=200)

-- Full-text indexes for keyword search
CREATE FTS INDEX drawer_content_idx ON Drawer(content)
CREATE FTS INDEX entity_name_idx ON Entity(name, description)

-- Property indexes for efficient filtering
CREATE INDEX wing_name_idx ON Wing(name)
CREATE INDEX room_hall_idx ON Room(hall_type)
CREATE INDEX entity_type_idx ON Entity(entity_type)
CREATE INDEX rel_valid_idx ON RELATES_TO(valid_from, valid_to)
```

## Relation Type Weight Table

Used by the structural cost component of Semantic A*:

| Relation | Base Cost | Category |
|----------|-----------|----------|
| `CONTAINS` | 0.2 | Palace hierarchy (cheapest) |
| `HAS_ROOM` | 0.3 | Palace hierarchy |
| `HAS_CLOSET` | 0.3 | Palace hierarchy |
| `HAS_DRAWER` | 0.3 | Palace hierarchy |
| `SIMILAR_TO` | 0.4 | Semantic similarity |
| `HALL` | 0.5 | Same-wing navigation |
| `REFERENCES` | 0.5 | Memory ↔ Entity |
| `TUNNEL` | 0.7 | Cross-wing navigation |
| `RELATES_TO` | 0.8 | Knowledge graph (default) |
| `instance_of` | 0.3 | Knowledge graph (specific) |
| `subclass_of` | 0.3 | Knowledge graph (specific) |
| `causes` / `inhibits` | 0.6 | Causal relationships |
| `correlates_with` | 0.7 | Statistical relationships |
| `DEFAULT` | 1.0 | Unknown relations |
