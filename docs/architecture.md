# Architecture Overview

GraphPalace is a layered system built on top of the Kuzu embedded graph database. The Rust crate layer implements the memory palace semantics — stigmergy, pathfinding, agents, and MCP tooling — while Kuzu provides the graph storage, Cypher query engine, vector indexes, and WASM compilation target.

## System Layers

```
┌─────────────────────────────────────────────────┐
│              Application Layer                   │
│  MCP Server │ CLI │ Python bindings │ WASM/JS   │
├─────────────────────────────────────────────────┤
│              Intelligence Layer                   │
│  gp-agents (Active Inference)                    │
│  gp-swarm  (Multi-agent coordination)            │
├─────────────────────────────────────────────────┤
│              Navigation Layer                     │
│  gp-pathfinding (Semantic A*)                    │
│  gp-stigmergy  (Pheromone system)                │
├─────────────────────────────────────────────────┤
│              Foundation Layer                     │
│  gp-core       (Types, schema, config)           │
│  gp-embeddings (Semantic vectors)                │
├─────────────────────────────────────────────────┤
│              Storage Layer                        │
│  Kuzu (C++20)                                    │
│  Cypher · HNSW · FTS · ACID · Columnar · WASM   │
└─────────────────────────────────────────────────┘
```

## Crate Dependency Graph

```
gp-wasm ──→ gp-mcp ──→ gp-agents ──→ gp-pathfinding ──→ gp-stigmergy ──→ gp-core
                    │            │                    │                │
                    │            └──→ gp-embeddings   │                │
                    │                                  └──→ gp-core    │
                    └──→ gp-swarm ──→ gp-agents                       │
                                  └──→ gp-stigmergy                   │
                                                                      │
                                          gp-embeddings ──→ gp-core ──┘
```

### Dependency Summary

| Crate | Depends On | Purpose |
|-------|-----------|---------|
| `gp-core` | (none — leaf crate) | Types, schema DDL, config, error types |
| `gp-stigmergy` | `gp-core` | Pheromone system, decay, cost recomputation |
| `gp-embeddings` | `gp-core` | Embedding engine trait, similarity functions |
| `gp-pathfinding` | `gp-core`, `gp-stigmergy` | Semantic A*, composite cost, heuristic |
| `gp-agents` | `gp-core`, `gp-pathfinding`, `gp-embeddings` | Active Inference, beliefs, archetypes |
| `gp-swarm` | `gp-core`, `gp-agents`, `gp-stigmergy` | Multi-agent coordinator, convergence |
| `gp-mcp` | `gp-core`, `gp-agents`, `gp-swarm` | 28 MCP tool definitions, PALACE_PROTOCOL |
| `gp-wasm` | `gp-mcp` | WASM bindgen entry point |

## Data Flow

### Memory Storage (Write Path)

```
User Content
    │
    ▼
┌──────────────┐
│ gp-embeddings│ ── Encode text → 384-dim vector
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ gp-core      │ ── Create Drawer node with content + embedding
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Kuzu         │ ── INSERT into Drawer table
│              │ ── Build HNSW vector index entry
│              │ ── Build FTS index entry
└──────────────┘
```

### Memory Retrieval (Read Path — Semantic Search)

```
Search Query
    │
    ▼
┌──────────────┐
│ gp-embeddings│ ── Encode query → 384-dim vector
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Kuzu HNSW    │ ── Vector similarity search → top-k candidates
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ gp-stigmergy │ ── Boost scores by exploitation pheromone
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ gp-stigmergy │ ── Deposit pheromones on accessed nodes/edges
└──────────────┘
```

### Memory Retrieval (Read Path — A* Navigation)

```
Start Node, Goal Node
    │
    ▼
┌────────────────┐
│ gp-pathfinding │ ── Semantic A* with composite cost:
│                │    0.4 × semantic + 0.3 × pheromone + 0.3 × structural
└──────┬─────────┘
       │
       ├── gp-embeddings: cosine similarity for semantic cost
       ├── gp-stigmergy: read pheromone levels for pheromone cost
       └── gp-core: relation type weights for structural cost
       │
       ▼
┌──────────────┐
│ PathResult   │ ── Path + cost + iterations + provenance
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ gp-stigmergy │ ── Deposit success pheromones along path
└──────────────┘
```

### Agent Exploration Cycle

```
┌──────────────────────────────────────────────┐
│              gp-swarm Coordinator             │
│                                              │
│  for each cycle:                             │
│    1. SENSE — Get frontier nodes             │
│    2. SCORE — Compute interest scores        │
│    3. DECIDE — Each agent selects via EFE    │
│    4. ACT — Navigate to selected nodes       │
│    5. UPDATE — Deposit pheromones            │
│    6. DECAY — Every N cycles, decay all      │
│    7. CHECK — Convergence? Stop if 2/3 met   │
│                                              │
│  Agent types: Explorer, Exploiter, Balanced, │
│               Specialist, Generalist         │
└──────────────────────────────────────────────┘
```

## Storage Backends

| Backend | Target | Storage |
|---------|--------|---------|
| **File System** | Native (CLI, server) | Standard Kuzu database directory |
| **IndexedDB** | Browser WASM | Via Emscripten FS or OPFS |
| **OPFS** | Browser WASM (preferred) | Origin Private File System |
| **Memory-only** | Testing, ephemeral | In-memory database |

## Build Targets

| Target | Toolchain | Output | Size |
|--------|-----------|--------|------|
| **Browser WASM** | wasm-pack + wasm-bindgen | `.wasm` + JS | <20MB |
| **Node.js** | napi-rs or native addon | npm `@graphpalace/core` | Native |
| **Native CLI** | cargo build | `graphpalace` binary | Native |
| **Python** | PyO3 + maturin | `pip install graphpalace` | Native |
| **Edge (WASI)** | wasm32-wasi | Standalone WASM module | <20MB |

## Key Design Principles

1. **Verbatim storage** — Drawers store original content, never summarized. Closets hold summaries; drawers hold truth. (From MemPalace's insight that verbatim + embeddings beats extraction for recall.)

2. **Collective intelligence** — Pheromone trails encode the swarm's accumulated navigation experience. Paths that lead to useful results get reinforced; unused paths decay. No central planning needed.

3. **Spatial organization** — The palace hierarchy (Wings → Rooms → Closets → Drawers) isn't metadata tagging — it's first-class graph structure that A* can navigate through.

4. **Active Inference** — Agents don't just follow rules. They maintain Bayesian beliefs about the palace and choose actions that minimize Expected Free Energy — balancing exploration and exploitation.

5. **Fully local** — No cloud, no API keys, no data exfiltration. The entire system runs on-device, including embeddings (ONNX/WASM) and graph storage (Kuzu).
