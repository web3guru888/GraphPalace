# GraphPalace

**A memory palace backed by a graph database.** GraphPalace is an AI memory system that combines spatial organization (the Method of Loci) with stigmergic pathfinding, active inference agents, and semantic search — all built on top of an embedded graph database.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What Is It?

GraphPalace gives AI agents persistent, navigable memory organized as a **memory palace** — a spatial graph where:

- **Wings** are top-level domains (projects, people, topics)
- **Rooms** are specific subjects within a wing
- **Halls** connect rooms in the same wing
- **Tunnels** connect rooms across wings (same topic, different domain)
- **Closets** are summary containers pointing to drawers
- **Drawers** store verbatim original content (never summarized)
- **Entities** form a knowledge graph of things and relationships
- **Agents** are specialist navigators with persistent diaries

Every node and edge carries **pheromone trails** — signals left by past searches that guide future navigation, just like ants finding the shortest path to food.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    MCP Server                        │
│              28 tools for LLM agents                 │
├──────────┬──────────┬──────────┬───────────┬────────┤
│ gp-core  │gp-stigm. │gp-path  │ gp-agents │gp-embed│
│  Types   │Pheromones│Semantic  │  Active   │  ONNX  │
│  Schema  │  Decay   │  A*      │ Inference │Vectors │
│  Config  │ Rewards  │Heuristic │ Beliefs   │ Search │
├──────────┴──────────┴──────────┴───────────┴────────┤
│              Kuzu Graph Database (C++20)              │
│     Cypher · HNSW Vector Index · FTS · ACID · WASM   │
└─────────────────────────────────────────────────────┘
```

### Rust Crates

| Crate | Description |
|-------|-------------|
| `gp-core` | Core types: node/edge structs, pheromone fields, config, errors, Cypher DDL |
| `gp-stigmergy` | Pheromone system: 5 types, exponential decay, position-weighted rewards, edge cost recomputation |
| `gp-pathfinding` | Semantic A*: composite cost model (40/30/30), adaptive cross/same-domain heuristic |
| `gp-agents` | Active Inference: EFE computation, Bayesian beliefs, softmax action selection, temperature annealing |
| `gp-embeddings` | Embedding engine trait + mock implementation (ONNX integration planned) |
| `gp-mcp` | MCP tool definitions: 28 tool schemas, PALACE_PROTOCOL prompt |
| `gp-wasm` | WASM bindgen stubs (browser deployment planned) |

## Key Algorithms

### Stigmergy (Pheromone System)

Five pheromone types guide navigation through collective intelligence:

| Type | Applied To | Signal | Decay Rate | Half-life |
|------|-----------|--------|------------|-----------|
| Exploitation | Nodes | "This is valuable" | 0.02 | ~35 cycles |
| Exploration | Nodes | "Already searched" | 0.05 | ~14 cycles |
| Success | Edges | "Good outcomes" | 0.01 | ~69 cycles |
| Traversal | Edges | "Frequently used" | 0.03 | ~23 cycles |
| Recency | Edges | "Used recently" | 0.10 | ~7 cycles |

### Semantic A* Pathfinding

Composite edge cost model balances three signals:

```
cost(edge) = 0.4 × semantic + 0.3 × pheromone + 0.3 × structural
```

The heuristic adapts between cross-domain (50/50) and same-domain (90/10) search based on semantic similarity.

### Active Inference Agents

Agents minimize Expected Free Energy (EFE) to decide where to look:

```
EFE = -(epistemic + pragmatic + edge_quality)
```

- **Epistemic**: How much will we learn? (1/precision)
- **Pragmatic**: How close to our goal? (cosine similarity)
- **Edge quality**: Collective intelligence signal (pheromones)

## Building

### Rust Crates (native)

```bash
cd rust
cargo build --release
cargo test --workspace
```

### Kuzu Core (C++ — optional)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Configuration

Copy `graphpalace.toml` to your project and customize:

```toml
[palace]
name = "My Palace"
embedding_model = "all-MiniLM-L6-v2"
embedding_dim = 384

[pheromones]
exploitation_decay = 0.02
exploration_decay = 0.05

[cost_weights]
semantic = 0.4
pheromone = 0.3
structural = 0.3
```

See [`graphpalace.toml`](graphpalace.toml) for all options.

## Skills File

The [`skills/graphpalace.md`](skills/graphpalace.md) file teaches any LLM agent how to navigate the palace. Load it as context to give your agent palace navigation abilities.

## Palace Graph Schema

The palace is a property graph with 7 node types and 11 edge types. See [`rust/gp-core/src/schema.rs`](rust/gp-core/src/schema.rs) for the complete Cypher DDL.

### Node Types
- `Palace` — Top-level container
- `Wing` — Domain grouping (person, project, domain, topic)
- `Room` — Subject within a wing
- `Closet` — Summary container
- `Drawer` — Verbatim content storage
- `Entity` — Knowledge graph node
- `Agent` — Specialist navigator

### Edge Types
- `CONTAINS` — Palace → Wing
- `HAS_ROOM` — Wing → Room
- `HAS_CLOSET` — Room → Closet
- `HAS_DRAWER` — Closet → Drawer
- `HALL` — Room ↔ Room (same wing)
- `TUNNEL` — Room ↔ Room (across wings)
- `RELATES_TO` — Entity ↔ Entity (knowledge graph)
- `REFERENCES` — Drawer → Entity
- `SIMILAR_TO` — Drawer ↔ Drawer (auto-computed)
- `MANAGES` — Agent → Wing
- `INVESTIGATED` — Agent → Drawer

## Research Heritage

GraphPalace builds on:

- **MemPalace** (Jovovich & Sigman, 2026) — Verbatim storage philosophy, 96.6% LongMemEval recall
- **Method of Loci** (Simonides, ~500 BC) — Palace spatial metaphor
- **STAN_X v8** — Stigmergic coordination, Semantic A*, Active Inference agents
- **Kùzu** (Amine et al., 2023-2025) — Embedded graph database with Cypher, vector search, WASM
- **Active Inference** (Karl Friston, 2006+) — EFE minimization, Bayesian beliefs
- **MCP Protocol** (Anthropic, 2024) — Standard LLM ↔ tool communication

## License

MIT — see [LICENSE](LICENSE) for details.

## Roadmap

- [x] Phase 1: Foundation (Rust workspace, core types, schema, config)
- [ ] Phase 2: Kuzu FFI integration
- [ ] Phase 3: Live pheromone system
- [ ] Phase 4: Swarm coordination
- [ ] Phase 5: MCP server implementation
- [ ] Phase 6: WASM browser deployment
- [ ] Phase 7: Distribution (npm, pip, CLI)
