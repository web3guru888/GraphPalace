# GraphPalace

**A Stigmergic Memory Palace Engine for AI Agents**

[![CI](https://github.com/web3guru888/GraphPalace/actions/workflows/graphpalace-ci.yml/badge.svg)](https://github.com/web3guru888/GraphPalace/actions/workflows/graphpalace-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

GraphPalace is an embedded graph database that makes the memory palace metaphor computationally real. It forks [Kùzu](https://github.com/kuzudb/kuzu) — a property graph database with Cypher, native HNSW vector search, full-text search, and WASM bindings — and extends it with stigmergic navigation, spatial hierarchy, semantic A* pathfinding, and Active Inference agents. The result is a **fully local, private, self-optimizing AI memory system** that runs in a browser tab, on a server, or on an edge device — no cloud, no API keys, no data exfiltration.

## What Makes It Different

| System | Storage | Retrieval | Intelligence | Runs Where | Cost |
|--------|---------|-----------|-------------|------------|------|
| MemPalace | ChromaDB (flat vectors) | Cosine similarity + metadata filter | None (passive) | Local Python | Free |
| Mem0 | LLM-extracted facts | LLM retrieval | LLM-dependent | Cloud | $19-249/mo |
| Zep/Graphiti | Neo4j (graph) | Graph traversal | Entity resolution | Cloud | $25+/mo |
| **GraphPalace** | **Property graph + vectors + FTS** | **Stigmergic A\* (semantic+pheromone+structural)** | **Active Inference agents** | **Browser/Edge/Server (WASM)** | **Free** |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         MCP Server                               │
│                    28 tools for LLM agents                       │
├────────────┬────────────┬────────────┬───────────┬──────────────┤
│            │            │            │           │              │
│  gp-core   │gp-stigmergy│gp-pathfind.│ gp-agents │gp-embeddings │
│   Types    │ Pheromones │ Semantic   │  Active   │    ONNX      │
│   Schema   │   Decay    │   A*       │ Inference │   Vectors    │
│   Config   │  Rewards   │ Heuristic  │  Beliefs  │   Search     │
│            │            │            │           │              │
├────────────┴────────────┴────────────┴───────────┴──────────────┤
│                     gp-swarm                                     │
│       Multi-agent coordination · Convergence detection           │
├──────────────────────────────────────────────────────────────────┤
│                 Kuzu Graph Database (C++20)                       │
│        Cypher · HNSW Vector Index · FTS · ACID · WASM            │
└──────────────────────────────────────────────────────────────────┘
```

### Palace Spatial Hierarchy

```
Palace
 └── Wing (domain: "project", "person", "topic")
      ├── Room (subject within the wing)
      │    ├── Hall ──→ Room (same wing connection)
      │    ├── Tunnel ──→ Room (cross-wing connection)
      │    └── Closet (topic summary)
      │         └── Drawer (verbatim memory — NEVER summarized)
      │              └── ──REFERENCES──→ Entity (knowledge graph)
      └── ...more rooms
```

Every node and edge carries **pheromone trails** — signals left by past searches that guide future navigation, just like ants finding the shortest path to food.

## Quick Start

### Build Rust Crates

```bash
git clone https://github.com/web3guru888/GraphPalace.git
cd GraphPalace/rust
cargo build --release
cargo test --workspace
```

### Build WASM Bundle

```bash
cd rust/gp-wasm
wasm-pack build --target web --release
# Output: pkg/graphpalace_bg.wasm + pkg/graphpalace.js
```

### Build Kuzu Core (C++ — optional)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Crate Map

| Crate | Description | Key Types |
|-------|-------------|-----------|
| **[gp-core](rust/gp-core/)** | Core types, palace schema, config, error handling | `Palace`, `Wing`, `Room`, `Drawer`, `Entity`, `PheromoneField` |
| **[gp-stigmergy](rust/gp-stigmergy/)** | 5-type pheromone system with exponential decay | `PheromoneManager`, `DecayEngine`, `RewardCalculator` |
| **[gp-pathfinding](rust/gp-pathfinding/)** | Semantic A* with composite cost model (40/30/30) | `SemanticAStar`, `CostWeights`, `PathResult` |
| **[gp-agents](rust/gp-agents/)** | Active Inference: EFE, Bayesian beliefs, archetypes | `ActiveInferenceAgent`, `BeliefState`, `GenerativeModel` |
| **[gp-swarm](rust/gp-swarm/)** | Multi-agent swarm coordination + convergence detection | `SwarmCoordinator`, `ConvergenceDetector`, `InterestScore` |
| **[gp-embeddings](rust/gp-embeddings/)** | Embedding engine trait + similarity search | `EmbeddingEngine`, `MockEmbeddingEngine` |
| **[gp-mcp](rust/gp-mcp/)** | 28 MCP tool schemas + PALACE_PROTOCOL prompt | `ToolDefinition`, `PalaceProtocol` |
| **[gp-wasm](rust/gp-wasm/)** | WASM bindgen stubs for browser deployment | `WasmPalace` |

## Key Algorithms

### Stigmergy (Pheromone System)

Five pheromone types guide navigation through collective intelligence:

| Type | Applied To | Signal | Decay Rate | Half-life |
|------|-----------|--------|------------|-----------|
| **Exploitation** | Nodes | "This is valuable" | 0.02 | ~35 cycles |
| **Exploration** | Nodes | "Already searched" | 0.05 | ~14 cycles |
| **Success** | Edges | "Good outcomes" | 0.01 | ~69 cycles |
| **Traversal** | Edges | "Frequently used" | 0.03 | ~23 cycles |
| **Recency** | Edges | "Used recently" | 0.10 | ~7 cycles |

Pheromones are deposited position-weighted along successful paths and decay exponentially over time:

```rust
// Deposit: earlier edges in a successful path get larger rewards
let reward = base_reward * (1.0 - position / path_length);

// Decay: exponential with configurable rate
let new_value = current * (1.0 - decay_rate);
```

### Semantic A* Pathfinding

Composite edge cost model balances three signals:

```
cost(edge) = 0.4 × semantic + 0.3 × pheromone + 0.3 × structural
```

The heuristic adapts based on whether search is cross-domain (balanced) or same-domain (trust semantics):

| Context | Semantic (α) | Pheromone (β) | Structural (γ) |
|---------|-------------|---------------|----------------|
| Default | 0.40 | 0.30 | 0.30 |
| Hypothesis Testing | 0.30 | 0.40 | 0.30 |
| Exploratory Research | 0.50 | 0.20 | 0.30 |
| Memory Recall | 0.50 | 0.30 | 0.20 |

### Active Inference Agents

Agents minimize Expected Free Energy (EFE) to decide where to look:

```
EFE = -(epistemic + pragmatic + edge_quality)
```

- **Epistemic**: How much will we learn? (1/precision — high uncertainty = high value)
- **Pragmatic**: How close to our goal? (cosine similarity to goal embedding)
- **Edge quality**: Collective intelligence signal (exploitation - exploration pheromones)

Five agent archetypes support different navigation strategies:

| Archetype | Temperature | Strategy |
|-----------|------------|----------|
| **Explorer** | 1.0 | Pure epistemic — discover new rooms, expand frontier |
| **Exploiter** | 0.1 | Follow proven paths, retrieve known memories |
| **Balanced** | 0.5 | Default — mix exploration and exploitation |
| **Specialist** | 0.3 | Manage a specific wing, keep persistent diary |
| **Generalist** | 0.7 | Cross-wing connections, find tunnels |

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
success_decay = 0.01
traversal_decay = 0.03
recency_decay = 0.10
decay_interval_cycles = 10

[cost_weights]
semantic = 0.4
pheromone = 0.3
structural = 0.3
```

See [`graphpalace.toml`](graphpalace.toml) for all options.

## Skills File

The [`skills/graphpalace.md`](skills/graphpalace.md) file teaches any LLM agent how to navigate the palace. Load it as context to give your agent palace navigation abilities — it includes Cypher patterns, pheromone semantics, and tool usage guidance.

## Performance Targets

| Metric | Target | Basis |
|--------|--------|-------|
| Semantic search (top-10) | <50ms | Kuzu HNSW + cosine |
| A* pathfinding (cached) | <200ms | STAN_X achieves 211ms |
| A* pathfinding (uncached) | <500ms | STAN_X achieves 494ms |
| Embedding generation | <100ms per text | ONNX Runtime |
| Pheromone decay (10k edges) | <500ms | Bulk Cypher update |
| WASM bundle size | <20MB | Kuzu WASM ~10MB + model ~12MB |
| Memory (1M drawers) | <2GB | Kuzu columnar storage |
| Palace wake-up | <200 tokens | MemPalace architecture |

## Documentation

- [Architecture Overview](docs/architecture.md) — System architecture, crate dependencies, data flow
- [Stigmergy System](docs/stigmergy.md) — 5 pheromone types, decay, deposit, cost recomputation
- [Pathfinding](docs/pathfinding.md) — Semantic A*, adaptive heuristic, context weights
- [Active Inference Agents](docs/agents.md) — EFE, beliefs, archetypes, swarm coordination
- [MCP Tools Reference](docs/mcp-tools.md) — All 28 tool descriptions with parameters
- [Palace Schema](docs/palace-schema.md) — Full Cypher DDL schema
- [Skills Protocol](docs/skills-protocol.md) — How the skills.md protocol works for LLM integration

## Research Heritage

GraphPalace stands on the shoulders of:

| Contribution | Source | What We Take |
|---|---|---|
| Verbatim storage philosophy | MemPalace (Jovovich & Sigman, 2026) | Never summarize; store raw, search semantically |
| Palace spatial metaphor | Method of Loci (Simonides, ~500 BC) | Wings/Rooms/Halls/Tunnels as navigation structure |
| 96.6% LongMemEval recall | MemPalace benchmark | Proof that verbatim + embeddings beats extraction |
| Stigmergic coordination | STAN_X v8 (web3guru888, 2026) | 5 pheromone types, position-weighted rewards, decay |
| Semantic A* pathfinding | STAN_X v8 | 40/30/30 composite cost, adaptive heuristic |
| Active Inference agents | Karl Friston (2006+) / STAN_X v8 | EFE minimization, Bayesian beliefs, softmax selection |
| Embedded graph database | Kùzu (Amine et al., 2023-2025) | Cypher, vector search, FTS, WASM, columnar storage |
| WASM microservices | WO 2024/239068 A1 (VBRL Holdings) | Modular edge architecture, sandboxed execution |
| all-MiniLM-L6-v2 | Sentence-Transformers (Reimers & Gurevych) | 384-dim embeddings, proven by both MemPalace and STAN_X |
| MCP protocol | Anthropic (2024) | Standard LLM ↔ tool communication |

## Roadmap

- [x] Phase 1: Foundation — Rust workspace, core types, schema, config
- [x] Phase 2: Stigmergy — 5 pheromone types, decay, deposit, cost recomputation
- [x] Phase 3: Pathfinding — Semantic A*, composite cost, adaptive heuristic
- [x] Phase 4: Agents — Active Inference, beliefs, archetypes, swarm coordination
- [x] Phase 5: MCP + Skills — 28 tool schemas, PALACE_PROTOCOL, skills.md
- [x] Phase 6: WASM — wasm-bindgen stubs, browser deployment prep
- [ ] Phase 7: Distribution — CI/CD, NPM/PyPI packages, CLI, docs site
- [ ] Phase 8: Kuzu FFI — Connect Rust crates to Kuzu C++ engine
- [ ] Phase 9: Live Palace — End-to-end working memory palace
- [ ] Phase 10: Benchmarks — vs MemPalace (96.6% recall), STAN_X (90.9% A*)

## License

MIT — see [LICENSE](LICENSE) for details.
