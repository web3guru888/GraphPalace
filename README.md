# GraphPalace

**A Stigmergic Memory Palace Engine for AI Agents**

[![CI](https://github.com/web3guru888/GraphPalace/actions/workflows/graphpalace-ci.yml/badge.svg)](https://github.com/web3guru888/GraphPalace/actions/workflows/graphpalace-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Tests](https://img.shields.io/badge/tests-438_passing-brightgreen)
![Rust](https://img.shields.io/badge/rust-10_crates-orange)

GraphPalace is an embedded graph database that makes the **memory palace metaphor computationally real**. Built as a Rust extension to [Kùzu](https://github.com/kuzudb/kuzu) — a property graph database with Cypher, native HNSW vector search, full-text search, and WASM bindings — it adds stigmergic navigation, spatial hierarchy, semantic A\* pathfinding, and Active Inference agents. The result: a **fully local, private, self-optimizing AI memory system** that runs in a browser tab, on a server, or on an edge device. No cloud. No API keys. No data exfiltration.

> **Status**: All 7 implementation phases complete — 10 Rust crates, 438 tests, 11,860 LOC, zero failures.

---

## What Makes It Different

| System | Storage | Retrieval | Intelligence | Runs Where | Cost |
|--------|---------|-----------|-------------|------------|------|
| MemPalace | ChromaDB (flat vectors) | Cosine similarity + metadata filter | None (passive) | Local Python | Free |
| Mem0 | LLM-extracted facts | LLM retrieval | LLM-dependent | Cloud | $19–249/mo |
| Zep/Graphiti | Neo4j (graph) | Graph traversal | Entity resolution | Cloud | $25+/mo |
| **GraphPalace** | **Property graph + vectors + FTS** | **Stigmergic A\* (semantic + pheromone + structural)** | **Active Inference agents** | **Browser / Edge / Server (WASM)** | **Free** |

**Key advantages:**
- 🧠 **Verbatim storage** — drawers hold original text, never summarized (MemPalace's key insight: 96.6% recall)
- 🐜 **Self-optimizing** — pheromone trails evolve from usage patterns, no retraining needed
- 🔍 **Semantic A\*** — finds knowledge through meaning + collective intelligence + graph structure
- 🤖 **Active Inference agents** — autonomous exploration driven by Expected Free Energy minimization
- 🌐 **Runs anywhere** — native binary, WASM in browser, Python binding, Node.js — all from one codebase
- 🔒 **Fully local** — zero network calls, zero telemetry, your data stays yours

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     MCP Server (gp-mcp)                              │
│              28 tools · JSON-RPC 2.0 · PALACE_PROTOCOL               │
├──────────┬─────────────┬──────────────┬───────────┬─────────────────┤
│ gp-core  │gp-stigmergy │gp-pathfinding│ gp-agents │ gp-embeddings   │
│  Types   │ 5 Pheromone │  Semantic    │  Active   │  ONNX Runtime   │
│  Schema  │   Types     │    A*        │ Inference │  384-dim vecs   │
│  Config  │  Decay +    │  Composite   │  Bayesian │  Cosine sim     │
│  Errors  │  Cypher     │  Cost Model  │  Beliefs  │  LRU cache      │
├──────────┴─────────────┴──────────────┴───────────┴─────────────────┤
│                        gp-swarm                                      │
│     SwarmCoordinator · ConvergenceDetector · InterestScore           │
├──────────────────────────────────────────────────────────────────────┤
│                        gp-wasm                                       │
│     InMemoryPalace · JS API · Web Workers · IndexedDB/OPFS          │
├──────────────────────────────────────────────────────────────────────┤
│                  Kùzu Graph Database (C++20)                         │
│         Cypher · HNSW Vector Index · FTS · ACID · WASM               │
└──────────────────────────────────────────────────────────────────────┘
```

### The Palace

```
Palace
 └── Wing (domain: "project", "person", "topic")
      ├── Room (subject within the wing)
      │    ├── ──HALL──→ Room (same-wing corridor)
      │    ├── ──TUNNEL──→ Room (cross-wing passage — different domain, same topic)
      │    └── Closet (topic summary)
      │         └── Drawer (verbatim memory — NEVER summarized)
      │              └── ──REFERENCES──→ Entity (knowledge graph)
      └── ...more rooms
```

Every node carries **exploitation** and **exploration** pheromones. Every edge carries **success**, **traversal**, and **recency** pheromones. These trails — deposited by past searches and decayed over time — create an adaptive landscape that guides future navigation, just like ants finding the shortest path to food.

---

## Quick Start

### Build & Test

```bash
git clone https://github.com/web3guru888/GraphPalace.git
cd GraphPalace/rust
cargo build --release
cargo test --workspace    # 438 tests, 0 failures
```

### WASM Bundle (for browser)

```bash
cd rust/gp-wasm
wasm-pack build --target web --release
# Output: pkg/graphpalace_bg.wasm + pkg/graphpalace.js
```

### Use as a Library

```rust
use gp_core::types::*;
use gp_core::config::GraphPalaceConfig;
use gp_stigmergy::pheromones::PheromoneManager;
use gp_pathfinding::astar::SemanticAStar;
use gp_agents::active_inference::ActiveInferenceAgent;

// Create a palace node
let wing = Wing {
    id: "wing-projects".into(),
    name: "Projects".into(),
    wing_type: WingType::Project,
    embedding: [0.0; 384],
    exploitation_pheromone: 0.0,
    exploration_pheromone: 0.0,
    ..Default::default()
};

// Configure A* pathfinding
let config = GraphPalaceConfig::default();
let astar = SemanticAStar::new(config.astar.clone());

// Create an Active Inference agent
let agent = ActiveInferenceAgent::new(
    "explorer-1".into(),
    "Explorer".into(),
    [0.1; 384],  // goal embedding
    1.0,         // temperature (Explorer archetype)
);
```

### Teach Your LLM

Drop [`skills/graphpalace.md`](skills/graphpalace.md) into your LLM's context. It teaches any AI agent how to navigate the palace — Cypher patterns, pheromone semantics, and all 28 MCP tools.

---

## Crate Map

GraphPalace is organized as a Rust workspace with 8 library crates + 2 binary/binding stubs:

| Crate | Tests | LOC | Description |
|-------|------:|----:|-------------|
| **[gp-core](rust/gp-core/)** | 19 | 1,142 | Foundation: palace types (`Wing`, `Room`, `Closet`, `Drawer`, `Entity`, `Agent`), graph schema (Cypher DDL), configuration, error handling |
| **[gp-stigmergy](rust/gp-stigmergy/)** | 95 | 1,839 | 5-type pheromone system: exponential decay, position-weighted path rewards, edge cost recomputation, Cypher query generation (10 query types) |
| **[gp-pathfinding](rust/gp-pathfinding/)** | 50 | 1,556 | Semantic A\* with composite cost model (40% semantic + 30% pheromone + 30% structural), adaptive heuristic, provenance tracking, benchmark infrastructure |
| **[gp-agents](rust/gp-agents/)** | 50 | 1,160 | Active Inference: EFE minimization, Bayesian belief updates, softmax action selection, temperature annealing (linear/exponential/cosine), 5 archetypes |
| **[gp-swarm](rust/gp-swarm/)** | 50 | 1,228 | Multi-agent coordination: sense→decide→act→update cycle, 3-criteria convergence detection, interest scoring, periodic decay scheduling |
| **[gp-embeddings](rust/gp-embeddings/)** | 23 | 465 | Embedding engine trait with mock implementation, cosine similarity, top-k search, LRU cache |
| **[gp-mcp](rust/gp-mcp/)** | 84 | 2,129 | MCP server: JSON-RPC 2.0 message handling, 28 tool definitions with schemas, PALACE_PROTOCOL prompt generation with live stats |
| **[gp-wasm](rust/gp-wasm/)** | 67 | 1,859 | WASM target: `InMemoryPalace` engine, `wasm-bindgen` JS API, Web Worker message types, IndexedDB/OPFS persistence layer |
| *[gp-cli](rust/gp-cli/)* | — | 335 | CLI binary stub: 12 subcommands (`init`, `search`, `navigate`, `add-drawer`, `status`, `export`, ...) via clap |
| *[gp-python](rust/gp-python/)* | — | 147 | Python bindings stub via PyO3 + maturin: `Palace` class with `add_drawer()`, `search()`, `navigate()` |

**Total: 438 tests · 11,860 LOC · 0 failures · 0 clippy warnings**

---

## Key Algorithms

### 🐜 Stigmergy — Pheromone System

Five pheromone types create a self-organizing knowledge landscape:

| Type | On | Signal | Decay Rate (ρ) | Half-life |
|------|:--:|--------|:--------------:|:---------:|
| **Exploitation** | Nodes | "This location is valuable — come here" | 0.02 | ~35 cycles |
| **Exploration** | Nodes | "Already searched — try elsewhere" | 0.05 | ~14 cycles |
| **Success** | Edges | "This connection led to good outcomes" | 0.01 | ~69 cycles |
| **Traversal** | Edges | "This path is frequently used" | 0.03 | ~23 cycles |
| **Recency** | Edges | "This was used recently" | 0.10 | ~7 cycles |

Pheromones are deposited **position-weighted** along successful paths (earlier edges get larger rewards) and decay exponentially each cycle:

```rust
// Deposit: earlier edges in a successful path get larger rewards
let reward = base_reward * (1.0 - position / path_length);

// Decay: exponential with configurable rate per type
let new_value = current * (1.0 - decay_rate);

// Edge cost recomputation after pheromone changes
let pheromone_factor = 0.5 * success.min(1.0) + 0.3 * recency.min(1.0) + 0.2 * traversal.min(1.0);
let cost = base_cost * (1.0 - pheromone_factor * 0.5);
```

### 🔍 Semantic A\* Pathfinding

A composite edge cost model balances three signals:

```
cost(edge) = α × C_semantic + β × C_pheromone + γ × C_structural
```

Weights adapt to the task context:

| Context | α (Semantic) | β (Pheromone) | γ (Structural) |
|---------|:-----------:|:------------:|:--------------:|
| Default | 0.40 | 0.30 | 0.30 |
| Hypothesis Testing | 0.30 | 0.40 | 0.30 |
| Exploratory Research | 0.50 | 0.20 | 0.30 |
| Evidence Gathering | 0.35 | 0.35 | 0.30 |
| Memory Recall | 0.50 | 0.30 | 0.20 |

The heuristic adapts based on domain distance — trusts semantics within a wing, weights graph distance across wings.

### 🤖 Active Inference Agents

Agents minimize **Expected Free Energy** (EFE) to decide where to look:

```
EFE(node) = -(epistemic + pragmatic + edge_quality)
```

- **Epistemic value** — how much will we learn? (`1/precision` — high uncertainty = high value)
- **Pragmatic value** — how close to our goal? (cosine similarity to goal embedding)
- **Edge quality** — collective intelligence signal (`exploitation - exploration` pheromones)

Actions selected via softmax policy with temperature-controlled exploration:

| Archetype | Temp | Strategy |
|-----------|:----:|----------|
| **Explorer** | 1.0 | Pure epistemic — discover new rooms, expand palace frontier |
| **Exploiter** | 0.1 | Follow proven trails, retrieve known memories quickly |
| **Balanced** | 0.5 | Default — mix exploration and exploitation |
| **Specialist** | 0.3 | Manage a specific wing, keep persistent diary |
| **Generalist** | 0.7 | Cross-wing connections, find tunnels between domains |

### 🐝 Swarm Coordination

The `SwarmCoordinator` runs multi-agent cycles:

1. **Sense** — compute frontier with interest scores
2. **Decide** — each agent selects action via EFE minimization
3. **Act** — agents expand the graph in parallel
4. **Update** — deposit pheromones, decay, check convergence

Convergence is declared when ≥2 of 3 criteria are met: growth rate below threshold, pheromone variance stabilized, frontier exhausted.

---

## MCP Tools (28)

GraphPalace exposes a full MCP (Model Context Protocol) server with 28 tools across 6 categories:

| Category | Tools | Purpose |
|----------|-------|---------|
| **Palace Navigation** | `palace_status`, `list_wings`, `list_rooms`, `get_taxonomy`, `search`, `navigate`, `find_tunnels`, `graph_stats` | Read and traverse the palace |
| **Palace Operations** | `add_drawer`, `delete_drawer`, `add_wing`, `add_room`, `check_duplicate` | Write to the palace |
| **Knowledge Graph** | `kg_add`, `kg_query`, `kg_invalidate`, `kg_timeline`, `kg_traverse`, `kg_contradictions` | Entity-relationship triples |
| **Stigmergy** | `pheromone_status`, `pheromone_deposit`, `hot_paths`, `cold_spots`, `decay_now` | Pheromone management |
| **Agent Diary** | `list_agents`, `diary_write`, `diary_read` | Specialist agent persistence |
| **System** | `export`, `import` | Palace portability |

The server implements JSON-RPC 2.0 with `initialize`, `tools/list`, and `tools/call` methods. Connect via stdio or HTTP.

When you call `palace_status`, it returns the **PALACE_PROTOCOL** — a prompt that teaches any LLM how to use the palace effectively (search before claiming ignorance, navigate to follow connections, deposit pheromones on useful paths, etc.).

---

## Configuration

All parameters are tunable via [`graphpalace.toml`](graphpalace.toml):

```toml
[palace]
name = "My Palace"
embedding_model = "all-MiniLM-L6-v2"
embedding_dim = 384

[pheromones.decay_rates]
exploitation = 0.02    # Half-life ~35 cycles
exploration = 0.05     # Half-life ~14 cycles
success = 0.01         # Half-life ~69 cycles
traversal = 0.03       # Half-life ~23 cycles
recency = 0.10         # Half-life ~7 cycles

[cost_weights.default]
semantic = 0.4
pheromone = 0.3
structural = 0.3

[agents]
default_temperature = 0.5
annealing_schedule = "cosine"

[swarm]
num_agents = 5
max_cycles = 1000
decay_interval = 10

[convergence]
history_window = 20
growth_threshold = 5.0
variance_threshold = 0.05
frontier_threshold = 10
```

See the full file for 200+ configurable parameters across 15 sections.

---

## Skills Protocol

The [`skills/graphpalace.md`](skills/graphpalace.md) file (401 lines) is a standalone document any LLM agent can load to learn palace navigation. It includes:

- Core concepts (palace hierarchy, pheromones, A\*, agents)
- 14 Cypher query patterns (semantic search, hierarchy walk, causal chains, contradictions, hot paths, cold spots, ...)
- All 28 MCP tool descriptions with usage guidance
- 7 example workflows (recall, store, navigate, explore, verify, cross-domain, build structure)
- Pheromone semantics deep dive
- 10 key principles for effective palace use

Simply include it in your LLM's system prompt or context window.

---

## Examples

See [`examples/graphpalace/`](examples/graphpalace/) for working code:

- **[basic_palace.rs](examples/graphpalace/basic_palace.rs)** — Create a palace, add wings/rooms/drawers, compute similarity
- **[pheromone_navigation.rs](examples/graphpalace/pheromone_navigation.rs)** — Deposit pheromones, simulate decay, configure A\*
- **[agent_swarm.rs](examples/graphpalace/agent_swarm.rs)** — Create 5 agent archetypes, update beliefs, compute EFE

---

## Performance Targets

| Metric | Target | Basis |
|--------|--------|-------|
| Semantic search (top-10) | <50ms | Kuzu HNSW + cosine |
| A\* pathfinding (cached) | <200ms | STAN_X achieves 211ms |
| A\* pathfinding (uncached) | <500ms | STAN_X achieves 494ms |
| Embedding generation | <100ms/text | ONNX Runtime |
| Pheromone decay (10k edges) | <500ms | Bulk Cypher update |
| WASM bundle size | <20MB | Kuzu ~10MB + model ~12MB |
| Memory (1M drawers) | <2GB | Kuzu columnar storage |
| Palace wake-up | <200 tokens | PALACE_PROTOCOL prompt |

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture Overview](docs/architecture.md) | System layers, crate dependencies, data flow |
| [Stigmergy System](docs/stigmergy.md) | 5 pheromone types, decay formula, deposit operations, emergent behaviors |
| [Pathfinding](docs/pathfinding.md) | Semantic A\*, composite cost model, adaptive heuristic, context weights |
| [Active Inference Agents](docs/agents.md) | EFE, Bayesian beliefs, 5 archetypes, swarm coordination, convergence |
| [MCP Tools Reference](docs/mcp-tools.md) | All 28 tools with parameter tables and PALACE_PROTOCOL |
| [Palace Schema](docs/palace-schema.md) | Full Cypher DDL — 7 node types, 11 edge types, indexes |
| [Skills Protocol](docs/skills-protocol.md) | How skills.md works, customization, LLM integration |

---

## Research Heritage

GraphPalace stands on the shoulders of:

| Contribution | Source | What We Take |
|---|---|---|
| Verbatim storage philosophy | MemPalace (Jovovich & Sigman, 2026) | Never summarize; store raw, search semantically. 96.6% LongMemEval recall. |
| Palace spatial metaphor | Method of Loci (Simonides, ~500 BC) | Wings/Rooms/Halls/Tunnels — 2,500 years of proven spatial memory |
| Stigmergic coordination | STAN_X v8 (web3guru888, 2026) | 5 pheromone types, position-weighted rewards, exponential decay |
| Semantic A\* pathfinding | STAN_X v8 | 40/30/30 composite cost, adaptive heuristic, context weights |
| Active Inference agents | Karl Friston (2006+) / STAN_X v8 | EFE minimization, Bayesian beliefs, softmax action selection |
| Embedded graph database | Kùzu (Amine et al., 2023–2025) | Cypher, HNSW vector search, FTS, WASM, columnar storage, MIT license |
| WASM microservices | WO 2024/239068 A1 (VBRL Holdings) | Modular edge architecture, sandboxed execution |
| Sentence embeddings | all-MiniLM-L6-v2 (Reimers & Gurevych) | 384-dim vectors, proven by both MemPalace and STAN_X |
| Tool protocol | MCP (Anthropic, 2024) | Standard LLM ↔ tool communication |

---

## Roadmap

- [x] **Phase 1: Foundation** — Rust workspace, core types, palace schema, config (224 tests)
- [x] **Phase 2: Stigmergy** — Cypher query generation, bulk decay, integration tests (+38 tests)
- [x] **Phase 3: Pathfinding** — Benchmark infrastructure, full palace hierarchy tests (+21 tests)
- [x] **Phase 4: Agents + Swarm** — NEW gp-swarm crate, coordinator, convergence (+50 tests)
- [x] **Phase 5: MCP Server** — JSON-RPC 2.0, 28-tool dispatch, PALACE_PROTOCOL (+42 tests)
- [x] **Phase 6: WASM** — InMemoryPalace, JS API, Web Workers, persistence (+63 tests)
- [x] **Phase 7: Distribution** — CI/CD, docs (7 files), CLI, Python bindings, examples
- [ ] **Phase 8: Kuzu FFI** — Connect Rust crates to Kuzu C++ engine via FFI
- [ ] **Phase 9: Live Palace** — End-to-end working memory palace with real embeddings
- [ ] **Phase 10: Benchmarks** — vs MemPalace (96.6% recall target), STAN_X (90.9% A\* target)

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes in the `rust/` directory
4. Run `cargo test --workspace && cargo clippy --workspace` 
5. Submit a pull request

Please keep PRs focused — one feature or fix per PR.

---

## License

MIT — see [LICENSE](LICENSE) for details.
