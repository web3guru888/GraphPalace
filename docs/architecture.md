# Architecture Overview

GraphPalace is a layered system built on top of the Kuzu embedded graph database. The Rust crate layer implements the memory palace semantics — stigmergy, pathfinding, agents, and MCP tooling — while Kuzu provides the graph storage, Cypher query engine, vector indexes, and WASM compilation target.

## System Layers

```
┌─────────────────────────────────────────────────┐
│              Application Layer                   │
│  MCP Server │ CLI │ Python bindings │ WASM/JS   │
├─────────────────────────────────────────────────┤
│              Orchestration Layer                  │
│  gp-palace  (Unified orchestrator)               │
│  gp-bench   (Benchmark infrastructure)           │
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
│  gp-storage (StorageBackend trait)               │
│  KuzuBackend (C++ FFI) │ InMemoryBackend         │
└─────────────────────────────────────────────────┘
```

## Crate Dependency Graph

```
gp-wasm ──→ gp-mcp ──→ gp-palace ──→ gp-swarm ──→ gp-agents ──→ gp-pathfinding ──→ gp-stigmergy ──→ gp-core
                    │            │            └──→ gp-stigmergy                  │                │
                    │            │                                                └──→ gp-embeddings│
                    │            ├──→ gp-storage ──→ gp-core                                      │
                    │            │    (KuzuBackend / InMemoryBackend)                               │
                    │            ├──→ gp-embeddings                                                │
                    │            └──→ gp-stigmergy                                                 │
                    │                                                                              │
gp-bench ──→ gp-palace                                    gp-embeddings ──→ gp-core ──────────────┘
         └──→ gp-storage
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
| `gp-storage` | `gp-core` | StorageBackend trait, KuzuBackend (FFI), InMemoryBackend |
| `gp-palace` | `gp-storage`, `gp-embeddings`, `gp-stigmergy`, `gp-swarm` | Unified orchestrator — main entry point |
| `gp-bench` | `gp-palace`, `gp-storage` | Benchmark suite — recall, pathfinding, throughput |
| `gp-mcp` | `gp-palace`, `gp-core` | 28 MCP tool definitions, PALACE_PROTOCOL |
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

---

## Crate Details: gp-storage

The storage abstraction layer. Defines the `StorageBackend` trait and ships two implementations.

```
gp-storage/
├── Cargo.toml
└── src/
    ├── lib.rs                 # Re-exports, feature gates
    ├── backend.rs             # StorageBackend trait definition
    ├── in_memory.rs           # InMemoryBackend (HashMap-based)
    ├── kuzu.rs                # KuzuBackend (C FFI, behind `kuzu-ffi` feature)
    ├── schema.rs              # Cypher DDL generation (7 node + 11 edge tables)
    └── export.rs              # PalaceExport, ImportMode, serialization
```

**Key types**: `StorageBackend` (trait), `InMemoryBackend`, `KuzuBackend`, `PalaceExport`, `ImportMode`

**Feature flags**: Default is in-memory only. Enable `kuzu-ffi` for the Kuzu C++ backend. This keeps WASM builds lightweight and CI fast.

**Design**: Every operation in GraphPalace ultimately flows through `StorageBackend`. The trait is `Send + Sync` so `GraphPalace` can be used from async contexts. The in-memory backend uses brute-force cosine similarity for vector search (O(n), fine for <10K drawers). The Kuzu backend uses HNSW indexes for O(log n) search.

See [storage.md](storage.md) for full documentation.

## Crate Details: gp-palace

The unified orchestrator — the main entry point for applications. Ties together storage, embeddings, stigmergy, pathfinding, agents, and export/import into a single `GraphPalace` struct.

```
gp-palace/
├── Cargo.toml
└── src/
    ├── lib.rs                 # GraphPalace struct, PalaceBuilder
    ├── search.rs              # Semantic search with pheromone boosting
    ├── navigate.rs            # A* navigation with context-adaptive weights
    ├── knowledge_graph.rs     # Entity CRUD, relationship queries, contradictions
    ├── lifecycle.rs           # Create/populate/decay operations
    ├── export_import.rs       # Export, import with Replace/Merge/Overlay modes
    └── status.rs              # PalaceStatus, monitoring, hot paths, cold spots
```

**Key types**: `GraphPalace`, `PalaceBuilder`, `SearchOptions`, `NavigateOptions`, `PalaceStatus`

**Design**: `GraphPalace` owns a `Box<dyn StorageBackend>` and a `Box<dyn EmbeddingEngine>`. All public methods coordinate across subsystems — for example, `search()` encodes the query via the embedding engine, searches via the storage backend, boosts scores using stigmergy, then deposits recency pheromones. The `PalaceBuilder` follows the builder pattern for ergonomic construction.

**Lifecycle**: create → populate → search → navigate → reinforce → decay → export. See [palace.md](palace.md) for full documentation.

## Crate Details: gp-bench

Benchmark infrastructure for measuring recall, pathfinding, and throughput against MemPalace and STAN_X baselines.

```
gp-bench/
├── Cargo.toml
└── src/
    ├── lib.rs                 # run_all_benchmarks(), BenchConfig
    ├── recall.rs              # RecallBenchmark — target ≥96.6%
    ├── pathfinding.rs         # PathfindingBenchmark — target ≥90.9%
    ├── throughput.rs          # ThroughputBenchmark — insert/search/decay rates
    ├── generator.rs           # TestPalaceGenerator — deterministic test data
    └── report.rs              # Structured JSON reporting
```

**Key types**: `RecallBenchmark`, `PathfindingBenchmark`, `ThroughputBenchmark`, `TestPalaceGenerator`, `BenchReport`

**Design**: Each benchmark struct uses the builder pattern for configuration. `TestPalaceGenerator` uses a seeded RNG for deterministic, reproducible test palaces across runs. Results are structured as JSON for CI integration. The generator creates realistic multi-domain content across 6 domains (climate, economics, astrophysics, epidemiology, materials science, general knowledge).

See [benchmarks.md](benchmarks.md) for full documentation.
