# Storage Backends

The `gp-storage` crate provides the storage abstraction layer for GraphPalace. It defines the `StorageBackend` trait that all storage implementations must satisfy, and ships two backends: `KuzuBackend` (production, via C FFI) and `InMemoryBackend` (testing, WASM).

## Architecture

```
┌───────────────────────────────────────┐
│          gp-palace / gp-mcp           │  ← Consumers
├───────────────────────────────────────┤
│           StorageBackend trait         │  ← Abstraction
├──────────────────┬────────────────────┤
│   KuzuBackend    │  InMemoryBackend   │  ← Implementations
│  (kuzu-ffi gate) │  (default, WASM)   │
└──────────────────┴────────────────────┘
```

The crate uses a **feature flag** to control which backend is available:

| Feature | Backend | When to Use |
|---------|---------|-------------|
| `kuzu-ffi` | `KuzuBackend` | Native builds (CLI, server, Python) — requires Kuzu C++ library |
| *(default)* | `InMemoryBackend` | Testing, WASM, CI — no external dependencies |

## The `StorageBackend` Trait

```rust
pub trait StorageBackend: Send + Sync {
    // --- Schema ---
    fn initialize_schema(&mut self) -> Result<()>;

    // --- Palace CRUD ---
    fn create_palace(&mut self, id: &str, name: &str, description: &str) -> Result<()>;
    fn create_wing(&mut self, id: &str, name: &str, wing_type: &str, desc: &str) -> Result<String>;
    fn create_room(&mut self, wing_id: &str, id: &str, name: &str, hall_type: &str) -> Result<String>;
    fn create_closet(&mut self, room_id: &str, id: &str, name: &str, summary: &str) -> Result<String>;
    fn create_drawer(&mut self, closet_id: &str, drawer: &DrawerNode) -> Result<String>;

    // --- Read ---
    fn get_wing(&self, id: &str) -> Result<Option<WingNode>>;
    fn get_room(&self, id: &str) -> Result<Option<RoomNode>>;
    fn get_drawer(&self, id: &str) -> Result<Option<DrawerNode>>;
    fn list_wings(&self) -> Result<Vec<WingNode>>;
    fn list_rooms(&self, wing_id: &str) -> Result<Vec<RoomNode>>;
    fn get_taxonomy(&self) -> Result<PalaceTaxonomy>;

    // --- Search ---
    fn vector_search(&self, embedding: &[f32; 384], k: usize) -> Result<Vec<(DrawerNode, f32)>>;
    fn full_text_search(&self, query: &str, k: usize) -> Result<Vec<(DrawerNode, f32)>>;

    // --- Knowledge Graph ---
    fn add_entity(&mut self, entity: &EntityNode) -> Result<String>;
    fn add_relationship(&mut self, from: &str, predicate: &str, to: &str, confidence: f64) -> Result<()>;
    fn query_entity(&self, name: &str) -> Result<Vec<Relationship>>;

    // --- Pheromones ---
    fn update_node_pheromones(&mut self, id: &str, exploitation: f64, exploration: f64) -> Result<()>;
    fn update_edge_pheromones(&mut self, from: &str, to: &str, rel: &str, pheromones: &EdgePheromones) -> Result<()>;
    fn decay_all_pheromones(&mut self, config: &PheromoneConfig) -> Result<DecayStats>;

    // --- Export/Import ---
    fn export(&self) -> Result<PalaceExport>;
    fn import(&mut self, data: &PalaceExport, mode: ImportMode) -> Result<ImportStats>;

    // --- Stats ---
    fn node_count(&self) -> Result<usize>;
    fn edge_count(&self) -> Result<usize>;
}
```

## `KuzuBackend` — Production Storage

The `KuzuBackend` wraps Kuzu's C API via the `kuzu-sys` FFI crate. It provides full graph database capabilities: ACID transactions, HNSW vector indexes, BM25 full-text search, and columnar storage.

### Connection Lifecycle

```rust
use gp_storage::KuzuBackend;

// Open or create a database directory
let mut backend = KuzuBackend::open("./my_palace_db")?;

// Initialize schema (idempotent — safe to call on existing DB)
backend.initialize_schema()?;

// All operations go through the Kuzu connection
backend.create_wing("w-research", "Research", "domain", "Scientific research")?;

// Backend manages a connection pool internally
// Queries are executed as Cypher via kuzu_connection_query()
```

### How It Works

1. **Database**: `kuzu_database_init(path)` opens or creates the on-disk database directory
2. **Connection**: `kuzu_connection_init(db)` creates a thread-safe connection
3. **Query Cycle**: All operations translate to Cypher queries:
   - `CREATE (w:Wing {id: $id, name: $name, ...})` for node creation
   - `MATCH (w:Wing {id: $wing_id}) CREATE (w)-[:HAS_ROOM]->(r:Room {...})` for hierarchy
   - `CALL vector_search('drawer_embedding_idx', $embedding, $k)` for semantic search
4. **Results**: `kuzu_query_result_*` functions extract typed values from result sets

### Storage Targets

| Target | How |
|--------|-----|
| **File System** | Standard Kuzu database directory (native CLI, server) |
| **IndexedDB** | Via Emscripten FS (browser WASM, Kuzu's existing WASM support) |
| **OPFS** | Origin Private File System (preferred browser WASM — faster, persistent) |

## `InMemoryBackend` — Testing & WASM

The `InMemoryBackend` stores the entire palace graph in `HashMap`-based data structures. It requires no external dependencies and compiles to WASM without modification.

```rust
use gp_storage::InMemoryBackend;

let mut backend = InMemoryBackend::new();
backend.initialize_schema()?; // No-op for in-memory (schema is implicit)

backend.create_wing("w-test", "Test Wing", "domain", "For testing")?;
```

### Data Structures

```rust
pub struct InMemoryBackend {
    palaces: HashMap<String, PalaceRecord>,
    wings: HashMap<String, WingRecord>,
    rooms: HashMap<String, RoomRecord>,
    closets: HashMap<String, ClosetRecord>,
    drawers: HashMap<String, DrawerRecord>,
    entities: HashMap<String, EntityRecord>,
    agents: HashMap<String, AgentRecord>,

    // Edges stored as adjacency lists
    contains: HashMap<String, Vec<String>>,      // Palace → Wings
    has_room: HashMap<String, Vec<String>>,       // Wing → Rooms
    has_closet: HashMap<String, Vec<String>>,     // Room → Closets
    has_drawer: HashMap<String, Vec<String>>,     // Closet → Drawers
    relates_to: Vec<RelationshipRecord>,          // Entity → Entity
    references: Vec<ReferenceRecord>,             // Drawer → Entity
    similar_to: Vec<SimilarityRecord>,            // Drawer → Drawer
}
```

### Vector Search (Brute Force)

Without Kuzu's HNSW index, `InMemoryBackend` performs brute-force cosine similarity:

```rust
fn vector_search(&self, embedding: &[f32; 384], k: usize) -> Result<Vec<(DrawerNode, f32)>> {
    let mut scores: Vec<_> = self.drawers.values()
        .map(|d| (d, cosine_similarity(embedding, &d.embedding)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(k);
    Ok(scores.into_iter().map(|(d, s)| (d.into(), s)).collect())
}
```

This is O(n) per query — fine for testing and small palaces (<10K drawers), but `KuzuBackend` with HNSW should be used for production workloads.

## Schema Initialization

When `initialize_schema()` is called on `KuzuBackend`, it executes the full Cypher DDL from the GraphPalace schema specification:

### 7 Node Tables

| Table | Key Fields | Purpose |
|-------|-----------|---------|
| `Palace` | id, name, description | Root container |
| `Wing` | id, name, wing_type, embedding[384], pheromones | Domain grouping |
| `Room` | id, name, hall_type, embedding[384], pheromones | Subject area |
| `Closet` | id, name, summary, embedding[384], pheromones | Summary container |
| `Drawer` | id, content, embedding[384], source, importance, pheromones | Verbatim memory |
| `Entity` | id, name, entity_type, embedding[384], pheromones | Knowledge graph node |
| `Agent` | id, name, domain, diary, goal_embedding[384], temperature | Specialist agent |

### 11 Edge Tables

| Table | From → To | Key Fields |
|-------|-----------|-----------|
| `CONTAINS` | Palace → Wing | *(structural)* |
| `HAS_ROOM` | Wing → Room | base_cost, current_cost, 3 pheromones |
| `HAS_CLOSET` | Room → Closet | base_cost, current_cost, 3 pheromones |
| `HAS_DRAWER` | Closet → Drawer | base_cost, current_cost, 3 pheromones |
| `HALL` | Room → Room | hall_type, base_cost, current_cost, 3 pheromones |
| `TUNNEL` | Room → Room | base_cost, current_cost, 3 pheromones |
| `RELATES_TO` | Entity → Entity | predicate, confidence, valid_from/to, 3 pheromones |
| `REFERENCES` | Drawer → Entity | relevance, 3 pheromones |
| `SIMILAR_TO` | Drawer → Drawer | similarity, 3 pheromones |
| `MANAGES` | Agent → Wing | *(structural)* |
| `INVESTIGATED` | Agent → Drawer | result, investigated_at |

### Indexes

Schema initialization also creates:
- **3 HNSW vector indexes** on Drawer, Entity, and Room embeddings (cosine, M=16, ef=200)
- **2 FTS indexes** on Drawer content and Entity name/description
- **4 property indexes** on Wing name, Room hall_type, Entity type, and RELATES_TO temporal fields

## Example Usage

```rust
use gp_storage::{StorageBackend, InMemoryBackend};
use gp_core::types::DrawerNode;

fn main() -> anyhow::Result<()> {
    // Create backend (in-memory for this example)
    let mut backend = InMemoryBackend::new();
    backend.initialize_schema()?;

    // Build palace hierarchy
    backend.create_palace("palace-1", "Research Palace", "Scientific discoveries")?;
    backend.create_wing("w-climate", "Climate Science", "domain", "Climate research")?;
    backend.create_room("w-climate", "r-temperature", "Temperature Records", "facts")?;
    backend.create_closet("r-temperature", "c-global-avg", "Global Averages", "Global temperature data")?;

    // Store a verbatim memory
    let drawer = DrawerNode::new(
        "d-001",
        "Global average temperature rose 1.1°C above pre-industrial levels by 2023.",
    )
    .with_source("api")
    .with_importance(0.8)
    .with_embedding(embedding_engine.encode("global temperature rise")?);

    backend.create_drawer("c-global-avg", &drawer)?;

    // Search by embedding
    let query_emb = embedding_engine.encode("how much warmer is the planet?")?;
    let results = backend.vector_search(&query_emb, 5)?;
    for (drawer, score) in &results {
        println!("[{:.3}] {}", score, &drawer.content[..60]);
    }

    // Export/import round-trip
    let export = backend.export()?;
    let mut new_backend = InMemoryBackend::new();
    new_backend.import(&export, ImportMode::Replace)?;
    assert_eq!(backend.node_count()?, new_backend.node_count()?);

    Ok(())
}
```

## Feature Flags

In your `Cargo.toml`:

```toml
[dependencies]
gp-storage = "0.1"                    # In-memory backend only (default)
gp-storage = { version = "0.1", features = ["kuzu-ffi"] }  # + Kuzu backend
```

The `kuzu-ffi` feature:
- Links against Kuzu's C library (`libkuzu.so` / `libkuzu.dylib`)
- Requires Kuzu to be built from the C++ source (see build instructions)
- Enables `KuzuBackend::open(path)` constructor
- Without this feature, only `InMemoryBackend` is available
