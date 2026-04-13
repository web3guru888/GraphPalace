# Palace Orchestrator

The `gp-palace` crate provides the `GraphPalace` struct — the unified entry point that ties together storage, stigmergy, pathfinding, embeddings, agents, and MCP tools into a single coherent API. If you're building an application on GraphPalace, this is the crate you depend on.

## Architecture

```
┌─────────────────────────────────────────┐
│              GraphPalace                 │  ← You are here
│  (unified orchestrator)                 │
├─────────┬──────────┬────────┬───────────┤
│ Storage │Stigmergy │ Search │  Agents   │
│gp-storage│gp-stigmergy│gp-pathfinding│gp-swarm│
├─────────┴──────────┴────────┴───────────┤
│              gp-core + gp-embeddings    │
└─────────────────────────────────────────┘
```

## The `GraphPalace` Struct

```rust
pub struct GraphPalace {
    backend: Box<dyn StorageBackend>,
    embeddings: Box<dyn EmbeddingEngine>,
    pheromone_config: PheromoneConfig,
    cost_weights: CostWeights,
    astar_config: AStarConfig,
    status: PalaceStatus,
}
```

### Construction

```rust
use gp_palace::{GraphPalace, PalaceBuilder};

// Quick start with defaults (in-memory backend, mock embeddings)
let palace = GraphPalace::default();

// Builder pattern for full control
let palace = PalaceBuilder::new()
    .name("Research Palace")
    .backend(InMemoryBackend::new())
    .embeddings(MockEmbeddingEngine::new(384))
    .pheromone_config(PheromoneConfig::default())
    .cost_weights(CostWeights::default())
    .build()?;
```

## Palace Lifecycle

A palace goes through a natural lifecycle: **create → populate → search → navigate → reinforce → decay → export**. Each phase maps to methods on `GraphPalace`.

### 1. Create — Build the Palace Structure

```rust
// Create the spatial hierarchy
palace.create_wing("w-research", "Research", "domain", "Scientific research")?;
palace.create_room("w-research", "r-climate", "Climate Science", "facts")?;
palace.create_closet("r-climate", "c-temperature", "Temperature Data", "Global temperature records")?;
```

### 2. Populate — Store Memories

Drawers store **verbatim content** — never summarized. The palace auto-embeds content.

```rust
// Store a memory — embedding is computed automatically
let drawer_id = palace.add_drawer(
    "c-temperature",
    "Global average temperature rose 1.1°C above pre-industrial levels by 2023.",
    "api",       // source
    0.8,         // importance
)?;

// Duplicate detection before adding
let dupes = palace.check_duplicate(
    "Temperature increased by 1.1 degrees Celsius globally",
    0.85,   // similarity threshold
)?;
if dupes.is_empty() {
    palace.add_drawer("c-temperature", content, "conversation", 0.5)?;
}
```

### 3. Search — Find Memories

Search orchestrates **embedding similarity + pheromone boosting**:

```rust
// Semantic search with pheromone-boosted ranking
let results = palace.search("how much has temperature increased?", SearchOptions {
    k: 10,
    wing: Some("w-research"),
    room: None,
    boost_pheromones: true,  // Multiply score by (1 + exploitation_pheromone)
})?;

for result in &results {
    println!("[{:.3}] {}", result.score, &result.drawer.content[..80]);
}
```

The search pipeline:
1. Encode query → 384-dim embedding
2. Vector search via backend (HNSW or brute-force) → top-k candidates
3. If `boost_pheromones`: multiply scores by `(1.0 + node.exploitation_pheromone)`
4. Re-rank and return
5. Deposit recency pheromones on accessed nodes

### 4. Navigate — Follow Connections

Semantic A* pathfinding finds optimal routes through the palace graph:

```rust
// Find a path between two nodes
let path = palace.navigate("d-climate-001", "d-economics-042", NavigateOptions {
    context: Some("How does climate affect GDP?"),  // Adapts cost weights
    max_iterations: 10_000,
})?;

println!("Path found in {} iterations, cost {:.3}", path.iterations, path.total_cost);
for step in &path.provenance {
    println!("  {} --[{}]--> {} (cost {:.3})",
        step.from, step.relation, step.to, step.cost);
}
```

Context-adaptive weights are selected automatically:

| Context Keywords | α (Semantic) | β (Pheromone) | γ (Structural) |
|-----------------|-------------|---------------|----------------|
| "hypothesis", "test" | 0.30 | 0.40 | 0.30 |
| "explore", "discover" | 0.50 | 0.20 | 0.30 |
| "evidence", "support" | 0.35 | 0.35 | 0.30 |
| "recall", "remember" | 0.50 | 0.30 | 0.20 |
| *(default)* | 0.40 | 0.30 | 0.30 |

### 5. Reinforce — Deposit Pheromones

After a successful search or navigation, reinforce the paths that led to good results:

```rust
// Deposit success pheromones along a path (position-weighted)
palace.reinforce_path(&path, 1.0)?;
// Edge[0] gets reward 1.0, Edge[1] gets 0.67, Edge[2] gets 0.33, etc.

// Mark a node as valuable
palace.deposit_exploitation("d-climate-001", 0.5)?;

// Mark a node as explored (discourages revisiting)
palace.deposit_exploration("r-dead-end", 0.3)?;
```

### 6. Decay — Let Pheromones Fade

Pheromones decay exponentially. Run decay periodically to let the palace self-optimize:

```rust
// Run one decay cycle (applies configured rates)
let stats = palace.decay()?;
println!("Decayed {} nodes, {} edges", stats.nodes_decayed, stats.edges_decayed);

// Or schedule automatic decay every N operations
palace.set_auto_decay(10); // Decay every 10 search/navigate operations
```

The five pheromone types decay at different rates:

| Type | Rate (ρ) | Half-life | Meaning |
|------|---------|-----------|---------|
| Exploitation | 0.02 | ~35 cycles | Value fades slowly |
| Exploration | 0.05 | ~14 cycles | "Already searched" clears faster |
| Success | 0.01 | ~69 cycles | Proven paths persist longest |
| Traversal | 0.03 | ~23 cycles | Usage patterns fade moderately |
| Recency | 0.10 | ~7 cycles | Freshness fades quickly |

### 7. Export/Import — Portable Palaces

Export the entire palace as a JSON file for backup, sharing, or migration:

```rust
// Export
let export = palace.export()?;
let json = serde_json::to_string_pretty(&export)?;
std::fs::write("palace_backup.json", &json)?;

// Import with three modes
palace.import(&export, ImportMode::Replace)?;  // Drop existing, load new
palace.import(&export, ImportMode::Merge)?;    // Add new nodes, skip duplicates
palace.import(&export, ImportMode::Overlay)?;  // Add new, update existing
```

The `PalaceExport` struct contains all nodes, edges, pheromone states, and metadata:

```rust
pub struct PalaceExport {
    pub version: String,
    pub exported_at: DateTime<Utc>,
    pub palace: PalaceRecord,
    pub wings: Vec<WingRecord>,
    pub rooms: Vec<RoomRecord>,
    pub closets: Vec<ClosetRecord>,
    pub drawers: Vec<DrawerRecord>,
    pub entities: Vec<EntityRecord>,
    pub agents: Vec<AgentRecord>,
    pub edges: Vec<EdgeRecord>,
}
```

## Knowledge Graph Operations

GraphPalace includes a knowledge graph layer for entity-relationship triples:

```rust
// Add entities
palace.add_entity("e-co2", "CO₂ Emissions", "concept", "Atmospheric carbon dioxide")?;
palace.add_entity("e-temp", "Global Temperature", "concept", "Average surface temperature")?;

// Add a temporal relationship
palace.add_relationship("e-co2", "causes", "e-temp", 0.92)?;

// Query an entity's relationships
let rels = palace.query_entity("CO₂ Emissions")?;
for rel in &rels {
    println!("{} --[{}, conf={:.2}]--> {}", rel.subject, rel.predicate, rel.confidence, rel.object);
}

// Find contradictions (conflicting relationships for the same entity)
let contradictions = palace.find_contradictions("e-temp")?;

// Temporal query: what was known as of a specific date?
let rels = palace.query_entity_as_of("e-temp", "2025-01-01T00:00:00Z")?;

// Invalidate a relationship (set valid_to = now)
palace.invalidate_relationship("e-co2", "inhibits", "e-temp")?;
```

## Palace Status and Monitoring

```rust
let status = palace.status()?;
println!("Palace: {}", status.name);
println!("Wings: {}, Rooms: {}, Drawers: {}", status.wings, status.rooms, status.drawers);
println!("Entities: {}, Relationships: {}", status.entities, status.relationships);
println!("Avg exploitation pheromone: {:.3}", status.avg_exploitation);
println!("Avg exploration pheromone: {:.3}", status.avg_exploration);
println!("Hot paths (top 5): {:?}", status.hot_paths);
println!("Cold spots (top 5): {:?}", status.cold_spots);
```

## Example: Full Lifecycle

```rust
use gp_palace::{GraphPalace, PalaceBuilder, SearchOptions, NavigateOptions, ImportMode};
use gp_storage::InMemoryBackend;
use gp_embeddings::MockEmbeddingEngine;

fn main() -> anyhow::Result<()> {
    // 1. Create palace
    let mut palace = PalaceBuilder::new()
        .name("Demo Palace")
        .backend(InMemoryBackend::new())
        .embeddings(MockEmbeddingEngine::new(384))
        .build()?;

    // 2. Build structure
    palace.create_wing("w-science", "Science", "domain", "Natural sciences")?;
    palace.create_room("w-science", "r-physics", "Physics", "facts")?;
    palace.create_closet("r-physics", "c-thermo", "Thermodynamics", "Laws of heat")?;

    // 3. Store memories
    palace.add_drawer("c-thermo", "Energy cannot be created or destroyed", "textbook", 0.9)?;
    palace.add_drawer("c-thermo", "Heat flows from hot to cold spontaneously", "textbook", 0.8)?;

    // 4. Search
    let results = palace.search("conservation of energy", SearchOptions::default())?;
    assert!(!results.is_empty());

    // 5. Reinforce successful search path
    palace.deposit_exploitation(&results[0].drawer.id, 0.5)?;

    // 6. Decay
    palace.decay()?;

    // 7. Export
    let export = palace.export()?;

    // 8. Import into fresh palace
    let mut palace2 = PalaceBuilder::new()
        .name("Imported Palace")
        .backend(InMemoryBackend::new())
        .embeddings(MockEmbeddingEngine::new(384))
        .build()?;
    palace2.import(&export, ImportMode::Replace)?;

    // 9. Verify round-trip
    let status1 = palace.status()?;
    let status2 = palace2.status()?;
    assert_eq!(status1.drawers, status2.drawers);
    assert_eq!(status1.entities, status2.entities);

    println!("Full lifecycle complete — {} drawers preserved", status2.drawers);
    Ok(())
}
```
