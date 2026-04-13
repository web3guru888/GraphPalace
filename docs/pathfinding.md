# Semantic A* Pathfinding

GraphPalace uses a modified A* search algorithm that navigates the memory palace using three complementary signals: semantic meaning, collective intelligence (pheromones), and graph structure. This enables context-aware navigation that improves with use.

## Composite Edge Cost Model

Every edge traversal has a composite cost computed from three components:

```
cost(edge) = α × C_semantic + β × C_pheromone + γ × C_structural
```

Default weights: **α = 0.4, β = 0.3, γ = 0.3**

### Cost Components

#### Semantic Cost (C_semantic)

How far is the target node from the search goal, measured by embedding similarity:

```rust
fn semantic_cost(target_embedding: &[f32; 384], goal_embedding: &[f32; 384]) -> f64 {
    1.0 - cosine_similarity(target_embedding, goal_embedding) as f64
}
```

- **0.0** = target is identical to goal (maximum relevance)
- **1.0** = target is orthogonal to goal (no relevance)

#### Pheromone Cost (C_pheromone)

Inverse of collective trail strength — edges with strong pheromones are cheaper:

```rust
fn pheromone_cost(edge: &Edge) -> f64 {
    1.0 - (0.5 * edge.success_pheromone.min(1.0)
         + 0.3 * edge.recency_pheromone.min(1.0)
         + 0.2 * edge.traversal_pheromone.min(1.0))
}
```

- **0.0** = maximum pheromone signal (well-proven path)
- **1.0** = no pheromones (unexplored path)

#### Structural Cost (C_structural)

Base cost determined by relation type — palace hierarchy edges are cheap, knowledge graph edges are expensive:

```rust
fn structural_cost(relation_type: &str) -> f64 {
    match relation_type {
        "CONTAINS" => 0.2,           // Palace → Wing
        "HAS_ROOM" => 0.3,          // Wing → Room
        "HAS_CLOSET" => 0.3,        // Room → Closet
        "HAS_DRAWER" => 0.3,        // Closet → Drawer
        "SIMILAR_TO" => 0.4,        // Semantic similarity link
        "HALL" => 0.5,              // Same-wing room connection
        "REFERENCES" => 0.5,        // Drawer → Entity
        "TUNNEL" => 0.7,            // Cross-wing connection
        "RELATES_TO" => 0.8,        // Knowledge graph (default)
        _ => 1.0,                   // Unknown relations
    }
}
```

## Context-Adaptive Weights

The α/β/γ weights can be adjusted based on the task context:

| Context | α (Semantic) | β (Pheromone) | γ (Structural) | When to Use |
|---------|-------------|---------------|----------------|-------------|
| **Default** | 0.40 | 0.30 | 0.30 | General navigation |
| **Hypothesis Testing** | 0.30 | 0.40 | 0.30 | Follow proven causal chains |
| **Exploratory Research** | 0.50 | 0.20 | 0.30 | Prioritize semantic relevance |
| **Evidence Gathering** | 0.35 | 0.35 | 0.30 | Balance meaning and experience |
| **Memory Recall** | 0.50 | 0.30 | 0.20 | Trust semantic similarity, relax structure |

## Adaptive Heuristic

The A* heuristic adapts between two modes based on how similar the current node is to the goal:

```rust
fn semantic_heuristic(current: &Node, goal: &Node) -> f64 {
    let h_semantic = 1.0 - cosine_similarity(&current.embedding, &goal.embedding);
    let connectivity = (current.degree as f64 / 20.0).clamp(0.1, 1.0);
    let h_graph = (h_semantic / connectivity) * 0.5;

    let similarity = 1.0 - h_semantic;
    if similarity < 0.3 {
        // Cross-domain: weight graph distance more (50/50)
        0.5 * h_semantic + 0.5 * h_graph
    } else {
        // Same domain: trust semantic similarity (90/10)
        0.9 * h_semantic + 0.1 * h_graph
    }
}
```

**Intuition**:
- When the current node is **semantically far** from the goal (similarity < 0.3), the search is likely crossing domains — graph structure matters more for finding the right wing/room.
- When the current node is **semantically close** (similarity ≥ 0.3), we're in the right area — trust semantic similarity to find the exact target.
- **Connectivity** adjusts the estimate — highly connected nodes (hubs) are more likely to be on shortest paths.

## A* Implementation

```rust
pub struct SemanticAStar {
    max_iterations: usize,   // Default: 10_000
    cost_weights: CostWeights,
}

pub struct PathResult {
    pub path: Vec<Edge>,
    pub total_cost: f64,
    pub iterations: usize,
    pub nodes_expanded: usize,
    pub provenance: Vec<ProvenanceStep>,
}
```

The algorithm uses:
- **Open set**: `BinaryHeap` (min-heap via `ordered-float`)
- **Closed set**: `HashSet<NodeId>`
- **g-scores**: `HashMap<NodeId, f64>` — best known cost from start
- **f-scores**: `g(n) + h(n)` — estimated total cost through node

Terminates when:
1. Goal node is reached → return optimal path
2. `max_iterations` exceeded → return `None`
3. Open set is empty → return `None` (no path exists)

## Provenance Tracking

Every path result includes provenance — a record of why each step was chosen:

```rust
pub struct ProvenanceStep {
    pub from_node: String,
    pub to_node: String,
    pub edge_type: String,
    pub semantic_cost: f64,
    pub pheromone_cost: f64,
    pub structural_cost: f64,
    pub composite_cost: f64,
    pub explanation: String,
}
```

This allows agents and users to understand not just the path, but the reasoning behind each navigation decision.

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| A* cached pathfinding | <200ms | STAN_X achieves 211ms |
| A* uncached pathfinding | <500ms | STAN_X achieves 494ms |
| Nodes expanded (typical) | <1000 | With good heuristic |

## Crate: `gp-pathfinding`

Source: `rust/gp-pathfinding/src/`

| File | Purpose |
|------|---------|
| `lib.rs` | Public API, module exports |
| `astar.rs` | SemanticAStar implementation with open/closed sets |
| `heuristic.rs` | Adaptive cross/same-domain heuristic |
| `edge_cost.rs` | Composite cost model (α/β/γ), context-adaptive weights |
| `provenance.rs` | Path provenance tracking and explanation |
