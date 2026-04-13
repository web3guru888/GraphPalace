# Stigmergy System

The stigmergy system is GraphPalace's collective intelligence layer. Inspired by ant colony optimization and adapted from STAN_X v8, it uses five types of pheromone trails to guide navigation through the memory palace. As agents and searches traverse the graph, they leave behind chemical-like signals that future navigators can follow.

## Five Pheromone Types

### Node Pheromones

Applied to `Wing`, `Room`, `Closet`, `Drawer`, and `Entity` nodes:

| Type | Signal | Decay Rate (ρ) | Half-life | Effect |
|------|--------|----------------|-----------|--------|
| **Exploitation** | "This location is valuable — come here" | 0.02/cycle | ~35 cycles | Attracts agents to valuable areas |
| **Exploration** | "This location has been searched — try elsewhere" | 0.05/cycle | ~14 cycles | Repels agents from over-searched areas |

### Edge Pheromones

Applied to all edge types (`HAS_ROOM`, `HALL`, `TUNNEL`, `RELATES_TO`, etc.):

| Type | Signal | Decay Rate (ρ) | Half-life | Effect |
|------|--------|----------------|-----------|--------|
| **Success** | "This connection led to good outcomes" | 0.01/cycle | ~69 cycles | Long-lasting quality signal |
| **Traversal** | "This path is frequently used" | 0.03/cycle | ~23 cycles | Popularity signal |
| **Recency** | "This connection was used recently" | 0.10/cycle | ~7 cycles | Short-term freshness signal |

## Decay Formula

All pheromones decay exponentially each cycle:

```rust
fn decay(current: f64, rate: f64) -> f64 {
    current * (1.0 - rate)
}
```

**Half-life calculation**: `half_life = ln(2) / ln(1 / (1 - rate))` ≈ `0.693 / rate` for small rates.

Decay is applied in bulk via Cypher every N cycles (default: 10):

```cypher
-- Decay all node pheromones
MATCH (n)
WHERE n.exploitation_pheromone > 0.001 OR n.exploration_pheromone > 0.001
SET n.exploitation_pheromone = n.exploitation_pheromone * (1.0 - $exploitation_rate),
    n.exploration_pheromone = n.exploration_pheromone * (1.0 - $exploration_rate)

-- Decay all edge pheromones
MATCH ()-[e]->()
WHERE e.success_pheromone > 0.001 OR e.traversal_pheromone > 0.001 OR e.recency_pheromone > 0.001
SET e.success_pheromone = e.success_pheromone * (1.0 - $success_rate),
    e.traversal_pheromone = e.traversal_pheromone * (1.0 - $traversal_rate),
    e.recency_pheromone = e.recency_pheromone * (1.0 - $recency_rate)
```

Values below 0.001 are treated as zero (noise floor).

## Pheromone Deposit Operations

### After Successful Search (Position-Weighted)

When a search path leads to a useful result, pheromones are deposited along the entire path. Earlier edges in the path (closer to the start) receive larger rewards:

```rust
fn deposit_path_success(path: &[Edge], base_reward: f64) {
    for (i, edge) in path.iter().enumerate() {
        let position_weight = 1.0 - (i as f64 / path.len() as f64);
        let reward = base_reward * position_weight;
        edge.success_pheromone += reward;
        edge.traversal_pheromone += 0.1;
        edge.recency_pheromone = 1.0; // Reset to max
    }
    for node in path.nodes() {
        node.exploitation_pheromone += 0.2;
    }
}
```

**Why position-weighted?** The first edges in a successful path are the most valuable decisions — they set the right direction. Later edges contribute less to the overall success.

### After Exploration (Mark as Searched)

When a node is visited during exploration:

```rust
fn deposit_exploration(node: &mut Node) {
    node.exploration_pheromone += 0.3;
}
```

This discourages other agents from re-exploring the same area immediately, promoting diversity.

## Edge Cost Recomputation

After any pheromone change, edge costs are recomputed. Pheromones reduce the effective cost of an edge — stronger pheromone trails make paths cheaper to traverse:

```rust
fn recompute_edge_cost(edge: &mut Edge) {
    let pheromone_factor =
        0.5 * edge.success_pheromone.min(1.0)     // Quality: 50%
      + 0.3 * edge.recency_pheromone.min(1.0)     // Freshness: 30%
      + 0.2 * edge.traversal_pheromone.min(1.0);   // Popularity: 20%
    edge.current_cost = (edge.base_cost * (1.0 - pheromone_factor * 0.5))
        .clamp(0.0, 10.0);
}
```

**Interpretation**:
- An edge with zero pheromones has `current_cost == base_cost`
- An edge with maximum pheromones (all 1.0) has `current_cost == base_cost * 0.5`
- Pheromones can never reduce cost below 50% of base (prevents over-exploitation)
- The 50/30/20 weighting prioritizes quality (success) over freshness (recency) over popularity (traversal)

## Configuration

Default pheromone parameters in `graphpalace.toml`:

```toml
[pheromones]
exploitation_decay = 0.02    # Node: ~35 cycle half-life
exploration_decay = 0.05     # Node: ~14 cycle half-life
success_decay = 0.01         # Edge: ~69 cycle half-life (most persistent)
traversal_decay = 0.03       # Edge: ~23 cycle half-life
recency_decay = 0.10         # Edge: ~7 cycle half-life (most volatile)
decay_interval_cycles = 10   # How often to run bulk decay
```

## Emergent Behaviors

The pheromone system produces several emergent properties without explicit programming:

1. **Highway formation** — Frequently successful paths develop strong pheromone trails, creating "highways" through the palace that agents prefer.

2. **Exploration pressure** — The exploration pheromone's faster decay (14 cycles vs. 35) means explored areas become "available" again before valuable areas lose their signal, creating a natural exploration/exploitation balance.

3. **Recency bias** — The recency pheromone's very fast decay (7 cycles) ensures recently-used connections are temporarily boosted, creating a "working memory" effect.

4. **Quality persistence** — Success pheromone's slow decay (69 cycles) means proven connections remain valuable long after they were last used.

5. **Dead path pruning** — Paths that are never successful have their pheromones decay to zero, effectively pruning them from consideration without explicit deletion.

## Crate: `gp-stigmergy`

Source: `rust/gp-stigmergy/src/`

| File | Purpose |
|------|---------|
| `lib.rs` | Public API, module exports |
| `pheromones.rs` | PheromoneManager — read/write pheromone levels |
| `decay.rs` | DecayEngine — exponential decay with configurable rates |
| `rewards.rs` | RewardCalculator — position-weighted path rewards |
| `cost.rs` | Edge cost recomputation after pheromone changes |
