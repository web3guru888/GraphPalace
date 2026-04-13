# Active Inference Agents

GraphPalace agents are autonomous navigators of the memory palace. Based on Karl Friston's Active Inference framework (adapted from STAN_X v8), they maintain Bayesian beliefs about the palace and choose actions that minimize Expected Free Energy — naturally balancing exploration of unknown areas with exploitation of known valuable paths.

## Agent Architecture

```rust
pub struct ActiveInferenceAgent {
    pub id: String,
    pub name: String,
    pub beliefs: HashMap<String, BeliefState>,  // Node ID → belief
    pub generative_model: GenerativeModel,
    pub goal_embedding: [f32; 384],
    pub temperature: f64,                       // [0.1, 1.0]
}

pub struct BeliefState {
    pub mean: f64,       // Expected value (prior: 20.0)
    pub precision: f64,  // 1/variance (prior: 0.1 = high uncertainty)
}
```

Each agent has:
- **Beliefs** about every palace node it has encountered (mean + precision)
- A **generative model** that predicts what it will find in unexplored areas
- A **goal embedding** representing what it's looking for
- A **temperature** controlling exploration vs. exploitation

## Expected Free Energy (EFE)

The core decision-making mechanism. Agents evaluate candidate nodes by computing EFE — lower is better:

```rust
fn expected_free_energy(node: &Node, agent: &ActiveInferenceAgent) -> f64 {
    let belief = agent.beliefs.get(&node.id).unwrap_or(&DEFAULT_BELIEF);

    // Epistemic value: how much will we learn?
    let epistemic = 1.0 / belief.precision;

    // Pragmatic value: how close to the goal?
    let pragmatic = cosine_similarity(&node.embedding, &agent.goal_embedding)
        .max(0.0);

    // Edge quality: collective intelligence signal
    let edge_quality = 0.5 * node.exploitation_pheromone
                     - 0.3 * node.exploration_pheromone;

    -(epistemic + pragmatic + edge_quality) // Minimize → negate
}
```

### EFE Components

| Component | Measures | High Value Means |
|-----------|---------|-----------------|
| **Epistemic** | 1/precision — uncertainty about a node | "We don't know much about this node — visiting it reduces uncertainty" |
| **Pragmatic** | Cosine similarity to goal | "This node is relevant to what we're looking for" |
| **Edge quality** | Exploitation - exploration pheromones | "The swarm thinks this node is valuable and hasn't been over-explored" |

The negation means **lower EFE = better choice**. An agent selects the node with the lowest EFE.

## Bayesian Belief Update

When an agent visits a node and observes its value, beliefs are updated using Bayesian inference:

```rust
impl BeliefState {
    fn update(&mut self, observation: f64, observation_precision: f64) {
        let prior_precision = self.precision;
        let prior_mean = self.mean;

        // Posterior precision = prior + observation
        self.precision = prior_precision + observation_precision;

        // Posterior mean = precision-weighted average
        self.mean = (prior_precision * prior_mean
                   + observation_precision * observation) / self.precision;
    }
}
```

**Properties**:
- More precise observations (higher `observation_precision`) have more influence
- Precision only increases — agents become more certain over time
- Prior mean (20.0) is intentionally high to encourage initial exploration (optimistic priors)

### Belief Merging (Multi-Agent)

When multiple agents explore the same area, their beliefs can be merged:

```rust
fn merge(beliefs: &[&BeliefState]) -> BeliefState {
    let total_precision: f64 = beliefs.iter().map(|b| b.precision).sum();
    let merged_mean: f64 = beliefs.iter()
        .map(|b| b.precision * b.mean)
        .sum::<f64>() / total_precision;
    BeliefState { mean: merged_mean, precision: total_precision }
}
```

## Action Selection (Softmax Policy)

Agents don't always pick the best option — they sample from a softmax distribution over EFE scores, controlled by temperature:

```rust
fn select_action(
    candidates: &[(String, f64)],  // (node_id, EFE)
    temperature: f64,
) -> String {
    let weights: Vec<f64> = candidates.iter()
        .map(|(_, efe)| (-efe / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();
    weighted_sample(candidates, &probs)
}
```

- **Low temperature (0.1)** → nearly deterministic, always picks lowest EFE
- **High temperature (1.0)** → nearly uniform random, explores widely
- **Medium temperature (0.5)** → balanced sampling

## Temperature Annealing

Temperature can change over time using one of three schedules:

```rust
enum AnnealingSchedule {
    Linear { start: f64, end: f64 },
    Exponential { start: f64, decay: f64 },
    Cosine { start: f64, end: f64 },
}
```

**Cosine annealing** (default) provides smooth transitions:
```
T(t) = T_end + 0.5 × (T_start - T_end) × (1 + cos(π × progress))
```

This allows agents to explore broadly early in a session, then focus on exploitation as they build confidence.

## Five Agent Archetypes

| Archetype | Temperature | Goal | Palace Role |
|-----------|------------|------|-------------|
| **Explorer** | 1.0 | None (pure epistemic) | Discover new rooms, expand palace frontier |
| **Exploiter** | 0.1 | Domain-specific | Follow proven paths, retrieve known memories |
| **Balanced** | 0.5 | Domain-specific | Default — mix exploration and exploitation |
| **Specialist** | 0.3 | Fixed domain embedding | Manage a specific wing, keep diary |
| **Generalist** | 0.7 | Rotating | Cross-wing connections, find tunnels |

### Agent Diaries

Specialist agents maintain persistent diaries stored in the Agent node's `diary` field. Diary entries are compressed using the AAAK dialect (a compressed communication format) and persisted to the graph.

## Swarm Coordination

Multiple agents operate in coordinated cycles via `gp-swarm`:

```
for each cycle:
    1. SENSE   — Get frontier nodes with interest scores
    2. DECIDE  — Each agent computes EFE, selects action via softmax
    3. ACT     — Agents navigate to selected nodes using A*
    4. UPDATE  — Deposit pheromones along successful paths
    5. DECAY   — Every N cycles, decay all pheromones
    6. CHECK   — Convergence? Stop if ≥ 2/3 criteria met
```

### Convergence Detection

The swarm declares convergence when at least 2 of 3 criteria are met:

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Average growth rate | < 5 nodes/cycle | Palace isn't expanding much |
| Pheromone variance | < 0.05 | Trails have stabilized |
| Frontier size | < 10 nodes | Few unexplored areas remain |

### Interest Score

Nodes are scored for how interesting they are to the swarm:

```rust
fn compute_interest_score(node: &Node) -> f64 {
    let structural = 1.0 / (1.0 + node.degree as f64);         // Prefer leaf nodes
    let pheromone = 0.6 * (exploitation - exploration) + 0.4 * exploitation;
    let noise = gaussian_noise(0.0, 0.1);                       // Prevent ties
    structural + pheromone + noise
}
```

## Crates

### `gp-agents`

Source: `rust/gp-agents/src/`

| File | Purpose |
|------|---------|
| `lib.rs` | Public API, module exports |
| `active_inference.rs` | EFE computation, agent navigation loop |
| `beliefs.rs` | Bayesian belief state, update, merge, KL divergence |
| `generative_model.rs` | Welford stats, hierarchical prediction |
| `action_selection.rs` | Softmax policy, temperature annealing |
| `archetypes.rs` | Explorer, Exploiter, Balanced, Specialist, Generalist |

### `gp-swarm`

Source: `rust/gp-swarm/src/`

| File | Purpose |
|------|---------|
| `lib.rs` | Public API, module exports |
| `coordinator.rs` | Multi-agent cycle orchestration |
| `convergence.rs` | 3-criteria convergence detection |
| `interest.rs` | Interest score computation |
| `decay_scheduler.rs` | Async decay scheduling |
