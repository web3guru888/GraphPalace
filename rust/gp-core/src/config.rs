//! Configuration types for GraphPalace: decay rates, cost weights, thresholds.

use serde::{Deserialize, Serialize};

/// Embedding dimension used throughout GraphPalace (all-MiniLM-L6-v2).
pub const EMBEDDING_DIM: usize = 384;

/// Pheromone decay rates per type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PheromoneConfig {
    /// Exploitation decay rate (node pheromone). Half-life ~35 cycles.
    pub exploitation_decay: f64,
    /// Exploration decay rate (node pheromone). Half-life ~14 cycles.
    pub exploration_decay: f64,
    /// Success decay rate (edge pheromone). Half-life ~69 cycles.
    pub success_decay: f64,
    /// Traversal decay rate (edge pheromone). Half-life ~23 cycles.
    pub traversal_decay: f64,
    /// Recency decay rate (edge pheromone). Half-life ~7 cycles.
    pub recency_decay: f64,
    /// How many cycles between bulk decay operations.
    pub decay_interval_cycles: usize,
    /// Maximum exploitation pheromone value (τ_max). Default 5.0.
    pub exploitation_max: f64,
    /// Maximum exploration pheromone value (τ_max). Default 3.0.
    pub exploration_max: f64,
    /// Maximum success pheromone value (τ_max). Default 5.0.
    pub success_max: f64,
    /// Maximum traversal pheromone value (τ_max). Default 2.0.
    pub traversal_max: f64,
    /// Maximum recency pheromone value (τ_max). Default 1.0.
    pub recency_max: f64,
}

impl Default for PheromoneConfig {
    fn default() -> Self {
        Self {
            exploitation_decay: 0.02,
            exploration_decay: 0.05,
            success_decay: 0.01,
            traversal_decay: 0.03,
            recency_decay: 0.10,
            decay_interval_cycles: 10,
            // Saturation ceilings (τ_max)
            exploitation_max: 5.0,
            exploration_max: 3.0,
            success_max: 5.0,
            traversal_max: 2.0,
            recency_max: 1.0,
        }
    }
}

/// Weights for the composite edge cost model (must sum to 1.0).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CostWeights {
    /// Weight for semantic similarity component.
    pub semantic: f64,
    /// Weight for pheromone guidance component.
    pub pheromone: f64,
    /// Weight for structural cost component.
    pub structural: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            semantic: 0.4,
            pheromone: 0.3,
            structural: 0.3,
        }
    }
}

impl CostWeights {
    /// Context-adaptive weight presets.
    pub fn hypothesis_testing() -> Self {
        Self { semantic: 0.30, pheromone: 0.40, structural: 0.30 }
    }

    pub fn exploratory_research() -> Self {
        Self { semantic: 0.50, pheromone: 0.20, structural: 0.30 }
    }

    pub fn evidence_gathering() -> Self {
        Self { semantic: 0.35, pheromone: 0.35, structural: 0.30 }
    }

    pub fn memory_recall() -> Self {
        Self { semantic: 0.50, pheromone: 0.30, structural: 0.20 }
    }
}

/// A* pathfinding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AStarConfig {
    /// Maximum iterations before giving up.
    pub max_iterations: usize,
    /// Similarity threshold below which cross-domain heuristic is used.
    pub cross_domain_threshold: f64,
}

impl Default for AStarConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            cross_domain_threshold: 0.3,
        }
    }
}

/// Active Inference agent defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Default temperature for softmax action selection.
    pub default_temperature: f64,
    /// Annealing schedule name.
    pub annealing_schedule: String,
    /// Prior mean for belief states.
    pub belief_prior_mean: f64,
    /// Prior precision for belief states (1/variance).
    pub belief_prior_precision: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            default_temperature: 0.5,
            annealing_schedule: "cosine".to_string(),
            belief_prior_mean: 20.0,
            belief_prior_precision: 0.1,
        }
    }
}

/// Swarm coordination configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub num_agents: usize,
    pub max_cycles: usize,
    pub convergence_window: usize,
    pub growth_threshold: f64,
    pub variance_threshold: f64,
    pub frontier_threshold: usize,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            num_agents: 5,
            max_cycles: 1000,
            convergence_window: 20,
            growth_threshold: 5.0,
            variance_threshold: 0.05,
            frontier_threshold: 10,
        }
    }
}

/// Cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub embedding_ttl_seconds: u64,
    pub search_ttl_seconds: u64,
    pub max_cache_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            embedding_ttl_seconds: 3600,
            search_ttl_seconds: 300,
            max_cache_entries: 10_000,
        }
    }
}

/// Palace-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceConfig {
    pub name: String,
    pub embedding_model: String,
    pub embedding_dim: usize,
}

impl Default for PalaceConfig {
    fn default() -> Self {
        Self {
            name: "My Palace".to_string(),
            embedding_model: "all-MiniLM-L6-v2".to_string(),
            embedding_dim: EMBEDDING_DIM,
        }
    }
}

/// Top-level configuration for GraphPalace.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct GraphPalaceConfig {
    pub palace: PalaceConfig,
    pub pheromones: PheromoneConfig,
    pub cost_weights: CostWeights,
    pub astar: AStarConfig,
    pub agents: AgentConfig,
    pub swarm: SwarmConfig,
    pub cache: CacheConfig,
}

