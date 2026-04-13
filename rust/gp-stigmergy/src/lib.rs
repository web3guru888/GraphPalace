//! Stigmergic pheromone dynamics for GraphPalace.
//!
//! Deposit, evaporate, diffuse, and query pheromone fields on graph edges.
//!
//! # Modules
//!
//! - [`pheromones`]: Pheromone type enum and field accessors
//! - [`decay`]: Exponential pheromone decay (§4.2)
//! - [`rewards`]: Position-weighted path reward deposits (§4.3)
//! - [`cost`]: Pheromone-modulated edge cost recomputation (§4.4)
//! - [`cypher`]: Cypher query generation for bulk pheromone operations (§4.2)

pub mod cost;
pub mod cypher;
pub mod decay;
pub mod pheromones;
pub mod rewards;

// Re-export key items at crate root for convenience.
pub use cost::{pheromone_factor, recompute_edge_cost};
pub use cypher::{CypherQuery, CypherValue};
pub use decay::{
    bulk_decay_edges, bulk_decay_nodes, decay, decay_edge_pheromones, decay_node_pheromones,
    PHEROMONE_FLOOR,
};
pub use pheromones::{
    default_decay_rate, get_edge_pheromone, get_node_pheromone, set_edge_pheromone,
    set_node_pheromone, PheromoneType,
};
pub use rewards::{deposit_exploration, deposit_path_success};
