//! Agent framework for GraphPalace.
//!
//! Ant-like agents that traverse graphs, deposit pheromones, and learn
//! from paths. Implements Active Inference (spec §6) with Bayesian
//! belief updates, Expected Free Energy action selection, and
//! temperature-annealed softmax policies.

pub mod action_selection;
pub mod active_inference;
pub mod archetypes;
pub mod beliefs;
pub mod generative_model;

// Re-export key types at crate root for convenience.
pub use action_selection::{select_action, softmax_probabilities, AnnealingSchedule};
pub use active_inference::{expected_free_energy, ActiveInferenceAgent};
pub use archetypes::AgentArchetype;
pub use beliefs::BeliefState;
pub use generative_model::{GenerativeModel, WelfordStats};
