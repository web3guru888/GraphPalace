//! Swarm coordination for GraphPalace (spec §10).
//!
//! Orchestrates multiple Active Inference agents to collectively explore
//! and reinforce knowledge paths in the memory palace graph.
//!
//! # Modules
//!
//! - [`coordinator`] — Multi-agent sense→decide→act→update cycle (§10.1)
//! - [`convergence`] — 3-criteria convergence detection (§10.3)
//! - [`interest`] — Node interest score computation (§10.2)
//! - [`decay_scheduler`] — Periodic pheromone decay scheduling

pub mod coordinator;
pub mod convergence;
pub mod interest;
pub mod decay_scheduler;

pub use coordinator::{SwarmCoordinator, CycleResult, AgentAction};
pub use convergence::{ConvergenceDetector, CycleHistory, CycleStats};
pub use interest::compute_interest_score;
pub use decay_scheduler::DecayScheduler;
