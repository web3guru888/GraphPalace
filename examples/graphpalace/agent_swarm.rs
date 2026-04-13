//! Agent Swarm
//!
//! Demonstrates running a multi-agent swarm cycle with Active Inference
//! agents exploring a memory palace.
//!
//! Note: This example references GraphPalace crate APIs. It compiles
//! against the types but requires a live Kuzu backend to execute.

use gp_agents::{
    archetypes::Archetype,
    beliefs::BeliefState,
    ActiveInferenceAgent,
};
use gp_core::config::AgentConfig;

fn main() {
    let config = AgentConfig::default();
    println!("Agent configuration:");
    println!("  Default temperature: {}", config.default_temperature);
    println!("  Annealing schedule: {}", config.annealing_schedule);
    println!("  Prior mean: {}", config.belief_prior_mean);
    println!("  Prior precision: {}", config.belief_prior_precision);

    // Create agents of different archetypes
    let archetypes = [
        ("scout-1", Archetype::Explorer),
        ("analyst-1", Archetype::Exploiter),
        ("general-1", Archetype::Balanced),
        ("climate-specialist", Archetype::Specialist),
        ("cross-domain", Archetype::Generalist),
    ];

    println!("\nSwarm agents:");
    for (name, archetype) in &archetypes {
        let agent = ActiveInferenceAgent::from_archetype(name, archetype.clone());
        println!("  {} — temp={:.1}, archetype={:?}",
            agent.name, agent.temperature, archetype);
    }

    // Demonstrate belief updates
    println!("\nBayesian belief update demo:");
    let mut belief = BeliefState::new(20.0, 0.1); // Optimistic prior
    println!("  Prior: mean={:.1}, precision={:.1}", belief.mean, belief.precision);

    // Agent visits a node and observes value = 5.0 (disappointing)
    belief.update(5.0, 1.0);
    println!("  After observation (5.0, precision 1.0): mean={:.2}, precision={:.2}",
        belief.mean, belief.precision);

    // Agent visits again, observes value = 8.0 (better)
    belief.update(8.0, 2.0);
    println!("  After observation (8.0, precision 2.0): mean={:.2}, precision={:.2}",
        belief.mean, belief.precision);

    // Demonstrate belief merging (multi-agent consensus)
    println!("\nBelief merging (2 agents observing same node):");
    let belief_a = BeliefState::new(15.0, 3.0);
    let belief_b = BeliefState::new(10.0, 5.0);
    let merged = BeliefState::merge(&[&belief_a, &belief_b]);
    println!("  Agent A: mean={:.1}, precision={:.1}", belief_a.mean, belief_a.precision);
    println!("  Agent B: mean={:.1}, precision={:.1}", belief_b.mean, belief_b.precision);
    println!("  Merged:  mean={:.2}, precision={:.1}", merged.mean, merged.precision);

    // Show EFE computation intuition
    println!("\nExpected Free Energy components:");
    println!("  Epistemic (1/precision):  {:.2} — 'How uncertain are we?'", 1.0 / merged.precision);
    println!("  Pragmatic (similarity):   0.75 — 'How close to goal?'");
    println!("  Edge quality (pheromone):  0.30 — 'Collective intelligence signal'");
    println!("  EFE = -(0.13 + 0.75 + 0.30) = {:.2}", -(1.0/merged.precision + 0.75 + 0.30));

    println!("\nSwarm ready for autonomous exploration!");
}
