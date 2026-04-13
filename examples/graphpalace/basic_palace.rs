//! Basic Palace Operations
//!
//! Demonstrates creating a palace, adding wings/rooms/drawers,
//! and performing semantic search.
//!
//! Note: This example references GraphPalace crate APIs. It compiles
//! against the types but requires a live Kuzu backend to execute.

use gp_core::{
    config::PalaceConfig,
    types::{DrawerNode, PheromoneField, RoomNode, WingNode},
};
use gp_embeddings::EmbeddingEngine;

fn main() {
    // Load configuration
    let config = PalaceConfig::default();
    println!("Palace: {}", config.palace.name);
    println!("Embedding model: {}", config.palace.embedding_model);

    // Create wing
    let wing = WingNode::new("w-research", "Research", "domain")
        .with_description("Scientific research and discoveries");
    println!("Created wing: {} ({})", wing.name, wing.wing_type);

    // Create rooms in the wing
    let room_facts = RoomNode::new("r-climate-facts", "Climate Facts", "facts");
    let room_events = RoomNode::new("r-climate-events", "Climate Events", "events");
    println!("Created rooms: {}, {}", room_facts.name, room_events.name);

    // Create a drawer with verbatim content (NEVER summarized)
    let drawer = DrawerNode::new(
        "d-001",
        "Global average temperature rose 1.1°C above pre-industrial levels by 2023, \
         with the rate of warming accelerating to 0.2°C per decade since 1970.",
    )
    .with_source("api")
    .with_importance(0.8);
    println!("Created drawer: {} ({} chars)", drawer.id, drawer.content.len());

    // Check pheromone levels (all zero for new nodes)
    let pheromones = PheromoneField::default();
    println!(
        "Pheromones — exploitation: {:.2}, exploration: {:.2}",
        pheromones.exploitation, pheromones.exploration
    );

    // Demonstrate embedding generation (mock)
    let engine = gp_embeddings::MockEmbeddingEngine::new(384);
    let embedding = engine.encode("climate change temperature rise");
    println!("Generated embedding: {} dimensions", embedding.len());

    // Demonstrate similarity search
    let query = engine.encode("how much has temperature increased?");
    let sim = gp_embeddings::cosine_similarity(&query, &embedding);
    println!("Similarity: {:.4}", sim);

    println!("\nPalace ready for navigation!");
}
