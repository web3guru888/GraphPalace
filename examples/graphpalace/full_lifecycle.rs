//! Full Palace Lifecycle
//!
//! Demonstrates the complete GraphPalace lifecycle: create → populate →
//! search → navigate → reinforce → decay → export → import → verify.
//!
//! Note: This example references GraphPalace crate APIs. It compiles
//! against the types but requires gp-palace and gp-storage crates.

use gp_core::types::{DrawerNode, EntityNode, PheromoneField, WingNode};
use gp_embeddings::MockEmbeddingEngine;
use gp_palace::{GraphPalace, ImportMode, NavigateOptions, PalaceBuilder, SearchOptions};
use gp_storage::InMemoryBackend;
use gp_stigmergy::DecayEngine;

fn main() {
    println!("=== GraphPalace Full Lifecycle Demo ===\n");

    // --- 1. Create palace with in-memory backend ---
    let embedding_engine = MockEmbeddingEngine::new(384);
    println!("1. Creating palace with in-memory backend...");
    let mut palace = PalaceBuilder::new()
        .name("Research Palace")
        .backend(InMemoryBackend::new())
        .embeddings(embedding_engine)
        .build()
        .expect("Failed to build palace");
    println!("   Palace created: {}", palace.name());

    // --- 2. Add wings and rooms ---
    println!("2. Building palace structure...");
    palace
        .create_wing("w-climate", "Climate Science", "domain", "Climate research")
        .unwrap();
    palace
        .create_wing("w-economics", "Economics", "domain", "Economic analysis")
        .unwrap();

    palace
        .create_room("w-climate", "r-temp", "Temperature Records", "facts")
        .unwrap();
    palace
        .create_room("w-climate", "r-events", "Climate Events", "events")
        .unwrap();
    palace
        .create_room("w-economics", "r-gdp", "GDP Analysis", "facts")
        .unwrap();

    palace
        .create_closet("r-temp", "c-global", "Global Averages", "Temperature data")
        .unwrap();
    palace
        .create_closet("r-gdp", "c-world", "World GDP", "GDP data")
        .unwrap();

    let status = palace.status().unwrap();
    println!(
        "   Structure: {} wings, {} rooms, {} closets",
        status.wings, status.rooms, status.closets
    );

    // --- 3. Store memories (drawers with verbatim content — never summarized) ---
    println!("3. Storing memories...");
    let memories = [
        ("c-global", "Global average temperature rose 1.1°C above pre-industrial levels by 2023.", "api", 0.9),
        ("c-global", "Arctic sea ice extent reached a record low of 3.41 million km² in September 2012.", "api", 0.8),
        ("c-global", "The rate of warming has accelerated to 0.2°C per decade since 1970.", "paper", 0.85),
        ("c-world", "Global GDP reached $105 trillion in 2023, with US and China comprising 43%.", "api", 0.7),
        ("c-world", "Climate change is projected to reduce global GDP by 10-23% by 2100.", "paper", 0.9),
    ];

    for (closet, content, source, importance) in &memories {
        palace.add_drawer(closet, content, source, *importance).unwrap();
    }
    println!("   Stored {} memories", memories.len());

    // --- 4. Search for memories ---
    println!("4. Searching for memories...");
    let results = palace
        .search(
            "how much has the planet warmed?",
            SearchOptions {
                k: 3,
                wing: Some("w-climate".to_string()),
                room: None,
                boost_pheromones: true,
            },
        )
        .unwrap();

    for (i, result) in results.iter().enumerate() {
        let preview = if result.drawer.content.len() > 70 {
            format!("{}...", &result.drawer.content[..70])
        } else {
            result.drawer.content.clone()
        };
        println!("   [{}] score={:.3}: {}", i + 1, result.score, preview);
    }

    // --- 5. Navigate between memories (Semantic A*) ---
    println!("5. Navigating between climate and economics...");
    if let Some(first_climate) = results.first() {
        let path = palace.navigate(
            &first_climate.drawer.id,
            "c-world", // Navigate toward economics
            NavigateOptions {
                context: Some("How does climate change affect GDP?".to_string()),
                max_iterations: 10_000,
            },
        );
        match path {
            Ok(p) => println!(
                "   Path found: {} steps, cost={:.3}, {} iterations",
                p.path.len(),
                p.total_cost,
                p.iterations
            ),
            Err(_) => println!("   No direct path (expected in small palaces)"),
        }
    }

    // --- 6. Deposit pheromones on successful paths ---
    println!("6. Depositing pheromones...");
    if let Some(result) = results.first() {
        palace.deposit_exploitation(&result.drawer.id, 0.5).unwrap();
        println!(
            "   Deposited exploitation pheromone (0.5) on {}",
            result.drawer.id
        );
    }
    palace.deposit_exploration("r-events", 0.3).unwrap();
    println!("   Deposited exploration pheromone (0.3) on r-events");

    // --- 7. Run decay cycle ---
    println!("7. Running pheromone decay...");
    let decay_stats = palace.decay().unwrap();
    println!(
        "   Decayed {} nodes, {} edges",
        decay_stats.nodes_decayed, decay_stats.edges_decayed
    );

    // --- 8. Export palace to JSON ---
    println!("8. Exporting palace...");
    let export = palace.export().unwrap();
    let json = serde_json::to_string_pretty(&export).unwrap();
    println!("   Exported {} bytes", json.len());

    // --- 9. Import into new palace ---
    println!("9. Importing into fresh palace...");
    let mut palace2 = PalaceBuilder::new()
        .name("Imported Palace")
        .backend(InMemoryBackend::new())
        .embeddings(MockEmbeddingEngine::new(384))
        .build()
        .unwrap();
    palace2.import(&export, ImportMode::Replace).unwrap();
    println!("   Import complete");

    // --- 10. Verify round-trip ---
    println!("10. Verifying round-trip...");
    let status1 = palace.status().unwrap();
    let status2 = palace2.status().unwrap();
    assert_eq!(status1.wings, status2.wings, "Wing count mismatch");
    assert_eq!(status1.rooms, status2.rooms, "Room count mismatch");
    assert_eq!(status1.drawers, status2.drawers, "Drawer count mismatch");
    println!(
        "   ✓ Round-trip verified: {} wings, {} rooms, {} drawers",
        status2.wings, status2.rooms, status2.drawers
    );

    println!("\n=== Full lifecycle complete! ===");
}
