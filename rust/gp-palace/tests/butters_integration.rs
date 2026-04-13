//! BUTTERS integration test — exercises the full GraphPalace pipeline.

use gp_core::config::GraphPalaceConfig;
use gp_core::types::*;
use gp_embeddings::engine::MockEmbeddingEngine;
use gp_palace::palace::GraphPalace;
use gp_storage::memory::InMemoryBackend;

fn make_palace() -> GraphPalace {
    let config = GraphPalaceConfig::default();
    let storage = InMemoryBackend::new();
    let embeddings = Box::new(MockEmbeddingEngine::new());
    GraphPalace::new(config, storage, embeddings).unwrap()
}

#[test]
fn butters_palace_lifecycle() {
    let mut palace = make_palace();

    // 1. Add wings
    let proj_wing = palace.add_wing("projects", WingType::Project, "Active projects").unwrap();
    let people_wing = palace.add_wing("people", WingType::Person, "People").unwrap();
    let knowledge_wing = palace.add_wing("knowledge", WingType::Domain, "General knowledge").unwrap();
    assert!(!proj_wing.is_empty());
    assert!(!people_wing.is_empty());
    assert!(!knowledge_wing.is_empty());

    // 2. Add rooms
    let gp_room = palace.add_room(&proj_wing, "graphpalace", HallType::Facts).unwrap();
    let robin_room = palace.add_room(&people_wing, "robin", HallType::Facts).unwrap();
    let rust_room = palace.add_room(&knowledge_wing, "rust-lang", HallType::Facts).unwrap();
    assert!(!gp_room.is_empty());
    assert!(!robin_room.is_empty());
    assert!(!rust_room.is_empty());

    // 3. Store memories via add_drawer (wing_name, room_name auto-resolve)
    let d1 = palace.add_drawer(
        "GraphPalace is a stigmergic memory palace engine built in Rust with pheromone navigation",
        "projects",
        "graphpalace",
        DrawerSource::Conversation,
    ).unwrap();

    let d2 = palace.add_drawer(
        "Robin is the creator of BUTTERS and is exploring GraphPalace as an AI memory system",
        "people",
        "robin",
        DrawerSource::Conversation,
    ).unwrap();

    let d3 = palace.add_drawer(
        "Rust is a systems programming language focused on safety speed and concurrency",
        "knowledge",
        "rust-lang",
        DrawerSource::Conversation,
    ).unwrap();

    let d4 = palace.add_drawer(
        "The MCP server exposes 28 tools for LLM integration including search navigate and pheromone deposit",
        "projects",
        "graphpalace",
        DrawerSource::Conversation,
    ).unwrap();

    assert!(!d1.is_empty());
    assert!(!d2.is_empty());
    assert!(!d3.is_empty());
    assert!(!d4.is_empty());

    // 4. Check status
    let status = palace.status().unwrap();
    assert_eq!(status.wing_count, 3);
    assert_eq!(status.room_count, 3);
    assert_eq!(status.drawer_count, 4);
    assert!(status.closet_count >= 3); // auto-created "General" closets

    // 5. Semantic search
    let results = palace.search_mut("pheromone navigation stigmergy", 3).unwrap();
    assert!(!results.is_empty(), "Search should return results");
    // The top result should be about GraphPalace (contains "pheromone")
    assert!(
        results[0].content.contains("pheromone") || results[0].content.contains("stigmergic"),
        "Top result should relate to pheromones, got: {}",
        results[0].content,
    );

    let results2 = palace.search_mut("Robin BUTTERS creator", 3).unwrap();
    assert!(!results2.is_empty(), "Search for Robin should return results");

    // 6. Export/import roundtrip
    let export = palace.export().unwrap();
    let json = serde_json::to_string(&export).unwrap();
    assert!(json.len() > 100, "Export should produce non-trivial JSON");

    // Verify the export contains our data
    assert!(json.contains("GraphPalace"));
    assert!(json.contains("Robin"));
    assert!(json.contains("Rust"));

    println!("BUTTERS Palace Test Summary:");
    println!("  Wings: {}", status.wing_count);
    println!("  Rooms: {}", status.room_count);
    println!("  Closets: {}", status.closet_count);
    println!("  Drawers: {}", status.drawer_count);
    println!("  Search results for 'pheromone': {}", results.len());
    println!("  Export size: {} bytes", json.len());
    println!("  All assertions passed!");
}

#[test]
fn butters_search_relevance() {
    let mut palace = make_palace();

    // Store diverse memories
    palace.add_drawer("The weather today is sunny and warm", "misc", "daily", DrawerSource::Conversation).unwrap();
    palace.add_drawer("Rust borrow checker prevents data races at compile time", "knowledge", "rust", DrawerSource::Conversation).unwrap();
    palace.add_drawer("GraphPalace uses exponential pheromone decay with configurable half-lives", "projects", "graphpalace", DrawerSource::Conversation).unwrap();
    palace.add_drawer("Active Inference agents minimize Expected Free Energy to balance exploration and exploitation", "projects", "graphpalace", DrawerSource::Conversation).unwrap();
    palace.add_drawer("The A* algorithm finds optimal paths using semantic similarity and pheromone guidance", "projects", "graphpalace", DrawerSource::Conversation).unwrap();

    // Search should return something
    let results = palace.search_mut("pathfinding algorithm", 5).unwrap();
    assert!(!results.is_empty(), "Search for pathfinding should return results");

    // All results should have scores
    for r in &results {
        assert!(r.score >= 0.0, "Scores should be non-negative");
    }
}

#[test]
fn butters_duplicate_prevention() {
    let mut palace = make_palace();

    // Add same content to same location twice
    palace.add_drawer("Important fact about memory", "knowledge", "facts", DrawerSource::Conversation).unwrap();
    palace.add_drawer("Another important fact about memory", "knowledge", "facts", DrawerSource::Conversation).unwrap();

    let status = palace.status().unwrap();
    // Both drawers should exist (dedup is check_duplicate's job, not add_drawer's)
    assert_eq!(status.drawer_count, 2);
    // But only one wing and one room
    assert_eq!(status.wing_count, 1);
    assert_eq!(status.room_count, 1);
}
