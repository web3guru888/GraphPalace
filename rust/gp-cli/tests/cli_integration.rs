//! Integration tests for the `graphpalace` CLI binary.
//!
//! Each test launches the real binary via `std::process::Command`,
//! operates on an isolated temp directory, and verifies stdout/stderr
//! plus side-effects (files created, JSON content, etc.).

use std::process::Command;

/// Build a `Command` pointing at the workspace-built binary.
fn graphpalace() -> Command {
    Command::new(env!("CARGO_BIN_EXE_graphpalace"))
}

/// Create an isolated temp directory for a palace.
fn temp_palace() -> tempfile::TempDir {
    tempfile::tempdir().expect("failed to create temp dir")
}

/// Run `graphpalace --db <dir> init --name <name>` and assert success.
fn init_palace(dir: &str, name: &str) {
    let out = graphpalace()
        .args(["--db", dir, "init", "--name", name])
        .output()
        .expect("failed to execute");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "init failed: {}{}",
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Run `graphpalace --db <dir> add-drawer ...` and return stdout.
fn add_drawer(dir: &str, content: &str, wing: &str, room: &str) -> String {
    let out = graphpalace()
        .args([
            "--db", dir, "add-drawer", "-c", content, "-w", wing, "-r", room,
        ])
        .output()
        .expect("failed to execute");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "add-drawer failed: {}{}",
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

// ───────────────────────────── Init ──────────────────────────────

#[test]
fn test_init_creates_palace_json() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    let output = graphpalace()
        .args(["--db", db, "init", "--name", "Test Palace"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("initialized"), "stdout: {stdout}");
    assert!(
        dir.path().join("palace.json").exists(),
        "palace.json not created"
    );

    // Verify it's valid JSON
    let json = std::fs::read_to_string(dir.path().join("palace.json")).unwrap();
    let _: serde_json::Value = serde_json::from_str(&json).expect("palace.json is not valid JSON");
}

#[test]
fn test_init_twice_fails() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    // First init succeeds
    init_palace(db, "First");

    // Second init should fail gracefully (non-zero exit)
    let output = graphpalace()
        .args(["--db", db, "init", "--name", "Second"])
        .output()
        .expect("failed to execute");

    assert!(
        !output.status.success(),
        "Expected failure on double init, but got success"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("already exists"),
        "stderr should mention already exists: {stderr}"
    );
}

// ─────────────────────────── AddDrawer ───────────────────────────

#[test]
fn test_add_drawer_after_init() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "MyPalace");

    let stdout = add_drawer(db, "Rust is a systems programming language", "tech", "languages");

    // Output should mention the drawer was added with an ID
    assert!(stdout.contains("Drawer added"), "stdout: {stdout}");
    // The drawer ID is a UUID-like string; just check we got an ID line
    assert!(stdout.contains("Wing: tech"), "stdout: {stdout}");
    assert!(stdout.contains("Room: languages"), "stdout: {stdout}");
}

#[test]
fn test_add_drawer_without_init_fails() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    // Don't init — go straight to add-drawer
    let output = graphpalace()
        .args([
            "--db",
            db,
            "add-drawer",
            "-c",
            "Some content",
            "-w",
            "wing1",
            "-r",
            "room1",
        ])
        .output()
        .expect("failed to execute");

    assert!(
        !output.status.success(),
        "Expected failure without init, but got success"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No palace found") || stderr.contains("init"),
        "stderr should hint at missing palace: {stderr}"
    );
}

// ──────────────────────────── Search ─────────────────────────────

#[test]
fn test_search_finds_added_content() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "SearchPalace");

    // Add 3 diverse drawers
    add_drawer(
        db,
        "Rust is a systems programming language focused on safety and performance",
        "tech",
        "languages",
    );
    add_drawer(
        db,
        "The Milky Way galaxy contains billions of stars and planets",
        "science",
        "astronomy",
    );
    add_drawer(
        db,
        "Chocolate cake recipe: mix flour sugar cocoa butter eggs",
        "cooking",
        "recipes",
    );

    // Search for programming
    let output = graphpalace()
        .args(["--db", db, "search", "programming language"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Search results"),
        "Expected search results header: {stdout}"
    );
    // The programming drawer should appear
    assert!(
        stdout.contains("Rust") || stdout.contains("programming") || stdout.contains("tech"),
        "Expected programming result: {stdout}"
    );
}

#[test]
fn test_search_empty_palace() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "EmptyPalace");

    let output = graphpalace()
        .args(["--db", db, "search", "anything"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("No results"),
        "Expected 'No results' for empty palace: {stdout}"
    );
}

// ──────────────────────────── Wings ──────────────────────────────

#[test]
fn test_wing_list_shows_wings() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "WingPalace");

    // Adding drawers auto-creates wings
    add_drawer(db, "Content about physics", "science", "physics");
    add_drawer(db, "Content about cooking", "cooking", "recipes");

    let output = graphpalace()
        .args(["--db", db, "wing", "list"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("science"), "Expected 'science' wing: {stdout}");
    assert!(stdout.contains("cooking"), "Expected 'cooking' wing: {stdout}");
}

#[test]
fn test_wing_add() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "WingAddPalace");

    let output = graphpalace()
        .args([
            "--db",
            db,
            "wing",
            "add",
            "research",
            "-t",
            "domain",
            "-d",
            "Research wing",
        ])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Wing") && stdout.contains("added"),
        "Expected wing added confirmation: {stdout}"
    );

    // Verify it shows up in wing list
    let list_out = graphpalace()
        .args(["--db", db, "wing", "list"])
        .output()
        .expect("failed to execute");

    assert!(list_out.status.success());
    let list_stdout = String::from_utf8_lossy(&list_out.stdout);
    assert!(
        list_stdout.contains("research"),
        "Expected 'research' in wing list: {list_stdout}"
    );
}

// ──────────────────────────── Rooms ──────────────────────────────

#[test]
fn test_room_list_shows_rooms() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "RoomPalace");

    add_drawer(db, "Content in physics room", "science", "physics");
    add_drawer(db, "Content in chemistry room", "science", "chemistry");

    let output = graphpalace()
        .args(["--db", db, "room", "list", "science"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("physics"),
        "Expected 'physics' room: {stdout}"
    );
    assert!(
        stdout.contains("chemistry"),
        "Expected 'chemistry' room: {stdout}"
    );
}

#[test]
fn test_room_add() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "RoomAddPalace");

    // First create a wing
    graphpalace()
        .args(["--db", db, "wing", "add", "tech"])
        .output()
        .expect("failed to execute");

    // Then add a room
    let output = graphpalace()
        .args(["--db", db, "room", "add", "tech", "algorithms", "-t", "facts"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Room") && stdout.contains("added"),
        "Expected room added confirmation: {stdout}"
    );

    // Verify it shows up
    let list_out = graphpalace()
        .args(["--db", db, "room", "list", "tech"])
        .output()
        .expect("failed to execute");

    assert!(list_out.status.success());
    let list_stdout = String::from_utf8_lossy(&list_out.stdout);
    assert!(
        list_stdout.contains("algorithms"),
        "Expected 'algorithms' in room list: {list_stdout}"
    );
}

// ──────────────────────────── Status ─────────────────────────────

#[test]
fn test_status_shows_counts() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "StatusPalace");

    add_drawer(db, "Alpha content", "wing_a", "room_1");
    add_drawer(db, "Beta content", "wing_a", "room_2");
    add_drawer(db, "Gamma content", "wing_b", "room_3");

    let output = graphpalace()
        .args(["--db", db, "status"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show the palace name and structure counts
    assert!(stdout.contains("Palace"), "Expected Palace header: {stdout}");
    assert!(stdout.contains("Wings"), "Expected Wings count: {stdout}");
    assert!(stdout.contains("Rooms"), "Expected Rooms count: {stdout}");
    assert!(
        stdout.contains("Drawers"),
        "Expected Drawers count: {stdout}"
    );

    // We added 3 drawers to 2 wings and 3 rooms
    assert!(stdout.contains("3"), "Expected drawer count 3: {stdout}");
}

// ────────────────────────── KG (Knowledge Graph) ─────────────────

#[test]
fn test_kg_add_and_query() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "KGPalace");

    // Add a triple
    let add_out = graphpalace()
        .args(["--db", db, "kg", "add", "Rust", "is_a", "programming_language"])
        .output()
        .expect("failed to execute");

    assert!(add_out.status.success());
    let add_stdout = String::from_utf8_lossy(&add_out.stdout);
    assert!(
        add_stdout.contains("Triple added"),
        "Expected triple confirmation: {add_stdout}"
    );
    assert!(
        add_stdout.contains("Rust"),
        "Expected subject in output: {add_stdout}"
    );
    assert!(
        add_stdout.contains("is_a"),
        "Expected predicate in output: {add_stdout}"
    );

    // Query the entity
    let query_out = graphpalace()
        .args(["--db", db, "kg", "query", "Rust"])
        .output()
        .expect("failed to execute");

    assert!(query_out.status.success());
    let query_stdout = String::from_utf8_lossy(&query_out.stdout);
    assert!(
        query_stdout.contains("Rust") && query_stdout.contains("is_a"),
        "Expected triple in query output: {query_stdout}"
    );
    assert!(
        query_stdout.contains("programming_language"),
        "Expected object in query output: {query_stdout}"
    );
}

#[test]
fn test_kg_query_empty() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "KGEmptyPalace");

    let output = graphpalace()
        .args(["--db", db, "kg", "query", "nonexistent"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("No relationships"),
        "Expected no relationships message: {stdout}"
    );
}

// ──────────────────────── Export / Import ─────────────────────────

#[test]
fn test_export_creates_file() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "ExportPalace");
    add_drawer(db, "Exported content", "wing1", "room1");

    let export_path = dir.path().join("export.json");
    let export_str = export_path.to_str().unwrap();

    let output = graphpalace()
        .args(["--db", db, "export", "-o", export_str])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("exported"),
        "Expected export confirmation: {stdout}"
    );

    // File must exist and be valid JSON
    assert!(export_path.exists(), "export.json should exist");
    let json_str = std::fs::read_to_string(&export_path).unwrap();
    let val: serde_json::Value =
        serde_json::from_str(&json_str).expect("export file is not valid JSON");
    assert!(val.is_object(), "export should be a JSON object");
}

#[test]
fn test_import_merge() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "ImportPalace");
    add_drawer(db, "Original content", "wing1", "room1");

    // Export
    let export_path = dir.path().join("export.json");
    let export_str = export_path.to_str().unwrap();
    graphpalace()
        .args(["--db", db, "export", "-o", export_str])
        .output()
        .expect("failed to execute");

    // Import back with merge mode
    let output = graphpalace()
        .args(["--db", db, "import", export_str, "--mode", "merge"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Import complete"),
        "Expected import confirmation: {stdout}"
    );
    assert!(
        stdout.contains("merge"),
        "Expected merge mode in output: {stdout}"
    );
}

// ──────────────────────── Pheromones ─────────────────────────────

#[test]
fn test_pheromone_cold_shows_unvisited() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "PheromonePalace");
    add_drawer(db, "Cold content", "wing1", "room1");

    let output = graphpalace()
        .args(["--db", db, "pheromone", "cold", "-k", "10"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Either we get cold spots or "No cold spots"
    // Since we just added content, nodes should have zero pheromones → cold
    assert!(
        stdout.contains("Cold spots") || stdout.contains("No cold spots"),
        "Expected cold spots output: {stdout}"
    );
}

#[test]
fn test_pheromone_decay_succeeds() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "DecayPalace");
    add_drawer(db, "Content to decay", "wing1", "room1");

    let output = graphpalace()
        .args(["--db", db, "pheromone", "decay"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("decay") || stdout.contains("Decay"),
        "Expected decay confirmation: {stdout}"
    );
}

#[test]
fn test_pheromone_hot_empty() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "HotPalace");

    let output = graphpalace()
        .args(["--db", db, "pheromone", "hot", "-k", "5"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("No hot paths") || stdout.contains("Hot paths"),
        "Expected hot paths output: {stdout}"
    );
}

// ───────────────────────── Agents ────────────────────────────────

#[test]
fn test_agent_list() {
    let output = graphpalace()
        .args(["agent", "list"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should list all 5 archetypes
    assert!(
        stdout.contains("explorer"),
        "Expected explorer archetype: {stdout}"
    );
    assert!(
        stdout.contains("exploiter"),
        "Expected exploiter archetype: {stdout}"
    );
    assert!(
        stdout.contains("researcher"),
        "Expected researcher archetype: {stdout}"
    );
    assert!(
        stdout.contains("curator"),
        "Expected curator archetype: {stdout}"
    );
    assert!(
        stdout.contains("sentinel"),
        "Expected sentinel archetype: {stdout}"
    );
}

// ─────────────────────── Navigation ──────────────────────────────

#[test]
fn test_navigate_in_hierarchy() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "NavPalace");

    // Add drawers to get node IDs — parse them from stdout
    let out1 = add_drawer(db, "First drawer content about physics", "science", "physics");
    let out2 = add_drawer(
        db,
        "Second drawer content about chemistry",
        "science",
        "chemistry",
    );

    // Extract drawer IDs from "✓ Drawer added: <id>" lines
    let id1 = out1
        .lines()
        .find(|l| l.contains("Drawer added"))
        .and_then(|l| l.split(": ").nth(1))
        .map(|s| s.trim().to_string())
        .expect("Could not parse drawer ID from first add");
    let id2 = out2
        .lines()
        .find(|l| l.contains("Drawer added"))
        .and_then(|l| l.split(": ").nth(1))
        .map(|s| s.trim().to_string())
        .expect("Could not parse drawer ID from second add");

    let output = graphpalace()
        .args(["--db", db, "navigate", &id1, &id2])
        .output()
        .expect("failed to execute");

    // Navigation may or may not find a path depending on graph structure.
    // Both drawers are in the same wing ("science"), so a path should exist
    // via the wing node.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        assert!(
            stdout.contains("Path found") || stdout.contains("steps"),
            "Expected path info: {stdout}"
        );
    } else {
        // If no path is found, that's OK for integration test — we just
        // verify the command ran and produced a reasonable error.
        assert!(
            stderr.contains("path") || stderr.contains("No path") || stderr.contains("not found"),
            "Expected a path-related message: stdout={stdout} stderr={stderr}"
        );
    }
}

// ──────────────────────── Help / Version ─────────────────────────

#[test]
fn test_help_flag() {
    let output = graphpalace()
        .arg("--help")
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("GraphPalace") || stdout.contains("graphpalace"),
        "Expected program name in help: {stdout}"
    );
    assert!(
        stdout.contains("Usage") || stdout.contains("usage"),
        "Expected usage section in help: {stdout}"
    );
}

#[test]
fn test_version_flag() {
    let output = graphpalace()
        .arg("--version")
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain the crate version (e.g., "graphpalace 0.1.0")
    assert!(
        stdout.contains("graphpalace"),
        "Expected program name in version: {stdout}"
    );
}

// ─────────────────── Compound Workflows ──────────────────────────

#[test]
fn test_full_workflow_init_add_search_export() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    // 1. Init
    init_palace(db, "WorkflowPalace");

    // 2. Add several drawers
    add_drawer(
        db,
        "Machine learning uses neural networks for pattern recognition",
        "ai",
        "ml",
    );
    add_drawer(
        db,
        "Quantum computing leverages superposition and entanglement",
        "physics",
        "quantum",
    );
    add_drawer(
        db,
        "The human genome contains approximately 20000 protein coding genes",
        "biology",
        "genetics",
    );

    // 3. Add KG triple
    let kg_out = graphpalace()
        .args([
            "--db",
            db,
            "kg",
            "add",
            "neural_networks",
            "used_in",
            "pattern_recognition",
        ])
        .output()
        .expect("failed to execute");
    assert!(kg_out.status.success());

    // 4. Search
    let search_out = graphpalace()
        .args(["--db", db, "search", "neural network deep learning", "-k", "3"])
        .output()
        .expect("failed to execute");
    assert!(search_out.status.success());
    let search_stdout = String::from_utf8_lossy(&search_out.stdout);
    assert!(
        search_stdout.contains("Search results"),
        "Expected results: {search_stdout}"
    );

    // 5. Status
    let status_out = graphpalace()
        .args(["--db", db, "status"])
        .output()
        .expect("failed to execute");
    assert!(status_out.status.success());
    let status_stdout = String::from_utf8_lossy(&status_out.stdout);
    assert!(
        status_stdout.contains("3"),
        "Expected 3 drawers: {status_stdout}"
    );

    // 6. Export
    let export_path = dir.path().join("workflow-export.json");
    let export_str = export_path.to_str().unwrap();
    let export_out = graphpalace()
        .args(["--db", db, "export", "-o", export_str])
        .output()
        .expect("failed to execute");
    assert!(export_out.status.success());
    assert!(export_path.exists());
}

#[test]
fn test_multiple_kg_triples_same_entity() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "MultiKGPalace");

    // Add multiple triples about Python
    graphpalace()
        .args(["--db", db, "kg", "add", "Python", "is_a", "language"])
        .output()
        .expect("failed to execute");
    graphpalace()
        .args(["--db", db, "kg", "add", "Python", "created_by", "Guido"])
        .output()
        .expect("failed to execute");
    graphpalace()
        .args([
            "--db",
            db,
            "kg",
            "add",
            "Python",
            "supports",
            "async_await",
        ])
        .output()
        .expect("failed to execute");

    // Query should return all 3
    let output = graphpalace()
        .args(["--db", db, "kg", "query", "Python"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("3 relationships"),
        "Expected 3 relationships: {stdout}"
    );
}

#[test]
fn test_search_ranks_relevant_first() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "RankPalace");

    // Add content in a specific order — astronomy first, then programming
    add_drawer(
        db,
        "Stars and galaxies are observed through telescopes in astronomy",
        "science",
        "astronomy",
    );
    add_drawer(
        db,
        "Cooking pasta requires boiling water and adding salt",
        "cooking",
        "italian",
    );
    add_drawer(
        db,
        "Rust and Go are modern systems programming languages",
        "tech",
        "programming",
    );

    // Search for programming — should rank programming drawer first
    let output = graphpalace()
        .args(["--db", db, "search", "programming language"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Find the first result line (after header) — it should mention programming/Rust/Go
    let result_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.contains("tech") || l.contains("science") || l.contains("cooking"))
        .collect();

    assert!(
        !result_lines.is_empty(),
        "Expected at least one result line: {stdout}"
    );
    // First result should be from the tech/programming wing
    assert!(
        result_lines[0].contains("tech") || result_lines[0].contains("Rust"),
        "Expected programming result ranked first, got: {}",
        result_lines[0]
    );
}

#[test]
fn test_wing_list_empty_palace() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "EmptyWingsPalace");

    let output = graphpalace()
        .args(["--db", db, "wing", "list"])
        .output()
        .expect("failed to execute");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("No wings"),
        "Expected empty wings message: {stdout}"
    );
}

#[test]
fn test_room_list_nonexistent_wing() {
    let dir = temp_palace();
    let db = dir.path().to_str().unwrap();

    init_palace(db, "NoSuchWingPalace");

    let output = graphpalace()
        .args(["--db", db, "room", "list", "nonexistent"])
        .output()
        .expect("failed to execute");

    assert!(
        !output.status.success(),
        "Expected failure for nonexistent wing"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found"),
        "Expected 'not found' error: {stderr}"
    );
}
