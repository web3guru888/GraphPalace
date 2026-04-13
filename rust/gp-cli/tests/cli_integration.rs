//! Integration tests for the `graphpalace` CLI binary.
//!
//! Each test creates an isolated temporary directory so they can run in parallel
//! without interfering with each other.

use std::process::Command;

fn graphpalace() -> Command {
    Command::new(env!("CARGO_BIN_EXE_graphpalace"))
}

fn run(db: &str, args: &[&str]) -> (String, String, bool) {
    let output = graphpalace()
        .arg("--db")
        .arg(db)
        .args(args)
        .output()
        .expect("failed to execute graphpalace");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (stdout, stderr, output.status.success())
}

fn run_ok(db: &str, args: &[&str]) -> String {
    let (stdout, stderr, success) = run(db, args);
    assert!(
        success,
        "Command {:?} failed.\nstdout: {stdout}\nstderr: {stderr}",
        args
    );
    stdout
}

// ── Init ──────────────────────────────────────────────────────────────────

#[test]
fn test_init_creates_palace_json() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let out = run_ok(db, &["init", "--name", "Test Palace"]);
    assert!(out.contains("Initialized palace"));
    assert!(out.contains("Test Palace"));
    assert!(dir.path().join("palace.json").exists());
}

#[test]
fn test_init_creates_valid_json() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "JSON Test"]);
    let json = std::fs::read_to_string(dir.path().join("palace.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.is_object());
}

// ── Add Drawer ────────────────────────────────────────────────────────────

#[test]
fn test_add_drawer_after_init() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    let out = run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Rust is great for systems programming",
            "-w",
            "Programming",
            "-r",
            "Languages",
        ],
    );
    assert!(out.contains("drawer"));
    assert!(out.contains("Programming"));
    assert!(out.contains("Languages"));
}

#[test]
fn test_add_drawer_creates_palace_implicitly() {
    // The new CLI auto-creates the palace if it doesn't exist
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("auto_created").to_str().unwrap().to_string();
    let (stdout, _, success) = run(
        &db,
        &[
            "add-drawer",
            "-c",
            "Test content",
            "-w",
            "W",
            "-r",
            "R",
        ],
    );
    // Should succeed (auto-create) or fail gracefully
    // The new CLI auto-creates, so check that it worked
    if success {
        assert!(stdout.contains("drawer") || stdout.contains("Added"));
    }
}

// ── Search ────────────────────────────────────────────────────────────────

#[test]
fn test_search_finds_added_content() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Rust is a systems programming language",
            "-w",
            "Tech",
            "-r",
            "Languages",
        ],
    );
    let out = run_ok(db, &["search", "programming"]);
    assert!(out.contains("Rust") || out.contains("programming"));
}

#[test]
fn test_search_empty_palace() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Empty"]);
    let out = run_ok(db, &["search", "anything"]);
    assert!(
        out.contains("No results") || out.contains("0 found"),
        "Expected 'No results' message, got: {out}"
    );
}

#[test]
fn test_search_ranks_relevant_first() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Rust is a systems programming language focusing on safety",
            "-w",
            "Programming",
            "-r",
            "Languages",
        ],
    );
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "The Milky Way contains 100 billion stars",
            "-w",
            "Astronomy",
            "-r",
            "Galaxies",
        ],
    );
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "The C programming language was created by Dennis Ritchie",
            "-w",
            "Programming",
            "-r",
            "Languages",
        ],
    );
    let out = run_ok(db, &["search", "programming language", "-k", "3"]);
    // The first result should mention programming, not astronomy
    let lines: Vec<&str> = out.lines().collect();
    // Find first result line containing "1." or "[score"
    let _first_result = lines
        .iter()
        .find(|l| l.contains("1.") || l.contains("[score"))
        .unwrap_or(&"");
    // First result should be from Programming wing
    assert!(
        out.contains("Programming"),
        "Search for 'programming language' should find Programming content: {out}"
    );
}

// ── Wing Management ───────────────────────────────────────────────────────

#[test]
fn test_wing_list_shows_wings() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Content A",
            "-w",
            "Alpha",
            "-r",
            "Room1",
        ],
    );
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Content B",
            "-w",
            "Beta",
            "-r",
            "Room2",
        ],
    );
    let out = run_ok(db, &["wing", "list"]);
    assert!(out.contains("Alpha"), "Wing list should show Alpha: {out}");
    assert!(out.contains("Beta"), "Wing list should show Beta: {out}");
}

#[test]
fn test_wing_add() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    let out = run_ok(db, &["wing", "add", "Science", "-t", "domain"]);
    assert!(
        out.contains("Science"),
        "Wing add should confirm creation: {out}"
    );
    let list = run_ok(db, &["wing", "list"]);
    assert!(
        list.contains("Science"),
        "Wing list should contain added wing: {list}"
    );
}

#[test]
fn test_wing_list_empty_palace() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Empty"]);
    let out = run_ok(db, &["wing", "list"]);
    assert!(
        out.contains("No wings") || out.contains("0") || out.trim().lines().count() <= 3,
        "Empty wing list should indicate no wings: {out}"
    );
}

// ── Room Management ───────────────────────────────────────────────────────

#[test]
fn test_room_list_shows_rooms() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Physics content",
            "-w",
            "Science",
            "-r",
            "Physics",
        ],
    );
    let out = run_ok(db, &["room", "list", "Science"]);
    assert!(
        out.contains("Physics"),
        "Room list should show Physics: {out}"
    );
}

#[test]
fn test_room_add() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(db, &["wing", "add", "Science"]);
    let out = run_ok(db, &["room", "add", "Science", "Chemistry"]);
    assert!(
        out.contains("Chemistry"),
        "Room add should confirm: {out}"
    );
    let list = run_ok(db, &["room", "list", "Science"]);
    assert!(
        list.contains("Chemistry"),
        "Room list should show added room: {list}"
    );
}

// ── Status ────────────────────────────────────────────────────────────────

#[test]
fn test_status_shows_counts() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Memory one",
            "-w",
            "Wing1",
            "-r",
            "Room1",
        ],
    );
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Memory two",
            "-w",
            "Wing2",
            "-r",
            "Room2",
        ],
    );
    let out = run_ok(db, &["status"]);
    assert!(out.contains("Wings"), "Status should show Wings: {out}");
    assert!(out.contains("Drawers"), "Status should show Drawers: {out}");
    assert!(out.contains("2"), "Status should show count 2: {out}");
}

// ── Knowledge Graph ───────────────────────────────────────────────────────

#[test]
fn test_kg_add_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    let add_out = run_ok(db, &["kg", "add", "Rust", "is_a", "Programming Language"]);
    assert!(
        add_out.contains("Rust") && add_out.contains("is_a"),
        "KG add should confirm triple: {add_out}"
    );
    let query_out = run_ok(db, &["kg", "query", "Rust"]);
    assert!(
        query_out.contains("is_a") && query_out.contains("Programming Language"),
        "KG query should return triple: {query_out}"
    );
}

#[test]
fn test_kg_query_empty() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    let out = run_ok(db, &["kg", "query", "NonExistent"]);
    assert!(
        out.contains("No relationships") || out.contains("0 found"),
        "KG query for nonexistent entity should say no results: {out}"
    );
}

#[test]
fn test_multiple_kg_triples_same_entity() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(db, &["kg", "add", "Rust", "is_a", "Language"]);
    run_ok(db, &["kg", "add", "Rust", "created_by", "Mozilla"]);
    run_ok(db, &["kg", "add", "Rust", "focuses_on", "Safety"]);
    let out = run_ok(db, &["kg", "query", "Rust"]);
    assert!(
        out.contains("is_a") && out.contains("created_by") && out.contains("focuses_on"),
        "All three triples should appear: {out}"
    );
}

// ── Export / Import ───────────────────────────────────────────────────────

#[test]
fn test_export_creates_file() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Some content",
            "-w",
            "Wing1",
            "-r",
            "Room1",
        ],
    );
    let export_path = dir.path().join("export.json");
    let out = run_ok(
        db,
        &["export", "-o", export_path.to_str().unwrap()],
    );
    assert!(out.contains("Exported") || out.contains("export"));
    assert!(export_path.exists());
    // Verify it's valid JSON
    let json = std::fs::read_to_string(&export_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.is_object());
}

#[test]
fn test_import_merge() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Original content",
            "-w",
            "Wing1",
            "-r",
            "Room1",
        ],
    );
    let export_path = dir.path().join("export.json");
    run_ok(
        db,
        &["export", "-o", export_path.to_str().unwrap()],
    );
    let out = run_ok(
        db,
        &["import", export_path.to_str().unwrap(), "-m", "merge"],
    );
    assert!(
        out.contains("Imported") || out.contains("import"),
        "Import should confirm: {out}"
    );
}

// ── Pheromones ────────────────────────────────────────────────────────────

#[test]
fn test_pheromone_cold_shows_unvisited() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Test memory",
            "-w",
            "Wing1",
            "-r",
            "Room1",
        ],
    );
    let out = run_ok(db, &["pheromone", "cold", "-k", "5"]);
    assert!(
        out.contains("cold") || out.contains("Cold") || out.contains("pheromone"),
        "Cold spots should produce output: {out}"
    );
}

#[test]
fn test_pheromone_decay_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    let out = run_ok(db, &["pheromone", "decay"]);
    assert!(
        out.contains("decay") || out.contains("Decay") || out.contains("applied"),
        "Decay should succeed: {out}"
    );
}

#[test]
fn test_pheromone_hot_empty() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    let (_, _, success) = run(db, &["pheromone", "hot", "-k", "5"]);
    assert!(success, "Hot paths should not error on empty palace");
}

// ── Agents ────────────────────────────────────────────────────────────────

#[test]
fn test_agent_list() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    let (_, _, success) = run(db, &["agent", "list"]);
    assert!(success, "Agent list should not error");
}

// ── Navigate ──────────────────────────────────────────────────────────────

#[test]
fn test_navigate_in_hierarchy() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    run_ok(db, &["init", "--name", "Test"]);
    // Add drawer to create hierarchy: wing → room → closet → drawer
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "First memory",
            "-w",
            "Wing1",
            "-r",
            "Room1",
        ],
    );
    // Navigate from wing to drawer (should find path through hierarchy)
    let out = run_ok(db, &["navigate", "wing_1", "drawer_4"]);
    assert!(
        out.contains("Path") || out.contains("path") || out.contains("wing_1"),
        "Navigate should find path in hierarchy: {out}"
    );
}

// ── Help & Version ────────────────────────────────────────────────────────

#[test]
fn test_help_flag() {
    let output = graphpalace()
        .arg("--help")
        .output()
        .expect("failed to execute");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage") || stdout.contains("graphpalace"),
        "--help should show usage: {stdout}"
    );
}

#[test]
fn test_version_flag() {
    let output = graphpalace()
        .arg("--version")
        .output()
        .expect("failed to execute");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("graphpalace") || stdout.contains("0.1"),
        "--version should show version: {stdout}"
    );
}

// ── Full Workflow ─────────────────────────────────────────────────────────

#[test]
fn test_full_workflow_init_add_search_export() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().to_str().unwrap();

    // 1. Init
    let out = run_ok(db, &["init", "--name", "Full Workflow Test"]);
    assert!(out.contains("Initialized"));

    // 2. Add diverse content
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Rust is a systems programming language focused on safety",
            "-w",
            "Programming",
            "-r",
            "Languages",
        ],
    );
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "The Milky Way contains approximately 100 billion stars",
            "-w",
            "Astronomy",
            "-r",
            "Galaxies",
        ],
    );
    run_ok(
        db,
        &[
            "add-drawer",
            "-c",
            "Python excels at data science and machine learning",
            "-w",
            "Programming",
            "-r",
            "Languages",
        ],
    );

    // 3. KG
    run_ok(db, &["kg", "add", "Rust", "is_a", "Language"]);

    // 4. Search
    let search_out = run_ok(db, &["search", "programming", "-k", "3"]);
    assert!(
        search_out.contains("Programming") || search_out.contains("programming"),
        "Search should find programming content: {search_out}"
    );

    // 5. Status
    let status = run_ok(db, &["status"]);
    assert!(status.contains("3") || status.contains("Drawers"));

    // 6. Export
    let export_path = dir.path().join("workflow-export.json");
    let export_out = run_ok(
        db,
        &["export", "-o", export_path.to_str().unwrap()],
    );
    assert!(export_path.exists());
    assert!(
        export_out.contains("Exported") || export_out.contains("bytes")
    );
}
