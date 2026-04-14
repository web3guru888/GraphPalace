//! Comprehensive GraphPalace evaluation with real ONNX embeddings.
//!
//! Tests: semantic search quality, scale, pathfinding, pheromones, KG.

use std::path::Path;
use std::time::Instant;

use gp_core::config::GraphPalaceConfig;
use gp_core::types::*;
use gp_embeddings::engine::{auto_engine, EmbeddingEngine};
use gp_palace::palace::GraphPalace;
use gp_storage::memory::InMemoryBackend;

fn make_palace(model_dir: Option<&Path>) -> GraphPalace {
    let config = GraphPalaceConfig::default();
    let storage = InMemoryBackend::new();
    let embeddings: Box<dyn EmbeddingEngine> = auto_engine(model_dir);
    GraphPalace::new(config, storage, embeddings).expect("palace creation")
}

// ═══════════════════════════════════════════════════════════════════════════
// Test corpus — diverse real-world content across multiple domains
// ═══════════════════════════════════════════════════════════════════════════

const CORPUS: &[(&str, &str, &str)] = &[
    // --- Software Engineering ---
    ("engineering", "rust", "Rust is a systems programming language that guarantees memory safety without a garbage collector through its ownership and borrowing system"),
    ("engineering", "rust", "The Rust borrow checker enforces strict rules at compile time: each value has exactly one owner, references must not outlive the data they point to"),
    ("engineering", "rust", "Cargo is the Rust package manager and build system, handling dependency resolution, compilation, testing, and documentation generation"),
    ("engineering", "python", "Python is a high-level interpreted language known for its readability and extensive standard library, widely used in data science and web development"),
    ("engineering", "python", "Python's Global Interpreter Lock prevents true parallel execution of threads, making multiprocessing the preferred approach for CPU-bound tasks"),
    ("engineering", "databases", "PostgreSQL is an advanced open-source relational database with support for JSONB, full-text search, and custom extensions"),
    ("engineering", "databases", "Redis is an in-memory key-value data store used as a cache, message broker, and session store with sub-millisecond response times"),
    ("engineering", "databases", "Graph databases like Neo4j store data as nodes and relationships, enabling efficient traversal of connected data without expensive joins"),
    ("engineering", "web", "REST APIs use HTTP methods GET, POST, PUT, DELETE to perform CRUD operations on resources identified by URIs"),
    ("engineering", "web", "WebSockets provide full-duplex communication channels over a single TCP connection, enabling real-time bidirectional data transfer"),

    // --- Science ---
    ("science", "physics", "The Standard Model of particle physics describes three of the four fundamental forces: electromagnetic, weak nuclear, and strong nuclear"),
    ("science", "physics", "General relativity describes gravity as the curvature of spacetime caused by mass and energy, predicting phenomena like gravitational lensing and black holes"),
    ("science", "physics", "Quantum entanglement is a phenomenon where two particles become correlated such that measuring one instantly determines the state of the other regardless of distance"),
    ("science", "biology", "CRISPR-Cas9 is a gene editing tool that allows precise modifications to DNA sequences by using a guide RNA to direct the Cas9 protein to specific genomic locations"),
    ("science", "biology", "Mitochondria are organelles that produce ATP through oxidative phosphorylation, often called the powerhouses of the cell"),
    ("science", "biology", "The human gut microbiome contains trillions of bacteria that influence digestion, immune function, and even mental health through the gut-brain axis"),
    ("science", "climate", "The greenhouse effect occurs when gases like CO2 and methane trap heat in the atmosphere, raising global temperatures and causing climate change"),
    ("science", "climate", "Ocean acidification from absorbed CO2 threatens marine ecosystems by dissolving calcium carbonate shells of coral, mollusks, and plankton"),

    // --- History ---
    ("history", "ancient", "The Roman Empire at its peak controlled territory spanning from Britain to Mesopotamia, with a population of approximately 70 million people"),
    ("history", "ancient", "The Library of Alexandria was one of the largest ancient libraries, containing an estimated 400,000 scrolls before its destruction"),
    ("history", "modern", "The Industrial Revolution began in Britain in the late 18th century with the mechanization of textile production and the development of steam power"),
    ("history", "modern", "The Apollo 11 mission in July 1969 achieved the first crewed Moon landing, with Neil Armstrong and Buzz Aldrin walking on the lunar surface"),
    ("history", "modern", "The fall of the Berlin Wall on November 9, 1989 marked the symbolic end of the Cold War and led to German reunification"),

    // --- AI/ML ---
    ("ai", "transformers", "Transformer architectures use self-attention mechanisms to process sequences in parallel, replacing recurrent approaches for most NLP tasks"),
    ("ai", "transformers", "Large language models like GPT and Claude are trained on massive text corpora using next-token prediction, developing emergent capabilities at scale"),
    ("ai", "reinforcement", "Reinforcement learning trains agents through trial and error, using reward signals to learn optimal policies for sequential decision-making"),
    ("ai", "reinforcement", "AlphaGo used Monte Carlo tree search combined with deep neural networks to defeat the world champion in the game of Go"),
    ("ai", "embeddings", "Word embeddings like Word2Vec and GloVe map words to dense vectors where semantic similarity corresponds to geometric proximity in the vector space"),
    ("ai", "embeddings", "Sentence transformers extend word embeddings to entire sentences, enabling semantic search, clustering, and similarity comparison at the sentence level"),
    ("ai", "agents", "Active Inference is a framework from neuroscience where agents minimize Expected Free Energy by reducing uncertainty about the world"),

    // --- GraphPalace-specific ---
    ("graphpalace", "architecture", "GraphPalace organizes memories in a spatial hierarchy: Wings contain Rooms, Rooms contain Closets, and Closets contain Drawers with verbatim content"),
    ("graphpalace", "stigmergy", "Stigmergy is indirect coordination through environmental modification — in GraphPalace, pheromone trails left by searches guide future navigation"),
    ("graphpalace", "pathfinding", "Semantic A* pathfinding in GraphPalace combines three cost components: embedding similarity, pheromone guidance, and structural relation weights"),
    ("graphpalace", "pheromones", "Five pheromone types in GraphPalace: exploitation (node value), exploration (already searched), success (good outcomes), traversal (frequency), recency (freshness)"),
];

// ═══════════════════════════════════════════════════════════════════════════
// Semantic Search Quality Tests
// ═══════════════════════════════════════════════════════════════════════════

struct SearchTest {
    query: &'static str,
    expected_top: &'static str, // substring that MUST appear in #1 result
    expected_domain: &'static str, // wing that should dominate results
    description: &'static str,
}

const SEARCH_TESTS: &[SearchTest] = &[
    // Exact concept recall
    SearchTest { query: "Rust ownership and borrowing memory safety", expected_top: "ownership", expected_domain: "engineering", description: "Exact concept: Rust ownership" },
    SearchTest { query: "how does CRISPR gene editing work", expected_top: "CRISPR", expected_domain: "science", description: "Exact concept: CRISPR" },
    SearchTest { query: "what is quantum entanglement", expected_top: "entanglement", expected_domain: "science", description: "Exact concept: entanglement" },
    SearchTest { query: "transformer self-attention mechanism", expected_top: "self-attention", expected_domain: "ai", description: "Exact concept: transformers" },

    // Paraphrase / synonym queries (no keywords in common)
    SearchTest { query: "energy factories inside living cells", expected_top: "mitochondria", expected_domain: "science", description: "Paraphrase: mitochondria" },
    SearchTest { query: "landing humans on the moon for the first time", expected_top: "Apollo", expected_domain: "history", description: "Paraphrase: Moon landing" },
    SearchTest { query: "preventing data races at compile time", expected_top: "borrow checker", expected_domain: "engineering", description: "Paraphrase: borrow checker" },
    SearchTest { query: "AI that learns by playing games against itself", expected_top: "Reinforcement", expected_domain: "ai", description: "Paraphrase: RL/AlphaGo" },

    // Cross-domain / abstract queries
    SearchTest { query: "insects leaving chemical trails to coordinate", expected_top: "Stigmergy", expected_domain: "graphpalace", description: "Abstract: stigmergy" },
    SearchTest { query: "how heat gets trapped in Earth's atmosphere", expected_top: "greenhouse", expected_domain: "science", description: "Abstract: greenhouse effect" },
    SearchTest { query: "ancient repository of human knowledge destroyed", expected_top: "Library of Alexandria", expected_domain: "history", description: "Abstract: Library of Alexandria" },
    SearchTest { query: "converting text into mathematical vectors for comparison", expected_top: "embedding", expected_domain: "ai", description: "Abstract: embeddings" },

    // Negative / adversarial queries (should still return something reasonable)
    SearchTest { query: "fast in-memory caching for web applications", expected_top: "Redis", expected_domain: "engineering", description: "Specific: Redis caching" },
    SearchTest { query: "bacteria living in the human digestive system", expected_top: "microbiome", expected_domain: "science", description: "Specific: gut microbiome" },
    SearchTest { query: "symbolic end of Cold War in Germany", expected_top: "Berlin Wall", expected_domain: "history", description: "Specific: Berlin Wall" },
];

fn run_search_quality(palace: &mut GraphPalace) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         SEMANTIC SEARCH QUALITY EVALUATION                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut pass = 0;
    let mut fail = 0;
    let mut domain_pass = 0;
    let mut total_top1_score = 0.0f32;
    let mut total_mrr = 0.0f64;

    println!("| # | Query | Top-1 Match? | Domain? | Score | MRR | Description |");
    println!("|---|-------|-------------|---------|-------|-----|-------------|");

    for (i, test) in SEARCH_TESTS.iter().enumerate() {
        let results = palace.search(test.query, 10).unwrap_or_default();

        let top1_match = results.first()
            .is_some_and(|r| r.content.contains(test.expected_top));
        let top1_score = results.first().map(|r| r.score).unwrap_or(0.0);
        let domain_match = results.first()
            .is_some_and(|r| r.wing_name == test.expected_domain);

        // Compute MRR: reciprocal rank of first correct result
        let rr = results.iter().enumerate()
            .find(|(_, r)| r.content.contains(test.expected_top))
            .map(|(rank, _)| 1.0 / (rank as f64 + 1.0))
            .unwrap_or(0.0);

        if top1_match { pass += 1; } else { fail += 1; }
        if domain_match { domain_pass += 1; }
        total_top1_score += top1_score;
        total_mrr += rr;

        let check = if top1_match { "YES" } else { "no" };
        let dcheck = if domain_match { "YES" } else { "no" };
        println!("| {} | {}... | {} | {} | {:.3} | {:.2} | {} |",
            i + 1,
            &test.query[..test.query.len().min(35)],
            check, dcheck, top1_score, rr,
            test.description);
    }

    let total = SEARCH_TESTS.len();
    let precision = pass as f64 / total as f64;
    let domain_acc = domain_pass as f64 / total as f64;
    let avg_score = total_top1_score / total as f32;
    let mrr = total_mrr / total as f64;

    println!("\n### Search Quality Summary");
    println!("  Top-1 Precision:     {pass}/{total} ({:.1}%)", precision * 100.0);
    println!("  Domain Accuracy:     {domain_pass}/{total} ({:.1}%)", domain_acc * 100.0);
    println!("  Average Top-1 Score: {avg_score:.4}");
    println!("  Mean Reciprocal Rank (MRR): {mrr:.4}");
    println!("  Failures: {fail}");

    if fail > 0 {
        println!("\n  Failed queries:");
        for (i, test) in SEARCH_TESTS.iter().enumerate() {
            let results = palace.search(test.query, 3).unwrap_or_default();
            let top1_match = results.first()
                .is_some_and(|r| r.content.contains(test.expected_top));
            if !top1_match {
                println!("    [{i}] '{}' — expected '{}' in top-1, got: '{}'",
                    test.query, test.expected_top,
                    results.first().map(|r| &r.content[..r.content.len().min(60)]).unwrap_or("(empty)"));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scale & Throughput Benchmarks
// ═══════════════════════════════════════════════════════════════════════════

fn run_throughput_benchmarks(palace: &mut GraphPalace) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         THROUGHPUT & LATENCY BENCHMARKS                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let status = palace.status().unwrap();
    println!("Palace size: {} drawers, {} wings, {} rooms\n", status.drawer_count, status.wing_count, status.room_count);

    // Search latency (warm)
    let queries = [
        "memory safety in systems programming",
        "machine learning neural networks",
        "ancient history empires",
        "climate change greenhouse gases",
        "graph database pathfinding",
    ];

    println!("### Search Latency (k=10, {} drawers)", status.drawer_count);
    let mut total_us = 0u128;
    for q in &queries {
        let t0 = Instant::now();
        let _ = palace.search(q, 10);
        let elapsed = t0.elapsed();
        total_us += elapsed.as_micros();
        println!("  '{}...' → {:.1}ms", &q[..q.len().min(40)], elapsed.as_secs_f64() * 1000.0);
    }
    println!("  Average: {:.1}ms\n", total_us as f64 / queries.len() as f64 / 1000.0);

    // Pathfinding latency
    println!("### Pathfinding Latency");
    let wings = palace.storage().list_wings();
    if wings.len() >= 2 {
        let rooms_a = palace.storage().list_rooms(&wings[0].id);
        let rooms_b = palace.storage().list_rooms(&wings[1].id);
        if !rooms_a.is_empty() && !rooms_b.is_empty() {
            // Same-wing
            if rooms_a.len() >= 2 {
                let t0 = Instant::now();
                let result = palace.navigate(&rooms_a[0].id, &rooms_a[rooms_a.len()-1].id, None);
                let elapsed = t0.elapsed();
                match result {
                    Ok(p) => println!("  Same-wing: {:.1}ms (path={} nodes, cost={:.3})",
                        elapsed.as_secs_f64() * 1000.0, p.path.len(), p.total_cost),
                    Err(e) => println!("  Same-wing: {:.1}ms (no path: {e})", elapsed.as_secs_f64() * 1000.0),
                }
            }
            // Cross-wing
            let t0 = Instant::now();
            let result = palace.navigate(&rooms_a[0].id, &rooms_b[0].id, None);
            let elapsed = t0.elapsed();
            match result {
                Ok(p) => println!("  Cross-wing: {:.1}ms (path={} nodes, cost={:.3})",
                    elapsed.as_secs_f64() * 1000.0, p.path.len(), p.total_cost),
                Err(e) => println!("  Cross-wing: {:.1}ms (no path: {e})", elapsed.as_secs_f64() * 1000.0),
            }
        }
    }

    // Pheromone decay latency
    let t0 = Instant::now();
    palace.decay_pheromones().unwrap();
    let elapsed = t0.elapsed();
    println!("  Pheromone decay: {:.1}ms\n", elapsed.as_secs_f64() * 1000.0);

    // Export size
    let export = palace.export().unwrap();
    let json = serde_json::to_string(&export).unwrap();
    println!("### Storage");
    println!("  Export JSON size: {:.1} KB", json.len() as f64 / 1024.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Pheromone Dynamics Test
// ═══════════════════════════════════════════════════════════════════════════

fn run_pheromone_test(palace: &mut GraphPalace) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         PHEROMONE DYNAMICS TEST                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // 1. Search for something and deposit pheromones on the path
    let results = palace.search("pheromone navigation stigmergy", 3).unwrap();
    println!("Search for 'pheromone navigation stigmergy':");
    for (i, r) in results.iter().enumerate() {
        println!("  {}: [score={:.4}] {} (id={})", i+1, r.score, &r.content[..r.content.len().min(70)], r.drawer_id);
    }

    // Deposit success pheromones on top results
    if results.len() >= 2 {
        let path: Vec<String> = results.iter().take(3).map(|r| r.drawer_id.clone()).collect();
        palace.deposit_pheromones(&path, 1.0).unwrap();
        println!("\n  Deposited pheromones on path: {}", path.join(" → "));
    }

    // 2. Check hot paths
    let hot = palace.hot_paths(5).unwrap();
    println!("\n### Hot Paths (after deposit):");
    if hot.is_empty() {
        println!("  (none yet)");
    }
    for p in &hot {
        println!("  {} → {} (success={:.4})", p.from_id, p.to_id, p.success_pheromone);
    }

    // 3. Check cold spots
    let cold = palace.cold_spots(5).unwrap();
    println!("\n### Cold Spots:");
    for s in &cold {
        println!("  {} ({}) — total={:.4}", s.node_id, s.name, s.total_pheromone);
    }

    // 4. Search again — pheromone boost should change ranking
    let results2 = palace.search("pheromone navigation stigmergy", 3).unwrap();
    println!("\n### Re-search after pheromone deposit:");
    for (i, r) in results2.iter().enumerate() {
        println!("  {}: [score={:.4}] {} (id={})", i+1, r.score, &r.content[..r.content.len().min(70)], r.drawer_id);
    }

    // 5. Decay and observe
    println!("\n### After 5 decay cycles:");
    for _ in 0..5 {
        palace.decay_pheromones().unwrap();
    }
    let hot2 = palace.hot_paths(3).unwrap();
    for p in &hot2 {
        println!("  {} → {} (success={:.4})", p.from_id, p.to_id, p.success_pheromone);
    }

    let mass = palace.status().unwrap().total_pheromone_mass;
    println!("  Total pheromone mass: {:.4}", mass);
}

// ═══════════════════════════════════════════════════════════════════════════
// Knowledge Graph Test
// ═══════════════════════════════════════════════════════════════════════════

fn run_kg_test(palace: &mut GraphPalace) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         KNOWLEDGE GRAPH TEST                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Build a knowledge graph
    let triples = [
        ("Rust", "is_a", "Programming Language"),
        ("Python", "is_a", "Programming Language"),
        ("PostgreSQL", "is_a", "Database"),
        ("Redis", "is_a", "Database"),
        ("Neo4j", "is_a", "Graph Database"),
        ("GraphPalace", "is_a", "Graph Database"),
        ("GraphPalace", "written_in", "Rust"),
        ("GraphPalace", "uses", "Pheromone Navigation"),
        ("GraphPalace", "uses", "Active Inference"),
        ("GraphPalace", "uses", "Semantic A*"),
        ("CRISPR", "is_a", "Gene Editing Tool"),
        ("Transformer", "is_a", "Neural Architecture"),
        ("GPT", "based_on", "Transformer"),
        ("Claude", "based_on", "Transformer"),
        ("AlphaGo", "uses", "Reinforcement Learning"),
        ("AlphaGo", "uses", "Monte Carlo Tree Search"),
        ("Apollo 11", "achieved", "Moon Landing"),
        ("Berlin Wall", "fell_in", "1989"),
    ];

    println!("### Adding {} triples...", triples.len());
    let t0 = Instant::now();
    for (s, p, o) in &triples {
        palace.kg_add(s, p, o).unwrap();
    }
    let elapsed = t0.elapsed();
    println!("  Added in {:.1}ms\n", elapsed.as_secs_f64() * 1000.0);

    // Query entities
    let queries = ["GraphPalace", "Transformer", "Rust", "AlphaGo"];
    for entity in &queries {
        let rels = palace.kg_query(entity).unwrap();
        println!("### {} ({} relationships)", entity, rels.len());
        for r in &rels {
            println!("  {} --{}--> {} (conf={:.2})", r.subject, r.predicate, r.object, r.confidence);
        }
        println!();
    }

    let status = palace.status().unwrap();
    println!("### KG Statistics");
    println!("  Entities:      {}", status.entity_count);
    println!("  Relationships: {}", status.relationship_count);
}

// ═══════════════════════════════════════════════════════════════════════════
// Similarity Graph Test
// ═══════════════════════════════════════════════════════════════════════════

fn run_similarity_test(palace: &GraphPalace) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         SIMILARITY GRAPH TEST                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let edge_count = palace.build_similarity_graph(0.5).unwrap();
    let elapsed = t0.elapsed();
    println!("Built similarity graph (threshold=0.5): {} edges in {:.1}ms\n",
        edge_count, elapsed.as_secs_f64() * 1000.0);

    // Find similar drawers for a few samples
    let d = palace.storage().read_data();
    let sample_drawers: Vec<_> = d.drawers.values().take(5).collect();
    for drawer in &sample_drawers {
        let similar = palace.find_similar(&drawer.id, 3).unwrap();
        println!("### Similar to: '{}'", &drawer.content[..drawer.content.len().min(60)]);
        if similar.is_empty() {
            println!("  (no similar drawers above threshold)");
        }
        for (other_id, score) in &similar {
            if let Some(other) = d.drawers.get(other_id.as_str()) {
                println!("  [{:.3}] {}", score, &other.content[..other.content.len().min(70)]);
            }
        }
        println!();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Insert Throughput with ONNX
// ═══════════════════════════════════════════════════════════════════════════

fn run_insert_benchmark(palace: &mut GraphPalace) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         INSERT THROUGHPUT (ONNX Embeddings)                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Generate extra content to stress test
    let extra_content = [
        "Machine learning models require careful hyperparameter tuning for optimal performance",
        "Kubernetes orchestrates containerized applications across clusters of machines",
        "TCP implements reliable ordered delivery of data streams between applications",
        "The Fourier transform decomposes signals into their constituent frequency components",
        "Hash tables provide average O(1) lookup time using a hash function to map keys to indices",
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules",
        "OAuth 2.0 is an authorization framework that enables limited access to user accounts",
        "The traveling salesman problem asks for the shortest route visiting all cities exactly once",
        "Blockchain creates an immutable distributed ledger using cryptographic hash chains",
        "DNA replication is semiconservative: each new double helix has one old and one new strand",
        "Functional programming treats computation as evaluation of mathematical functions avoiding mutable state",
        "The Doppler effect is the change in frequency of a wave relative to an observer moving toward or away from the source",
        "Gradient descent iteratively adjusts parameters to minimize a loss function by following the negative gradient",
        "The Turing test proposes that a machine is intelligent if a human cannot distinguish its responses from those of another human",
        "Enzymes are biological catalysts that lower activation energy and speed up chemical reactions",
        "MapReduce is a programming model for processing large data sets with a parallel distributed algorithm",
        "The double slit experiment demonstrates wave-particle duality of photons and electrons",
        "Microservices architecture decomposes applications into small independent services communicating via APIs",
        "Natural selection drives evolution by favoring organisms with traits that improve survival and reproduction",
        "Elliptic curve cryptography provides equivalent security to RSA with much shorter key lengths",
    ];

    let t0 = Instant::now();
    for (i, content) in extra_content.iter().enumerate() {
        let wing = if i < 7 { "engineering" } else if i < 14 { "science" } else { "ai" };
        let room = match i % 4 {
            0 => "concepts",
            1 => "algorithms",
            2 => "systems",
            _ => "theory",
        };
        palace.add_drawer(content, wing, room, DrawerSource::Api).unwrap();
    }
    let elapsed = t0.elapsed();
    let ops_per_sec = extra_content.len() as f64 / elapsed.as_secs_f64();

    println!("Inserted {} drawers with ONNX embedding in {:.1}ms",
        extra_content.len(), elapsed.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.1} inserts/sec", ops_per_sec);
    println!("  Per-insert: {:.1}ms (includes ONNX inference)\n", elapsed.as_secs_f64() * 1000.0 / extra_content.len() as f64);

    let status = palace.status().unwrap();
    println!("Palace now has {} drawers total", status.drawer_count);
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let model_path = std::env::args().nth(1);
    let model_dir = model_path.as_deref().map(Path::new);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    GraphPalace Comprehensive Evaluation Suite                ║");
    println!("║    Engine: {}                              ║",
        if model_dir.is_some_and(|d| d.join("model.onnx").exists()) { "ONNX (all-MiniLM-L6-v2)" } else { "TF-IDF / Mock fallback   " });
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut palace = make_palace(model_dir);

    // Populate with corpus
    println!("Loading {} documents into palace...", CORPUS.len());
    let t0 = Instant::now();
    for (wing, room, content) in CORPUS {
        palace.add_drawer(content, wing, room, DrawerSource::Conversation).unwrap();
    }
    let elapsed = t0.elapsed();
    println!("  Loaded in {:.1}ms ({:.1} docs/sec)\n",
        elapsed.as_secs_f64() * 1000.0,
        CORPUS.len() as f64 / elapsed.as_secs_f64());

    // Run all tests
    run_search_quality(&mut palace);
    run_throughput_benchmarks(&mut palace);
    run_pheromone_test(&mut palace);
    run_kg_test(&mut palace);
    run_similarity_test(&palace);
    run_insert_benchmark(&mut palace);

    // Final search after scaling up
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         POST-SCALE SEARCH QUALITY CHECK                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let status = palace.status().unwrap();
    println!("Palace: {} drawers, {} entities, {} relationships\n",
        status.drawer_count, status.entity_count, status.relationship_count);

    let spot_checks = [
        ("Rust borrow checker ownership", "borrow checker"),
        ("graph databases relationships traversal", "Graph databases"),
        ("moon landing astronauts", "Apollo"),
        ("gradient descent optimization", "Gradient descent"),
        ("blockchain distributed ledger", "Blockchain"),
    ];

    let mut pass = 0;
    for (query, expected) in &spot_checks {
        let results = palace.search(query, 1).unwrap();
        let hit = results.first().is_some_and(|r| r.content.contains(expected));
        if hit { pass += 1; }
        let mark = if hit { "PASS" } else { "FAIL" };
        let score = results.first().map(|r| r.score).unwrap_or(0.0);
        let detail = if hit { String::new() } else { format!("(expected '{expected}')") };
        println!("  [{}] '{}' → score={:.3} {}", mark, query, score, detail);
    }
    println!("\n  Post-scale precision: {pass}/{} ({:.0}%)", spot_checks.len(), pass as f64 / spot_checks.len() as f64 * 100.0);

    println!("\n═══ EVALUATION COMPLETE ═══");
}
