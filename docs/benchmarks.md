# Benchmarks

The `gp-bench` crate provides a comprehensive benchmark suite for GraphPalace, measuring recall accuracy, pathfinding performance, and throughput. It includes deterministic test data generators and structured result reporting.

## Benchmark Categories

| Category | Target | Comparison |
|----------|--------|-----------|
| **Recall** | ≥96.6% | MemPalace's LongMemEval benchmark |
| **Pathfinding** | ≥90.9% success | STAN_X v8's Semantic A* |
| **Throughput** | See §Performance Targets | Raw operations/sec |

## Recall Benchmarks

The recall benchmark measures how accurately GraphPalace retrieves previously stored memories — the core promise inherited from MemPalace's verbatim storage philosophy.

### Methodology

1. **Generate** a test palace with N drawers containing diverse, realistic content
2. **Query** with paraphrased versions of stored content (never the exact text)
3. **Score**: a query is a "hit" if the correct drawer appears in the top-k results
4. **Recall** = hits / total_queries

```rust
use gp_bench::{RecallBenchmark, TestPalaceGenerator};

let generator = TestPalaceGenerator::new(42); // deterministic seed
let palace = generator.generate(1000);        // 1,000 drawers

let bench = RecallBenchmark::new()
    .k(10)                    // top-10 retrieval
    .num_queries(500)         // 500 paraphrased queries
    .similarity_threshold(0.0); // no minimum score cutoff

let result = bench.run(&palace)?;
println!("Recall@10: {:.1}%", result.recall * 100.0);
println!("Mean Reciprocal Rank: {:.3}", result.mrr);
println!("Avg latency: {:.1}ms", result.avg_latency_ms);
```

### Scales

| Scale | Drawers | Wings | Rooms | Purpose |
|-------|---------|-------|-------|---------|
| **Tiny** | 100 | 2 | 5 | Quick sanity check |
| **Small** | 1,000 | 5 | 20 | Default benchmark |
| **Medium** | 10,000 | 10 | 50 | Realistic workload |
| **Large** | 100,000 | 20 | 100 | Stress test |

### Comparison Approach

To compare against MemPalace's 96.6% LongMemEval result:

1. Use the same query types: **single-session** (retrieve from one conversation), **multi-session** (retrieve across conversations), **knowledge-graph** (entity relationship queries), **temporal** (time-bounded queries)
2. Paraphrase queries using synonym substitution and sentence restructuring
3. Report recall@k for k ∈ {1, 3, 5, 10}
4. Measure with and without pheromone boosting to isolate stigmergy's contribution

### Result Structure

```rust
pub struct RecallResult {
    pub recall: f64,           // Recall@k
    pub mrr: f64,              // Mean Reciprocal Rank
    pub precision: f64,        // Precision@k
    pub avg_latency_ms: f64,   // Average query latency
    pub p50_latency_ms: f64,   // Median latency
    pub p99_latency_ms: f64,   // 99th percentile latency
    pub total_queries: usize,
    pub hits: usize,
    pub misses: usize,
    pub scale: BenchScale,
    pub pheromone_boosted: bool,
}
```

## Pathfinding Benchmarks

The pathfinding benchmark measures Semantic A*'s ability to find routes through the palace graph, targeting STAN_X v8's 90.9% success rate.

### Methodology

1. **Generate** a palace with cross-wing tunnels and inter-room halls
2. **Select** random start/goal pairs at varying graph distances
3. **Run** Semantic A* with the default 40/30/30 cost weights
4. **Measure** success rate, path cost, latency, and nodes expanded

```rust
use gp_bench::PathfindingBenchmark;

let bench = PathfindingBenchmark::new()
    .num_pairs(200)           // 200 start/goal pairs
    .max_iterations(10_000)   // per-path iteration budget
    .include_cross_wing(true); // include cross-wing (harder) pairs

let result = bench.run(&palace)?;
println!("Success rate: {:.1}%", result.success_rate * 100.0);
println!("Avg path cost: {:.3}", result.avg_cost);
println!("Avg latency: {:.1}ms", result.avg_latency_ms);
println!("Avg nodes expanded: {:.0}", result.avg_nodes_expanded);
```

### With vs Without Pheromones

A key question: does stigmergy actually help pathfinding?

```rust
// Run twice: pristine palace (no pheromones) vs primed palace (after swarm exploration)
let cold_result = bench.run(&cold_palace)?;

// Run a swarm for 100 cycles to build up pheromone trails
swarm.run(&mut warm_palace, 100)?;
let warm_result = bench.run(&warm_palace)?;

println!("Cold success: {:.1}%, Warm success: {:.1}%",
    cold_result.success_rate * 100.0,
    warm_result.success_rate * 100.0);
println!("Cold latency: {:.1}ms, Warm latency: {:.1}ms",
    cold_result.avg_latency_ms,
    warm_result.avg_latency_ms);
```

### Result Structure

```rust
pub struct PathfindingResult {
    pub success_rate: f64,          // Fraction of pairs where path was found
    pub avg_cost: f64,              // Average path cost (successful only)
    pub avg_latency_ms: f64,        // Average time per search
    pub p50_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub avg_nodes_expanded: f64,    // Average nodes expanded per search
    pub avg_path_length: f64,       // Average edges in found paths
    pub total_pairs: usize,
    pub successes: usize,
    pub failures: usize,
    pub pheromone_state: String,    // "cold" or "warm"
}
```

## Throughput Benchmarks

Raw operation throughput for capacity planning:

```rust
use gp_bench::ThroughputBenchmark;

let bench = ThroughputBenchmark::new()
    .warmup_iterations(100)
    .measurement_iterations(1000);

let result = bench.run(&mut palace)?;
println!("Insert:  {:.0} drawers/sec", result.insert_rate);
println!("Search:  {:.0} queries/sec", result.search_rate);
println!("Decay:   {:.0} nodes/sec", result.decay_rate);
println!("Export:  {:.1}ms for {} nodes", result.export_ms, result.node_count);
println!("Import:  {:.1}ms for {} nodes", result.import_ms, result.node_count);
```

### Operations Measured

| Operation | What | Target |
|-----------|------|--------|
| **Insert** | Create drawer + compute embedding + index | >1,000/sec (in-memory) |
| **Search** | Semantic search top-10 | <50ms (HNSW), <200ms (brute-force) |
| **Navigate** | A* pathfinding | <200ms (cached), <500ms (uncached) |
| **Decay** | One full pheromone decay cycle | <500ms for 10K edges |
| **Export** | Serialize entire palace to JSON | <5s for 100K drawers |
| **Import** | Deserialize and load from JSON | <10s for 100K drawers |

## Test Data Generators

The `TestPalaceGenerator` creates deterministic, reproducible test palaces:

```rust
pub struct TestPalaceGenerator {
    seed: u64,
    domain_mix: Vec<(&'static str, f64)>,  // domain name, proportion
}

impl TestPalaceGenerator {
    /// Create a generator with a fixed seed for reproducibility
    pub fn new(seed: u64) -> Self;

    /// Generate a palace with the given number of drawers
    pub fn generate(&self, num_drawers: usize) -> GraphPalace;

    /// Generate with specific structure
    pub fn generate_structured(
        &self,
        num_wings: usize,
        rooms_per_wing: usize,
        closets_per_room: usize,
        drawers_per_closet: usize,
    ) -> GraphPalace;
}
```

Default domain mix:

| Domain | Proportion | Content Style |
|--------|-----------|---------------|
| Climate Science | 20% | Temperature, CO₂, sea level data |
| Economics | 20% | GDP, inflation, trade statistics |
| Astrophysics | 15% | Stars, galaxies, cosmic measurements |
| Epidemiology | 15% | Disease prevalence, mortality rates |
| Materials Science | 15% | Material properties, synthesis methods |
| General Knowledge | 15% | Wikipedia-style factual statements |

Each drawer gets:
- Unique, realistic content (~50-200 words)
- Deterministic mock embedding (seeded)
- Assigned wing/room/closet based on domain
- Random importance score (0.3–1.0)
- Source tag from ["api", "conversation", "file", "agent"]

## How to Run Benchmarks

### Cargo Bench

```bash
# Run all benchmarks
cd rust && cargo bench --package gp-bench

# Run specific benchmark
cargo bench --package gp-bench -- recall
cargo bench --package gp-bench -- pathfinding
cargo bench --package gp-bench -- throughput

# With specific scale
BENCH_SCALE=medium cargo bench --package gp-bench -- recall

# Generate HTML report (via criterion)
cargo bench --package gp-bench -- --output-format=criterion
# Report at: target/criterion/report/index.html
```

### Programmatic

```rust
use gp_bench::{run_all_benchmarks, BenchConfig};

let config = BenchConfig {
    scale: BenchScale::Small,
    recall_k: 10,
    pathfinding_pairs: 200,
    throughput_iterations: 1000,
    output_json: true,
};

let report = run_all_benchmarks(config)?;
println!("{}", serde_json::to_string_pretty(&report)?);
```

## Report Format

Benchmark results are output as structured JSON for integration with CI/CD:

```json
{
  "timestamp": "2026-04-13T10:30:00Z",
  "graphpalace_version": "0.1.0",
  "scale": "small",
  "recall": {
    "recall_at_10": 0.968,
    "mrr": 0.891,
    "avg_latency_ms": 12.3,
    "p99_latency_ms": 45.1,
    "total_queries": 500,
    "pheromone_boosted": true
  },
  "pathfinding": {
    "success_rate": 0.925,
    "avg_cost": 2.341,
    "avg_latency_ms": 34.7,
    "pheromone_state": "warm"
  },
  "throughput": {
    "insert_per_sec": 4521.0,
    "search_per_sec": 892.0,
    "decay_ms": 12.4,
    "export_ms": 234.5,
    "import_ms": 456.7
  },
  "comparison": {
    "mempalace_recall": 0.966,
    "graphpalace_recall": 0.968,
    "recall_delta": "+0.2%",
    "stanx_pathfinding": 0.909,
    "graphpalace_pathfinding": 0.925,
    "pathfinding_delta": "+1.6%"
  }
}
```

### Interpreting Results

- **Recall@10 ≥ 96.6%**: Matches or exceeds MemPalace's LongMemEval score. The verbatim storage philosophy (never summarize) plus embedding search should achieve this by design.
- **Pathfinding success ≥ 90.9%**: Matches STAN_X v8. Pheromone-boosted ("warm") runs should exceed cold runs by 5-15%.
- **Latency under targets**: Search <50ms, pathfinding <500ms. If exceeded, check palace size vs benchmark scale.
- **Throughput**: In-memory backend should exceed targets easily. Kuzu backend numbers reflect real I/O.

## CI Integration

The benchmark suite can be run in CI to catch performance regressions:

```yaml
# .github/workflows/graphpalace-ci.yml
- name: Run benchmarks
  run: |
    cd rust
    BENCH_SCALE=tiny cargo bench --package gp-bench -- --output-format=json > bench-results.json
    # Compare against baseline
    python scripts/check_regression.py bench-results.json baseline.json --threshold 5%
```
