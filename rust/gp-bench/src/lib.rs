//! `gp-bench` — Comprehensive benchmark suite for GraphPalace.
//!
//! Provides deterministic test-data generators, recall benchmarks, pathfinding
//! benchmarks, throughput measurements, and a comparison framework that
//! produces Markdown / JSON reports.

pub mod comparison;
pub mod generators;
pub mod pathfinding;
pub mod recall;
pub mod throughput;
