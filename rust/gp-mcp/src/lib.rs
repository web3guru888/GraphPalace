//! MCP server for GraphPalace.
//!
//! Exposes graph operations as MCP tools over JSON-RPC / stdio transport.
//!
//! This crate defines:
//! - **28 tool schemas** ([`tools`] module) covering palace navigation,
//!   operations, knowledge graph, stigmergy, and agent diary.
//! - **PALACE_PROTOCOL** ([`protocol`] module) — the system prompt preamble
//!   that teaches LLM clients how to use the memory palace.
//!
//! Actual MCP protocol handling (JSON-RPC transport, session management)
//! is planned for Phase 5.

pub mod protocol;
pub mod tools;

// Re-exports for convenience
pub use protocol::PALACE_PROTOCOL;
pub use tools::{
    // Palace Navigation
    PalaceStatus, ListWings, ListRooms, GetTaxonomy,
    Search, Navigate, FindTunnels, GraphStats,
    // Palace Operations
    AddDrawer, DeleteDrawer, AddWing, AddRoom, CheckDuplicate,
    // Knowledge Graph
    KgAdd, KgQuery, KgInvalidate, KgTimeline, KgTraverse, KgContradictions, KgStats,
    // Stigmergy
    PheromoneStatus, PheromoneDeposit, HotPaths, ColdSpots, DecayNow,
    // Agent Diary
    ListAgents, DiaryWrite, DiaryRead,
    // Catalog
    ToolDefinition, tool_catalog,
};
