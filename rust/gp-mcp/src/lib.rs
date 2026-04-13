//! MCP server for GraphPalace.
//!
//! Exposes graph operations as MCP tools over JSON-RPC / stdio transport.
//!
//! This crate defines:
//! - **28 tool schemas** ([`tools`] module) covering palace navigation,
//!   operations, knowledge graph, stigmergy, and agent diary.
//! - **PALACE_PROTOCOL** ([`protocol`] module) — the static system prompt
//!   preamble that teaches LLM clients how to use the memory palace.
//! - **Palace Protocol** ([`palace_protocol`] module) — dynamic prompt
//!   generation with live palace statistics injection.
//! - **MCP Server** ([`server`] module) — JSON-RPC 2.0 server with tool
//!   routing, capability negotiation, and prompt serving.

pub mod palace_protocol;
pub mod protocol;
pub mod server;
pub mod tools;

// Re-exports for convenience
pub use protocol::PALACE_PROTOCOL;
pub use palace_protocol::{PalaceStats, generate_palace_protocol, generate_palace_protocol_minimal};
pub use server::{McpServer, JsonRpcRequest, JsonRpcResponse, JsonRpcError, ToolCallResult, ToolHandler};
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
