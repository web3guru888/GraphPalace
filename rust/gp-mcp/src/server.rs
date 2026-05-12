//! MCP server implementation using JSON-RPC 2.0 protocol.
//!
//! Handles the standard MCP lifecycle:
//! - `initialize` — Server capability negotiation
//! - `tools/list` — Return all 28 tool definitions
//! - `tools/call` — Route tool calls to handlers
//!
//! The server is protocol-level only — actual tool implementations
//! require a graph backend (Kuzu or mock) to be injected.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::palace_protocol::{generate_palace_protocol, PalaceStats};
use crate::tools::{tool_catalog, ToolDefinition};

/// Trait for providing backend tool execution.
///
/// Implement this to connect a real palace backend (e.g., `GraphPalace`).
/// Return `Some(result)` to handle the tool call, or `None` to fall through
/// to the default placeholder handling.
pub trait ToolHandler {
    fn handle_tool(&mut self, name: &str, arguments: Option<&Value>) -> Option<ToolCallResult>;
}

/// MCP protocol version.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Server name reported during initialization.
pub const SERVER_NAME: &str = "graphpalace";

/// Server version.
pub const SERVER_VERSION: &str = "0.1.0";

// ─── JSON-RPC Types ───────────────────────────────────────────────────────

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(id: Option<Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Create a method-not-found error.
    pub fn method_not_found(id: Option<Value>, method: &str) -> Self {
        Self::error(id, -32601, format!("Method not found: {method}"))
    }

    /// Create an invalid-params error.
    pub fn invalid_params(id: Option<Value>, msg: impl Into<String>) -> Self {
        Self::error(id, -32602, msg)
    }

    /// Create an internal error.
    pub fn internal_error(id: Option<Value>, msg: impl Into<String>) -> Self {
        Self::error(id, -32603, msg)
    }
}

// ─── MCP Message Types ────────────────────────────────────────────────────

/// Server capabilities reported during initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub tools: ToolsCapability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
}

/// Tool capability declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// Whether the tool list can change at runtime.
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// Prompts capability declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// Initialize result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
    pub capabilities: ServerCapabilities,
}

/// Server identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// A tool as returned by tools/list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// tools/list result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsListResult {
    pub tools: Vec<McpTool>,
}

/// tools/call request params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
}

/// tools/call result content item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// tools/call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: Vec<ToolResultContent>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolCallResult {
    /// Create a successful text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent {
                content_type: "text".to_string(),
                text: text.into(),
            }],
            is_error: None,
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent {
                content_type: "text".to_string(),
                text: message.into(),
            }],
            is_error: Some(true),
        }
    }
}

// ─── MCP Server ───────────────────────────────────────────────────────────

/// The GraphPalace MCP server.
///
/// Processes JSON-RPC 2.0 messages and routes them to the appropriate
/// handlers. Tool implementations are dispatched through the `handle_tool_call`
/// method which can be overridden or extended.
pub struct McpServer {
    /// Current palace statistics (refreshed periodically).
    pub stats: PalaceStats,
    /// Tool catalog (28 tools).
    tools: Vec<ToolDefinition>,
    /// Whether the server has been initialized.
    initialized: bool,
    /// Optional backend handler for real tool execution.
    handler: Option<Box<dyn ToolHandler>>,
}

impl McpServer {
    /// Create a new MCP server.
    pub fn new() -> Self {
        Self {
            stats: PalaceStats::default(),
            tools: tool_catalog(),
            initialized: false,
            handler: None,
        }
    }

    /// Create a server with initial palace statistics.
    pub fn with_stats(stats: PalaceStats) -> Self {
        Self {
            stats,
            tools: tool_catalog(),
            initialized: false,
            handler: None,
        }
    }

    /// Create a server with a backend tool handler.
    pub fn with_handler(handler: Box<dyn ToolHandler>) -> Self {
        Self {
            stats: PalaceStats::default(),
            tools: tool_catalog(),
            initialized: false,
            handler: Some(handler),
        }
    }

    /// Set the backend tool handler.
    pub fn set_handler(&mut self, handler: Box<dyn ToolHandler>) {
        self.handler = Some(handler);
    }

    /// Whether the server has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Process a JSON-RPC request and return a response.
    pub fn handle_message(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "initialized" => {
                // Notification — no response needed, but return success
                JsonRpcResponse::success(request.id.clone(), Value::Null)
            }
            "tools/list" => self.handle_tools_list(request),
            "tools/call" => self.handle_tools_call(request),
            "prompts/list" => self.handle_prompts_list(request),
            "prompts/get" => self.handle_prompts_get(request),
            _ => JsonRpcResponse::method_not_found(request.id.clone(), &request.method),
        }
    }

    /// Process a raw JSON string and return a JSON response string.
    pub fn handle_json(&mut self, json: &str) -> String {
        match serde_json::from_str::<JsonRpcRequest>(json) {
            Ok(request) => {
                let response = self.handle_message(&request);
                serde_json::to_string(&response).unwrap_or_else(|e| {
                    format!(
                        r#"{{"jsonrpc":"2.0","error":{{"code":-32603,"message":"Serialization error: {e}"}}}}"#
                    )
                })
            }
            Err(e) => {
                let response =
                    JsonRpcResponse::error(None, -32700, format!("Parse error: {e}"));
                serde_json::to_string(&response).unwrap()
            }
        }
    }

    // ─── Method Handlers ──────────────────────────────────────────────

    fn handle_initialize(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        self.initialized = true;
        let result = InitializeResult {
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            server_info: ServerInfo {
                name: SERVER_NAME.to_string(),
                version: SERVER_VERSION.to_string(),
            },
            capabilities: ServerCapabilities {
                tools: ToolsCapability {
                    list_changed: false,
                },
                prompts: Some(PromptsCapability {
                    list_changed: false,
                }),
            },
        };
        JsonRpcResponse::success(request.id.clone(), serde_json::to_value(result).unwrap())
    }

    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tools: Vec<McpTool> = self
            .tools
            .iter()
            .map(|t| McpTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            })
            .collect();
        let result = ToolsListResult { tools };
        JsonRpcResponse::success(request.id.clone(), serde_json::to_value(result).unwrap())
    }

    fn handle_tools_call(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params = match &request.params {
            Some(p) => match serde_json::from_value::<ToolCallParams>(p.clone()) {
                Ok(params) => params,
                Err(e) => {
                    return JsonRpcResponse::invalid_params(
                        request.id.clone(),
                        format!("Invalid tools/call params: {e}"),
                    )
                }
            },
            None => {
                return JsonRpcResponse::invalid_params(
                    request.id.clone(),
                    "Missing params for tools/call",
                )
            }
        };

        let result = self.dispatch_tool(&params.name, params.arguments.as_ref());
        JsonRpcResponse::success(request.id.clone(), serde_json::to_value(result).unwrap())
    }

    fn handle_prompts_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let prompts = serde_json::json!({
            "prompts": [{
                "name": "palace_protocol",
                "description": "The PALACE_PROTOCOL system prompt that teaches LLMs how to use the memory palace"
            }]
        });
        JsonRpcResponse::success(request.id.clone(), prompts)
    }

    fn handle_prompts_get(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let prompt = generate_palace_protocol(&self.stats);
        let result = serde_json::json!({
            "messages": [{
                "role": "system",
                "content": { "type": "text", "text": prompt }
            }]
        });
        JsonRpcResponse::success(request.id.clone(), result)
    }

    // ─── Tool Dispatch ────────────────────────────────────────────────

    /// Dispatch a tool call to the appropriate handler.
    ///
    /// Currently returns placeholder responses since the actual graph
    /// backend is not yet connected. Each tool validates its parameters
    /// and returns a structured response.
    pub fn dispatch_tool(&mut self, name: &str, arguments: Option<&Value>) -> ToolCallResult {
        // Verify tool exists
        if !self.tools.iter().any(|t| t.name == name) {
            return ToolCallResult::error(format!("Unknown tool: {name}"));
        }

        // Try custom handler first
        if let Some(ref mut handler) = self.handler {
            if let Some(result) = handler.handle_tool(name, arguments) {
                return result;
            }
        }

        // Fall through to default placeholder handling
        match name {
            // Palace Navigation (read)
            "palace_status" => {
                let protocol = generate_palace_protocol(&self.stats);
                ToolCallResult::text(protocol)
            }
            "list_wings" => ToolCallResult::text(format!(
                "{{\"wings\": [], \"count\": {}}}",
                self.stats.wing_count
            )),
            "list_rooms" => {
                let wing_id = arguments
                    .and_then(|a| a.get("wing_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                ToolCallResult::text(format!(
                    "{{\"wing_id\": \"{wing_id}\", \"rooms\": []}}"
                ))
            }
            "get_taxonomy" => ToolCallResult::text("{\"taxonomy\": {}}"),
            "search" => {
                let query = arguments
                    .and_then(|a| a.get("query"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if query.is_empty() {
                    return ToolCallResult::error("Missing required parameter: query");
                }
                ToolCallResult::text(format!(
                    "{{\"query\": \"{query}\", \"results\": []}}"
                ))
            }
            "navigate" => {
                let from_id = arguments
                    .and_then(|a| a.get("from_id"))
                    .and_then(|v| v.as_str());
                let to_id = arguments
                    .and_then(|a| a.get("to_id"))
                    .and_then(|v| v.as_str());
                match (from_id, to_id) {
                    (Some(f), Some(t)) => ToolCallResult::text(format!(
                        "{{\"from\": \"{f}\", \"to\": \"{t}\", \"path\": []}}"
                    )),
                    _ => {
                        ToolCallResult::error("Missing required parameters: from_id, to_id")
                    }
                }
            }
            "find_tunnels" => ToolCallResult::text("{\"tunnels\": []}"),
            "graph_stats" => {
                ToolCallResult::text(serde_json::to_string(&self.stats).unwrap_or_default())
            }

            // Palace Operations (write)
            "add_drawer" => {
                let content = arguments
                    .and_then(|a| a.get("content"))
                    .and_then(|v| v.as_str());
                let wing = arguments
                    .and_then(|a| a.get("wing"))
                    .and_then(|v| v.as_str());
                let room = arguments
                    .and_then(|a| a.get("room"))
                    .and_then(|v| v.as_str());
                match (content, wing, room) {
                    (Some(_c), Some(_w), Some(_r)) => ToolCallResult::text(
                        "{\"drawer_id\": \"pending\", \"status\": \"no_backend\"}",
                    ),
                    _ => ToolCallResult::error(
                        "Missing required parameters: content, wing, room",
                    ),
                }
            }
            "delete_drawer" => {
                let drawer_id = arguments
                    .and_then(|a| a.get("drawer_id"))
                    .and_then(|v| v.as_str());
                match drawer_id {
                    Some(id) => ToolCallResult::text(format!(
                        "{{\"deleted\": \"{id}\", \"status\": \"no_backend\"}}"
                    )),
                    None => {
                        ToolCallResult::error("Missing required parameter: drawer_id")
                    }
                }
            }
            "add_wing" => ToolCallResult::text(
                "{\"wing_id\": \"pending\", \"status\": \"no_backend\"}",
            ),
            "add_room" => ToolCallResult::text(
                "{\"room_id\": \"pending\", \"status\": \"no_backend\"}",
            ),
            "check_duplicate" => {
                let content = arguments
                    .and_then(|a| a.get("content"))
                    .and_then(|v| v.as_str());
                match content {
                    Some(_) => ToolCallResult::text("{\"duplicates\": []}"),
                    None => {
                        ToolCallResult::error("Missing required parameter: content")
                    }
                }
            }

            // Knowledge Graph
            "kg_add" => ToolCallResult::text(
                "{\"triple_id\": \"pending\", \"status\": \"no_backend\"}",
            ),
            "kg_query" => {
                let entity = arguments
                    .and_then(|a| a.get("entity"))
                    .and_then(|v| v.as_str());
                match entity {
                    Some(e) => ToolCallResult::text(format!(
                        "{{\"entity\": \"{e}\", \"relationships\": []}}"
                    )),
                    None => {
                        ToolCallResult::error("Missing required parameter: entity")
                    }
                }
            }
            "kg_invalidate" => ToolCallResult::text(
                "{\"invalidated\": true, \"status\": \"no_backend\"}",
            ),
            "kg_timeline" => {
                let entity = arguments
                    .and_then(|a| a.get("entity"))
                    .and_then(|v| v.as_str());
                match entity {
                    Some(e) => ToolCallResult::text(format!(
                        "{{\"entity\": \"{e}\", \"timeline\": []}}"
                    )),
                    None => {
                        ToolCallResult::error("Missing required parameter: entity")
                    }
                }
            }
            "kg_traverse" => ToolCallResult::text("{\"subgraph\": {}}"),
            "kg_contradictions" => ToolCallResult::text("{\"contradictions\": []}"),
            "kg_stats" => ToolCallResult::text(format!(
                "{{\"entities\": {}, \"relationships\": {}}}",
                self.stats.entity_count, self.stats.relationship_count
            )),

            // Stigmergy
            "pheromone_status" => ToolCallResult::text("{\"pheromones\": {}}"),
            "pheromone_deposit" => ToolCallResult::text(
                "{\"deposited\": true, \"status\": \"no_backend\"}",
            ),
            "hot_paths" => ToolCallResult::text("{\"paths\": []}"),
            "cold_spots" => ToolCallResult::text("{\"spots\": []}"),
            "decay_now" => ToolCallResult::text(
                "{\"decayed\": true, \"status\": \"no_backend\"}",
            ),

            // Agent Diary
            "list_agents" => ToolCallResult::text(format!(
                "{{\"agents\": [], \"count\": {}}}",
                self.stats.agent_count
            )),
            "diary_write" => ToolCallResult::text(
                "{\"written\": true, \"status\": \"no_backend\"}",
            ),
            "diary_read" => {
                let agent_id = arguments
                    .and_then(|a| a.get("agent_id"))
                    .and_then(|v| v.as_str());
                match agent_id {
                    Some(id) => ToolCallResult::text(format!(
                        "{{\"agent_id\": \"{id}\", \"entries\": []}}"
                    )),
                    None => {
                        ToolCallResult::error("Missing required parameter: agent_id")
                    }
                }
            }

            _ => ToolCallResult::error(format!("Tool '{name}' exists but has no handler")),
        }
    }

    /// Get the number of registered tools.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Get a list of all tool names.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    /// Update palace statistics.
    pub fn update_stats(&mut self, stats: PalaceStats) {
        self.stats = stats;
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(method: &str, params: Option<Value>) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(1.into())),
            method: method.to_string(),
            params,
        }
    }

    #[test]
    fn server_creation() {
        let server = McpServer::new();
        assert!(!server.is_initialized());
        assert_eq!(server.tool_count(), 28);
    }

    #[test]
    fn initialize_sets_flag() {
        let mut server = McpServer::new();
        let req = make_request("initialize", None);
        let resp = server.handle_message(&req);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        assert!(server.is_initialized());
    }

    #[test]
    fn initialize_returns_capabilities() {
        let mut server = McpServer::new();
        let req = make_request("initialize", None);
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
        assert_eq!(result["serverInfo"]["version"], SERVER_VERSION);
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_returns_28_tools() {
        let server = McpServer::new();
        let req = make_request("tools/list", None);
        // handle_tools_list doesn't need &mut since it's a read, but
        // handle_message takes &mut self — need a mut binding.
        let mut server = server;
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 28);
    }

    #[test]
    fn tools_list_has_names_and_descriptions() {
        let mut server = McpServer::new();
        let req = make_request("tools/list", None);
        let resp = server.handle_message(&req);
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        for tool in &tools {
            assert!(tool["name"].is_string());
            assert!(tool["description"].is_string());
            assert!(tool["inputSchema"].is_object());
        }
    }

    #[test]
    fn tools_call_palace_status() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "palace_status",
                "arguments": {}
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("PALACE_PROTOCOL v1.0"));
    }

    #[test]
    fn tools_call_search_with_query() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "search",
                "arguments": { "query": "test search" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("test search"));
    }

    #[test]
    fn tools_call_search_missing_query() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "search",
                "arguments": {}
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn tools_call_unknown_tool() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "nonexistent_tool",
                "arguments": {}
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Unknown tool"));
    }

    #[test]
    fn tools_call_navigate_with_params() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "navigate",
                "arguments": { "from_id": "room_a", "to_id": "room_b" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("room_a"));
        assert!(text.contains("room_b"));
    }

    #[test]
    fn tools_call_navigate_missing_params() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "navigate",
                "arguments": { "from_id": "room_a" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn method_not_found() {
        let mut server = McpServer::new();
        let req = make_request("unknown/method", None);
        let resp = server.handle_message(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn handle_json_valid() {
        let mut server = McpServer::new();
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#;
        let resp_json = server.handle_json(json);
        assert!(resp_json.contains("protocolVersion"));
    }

    #[test]
    fn handle_json_invalid() {
        let mut server = McpServer::new();
        let resp_json = server.handle_json("not json at all");
        assert!(resp_json.contains("Parse error"));
    }

    #[test]
    fn tools_call_missing_params() {
        let mut server = McpServer::new();
        let req = make_request("tools/call", None);
        let resp = server.handle_message(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[test]
    fn prompts_list() {
        let mut server = McpServer::new();
        let req = make_request("prompts/list", None);
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let prompts = result["prompts"].as_array().unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0]["name"], "palace_protocol");
    }

    #[test]
    fn prompts_get() {
        let mut server = McpServer::new();
        let req = make_request("prompts/get", None);
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["messages"][0]["content"]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("PALACE_PROTOCOL"));
    }

    #[test]
    fn tool_names_list() {
        let server = McpServer::new();
        let names = server.tool_names();
        assert_eq!(names.len(), 28);
        assert!(names.contains(&"palace_status".to_string()));
        assert!(names.contains(&"search".to_string()));
        assert!(names.contains(&"add_drawer".to_string()));
        assert!(names.contains(&"kg_add".to_string()));
    }

    #[test]
    fn update_stats() {
        let mut server = McpServer::new();
        assert_eq!(server.stats.wing_count, 0);
        server.update_stats(PalaceStats {
            wing_count: 5,
            room_count: 20,
            ..PalaceStats::default()
        });
        assert_eq!(server.stats.wing_count, 5);
    }

    #[test]
    fn server_with_stats() {
        let server = McpServer::with_stats(PalaceStats {
            wing_count: 3,
            ..PalaceStats::default()
        });
        assert_eq!(server.stats.wing_count, 3);
    }

    #[test]
    fn all_28_tools_dispatchable() {
        let mut server = McpServer::new();
        let tool_names = server.tool_names();
        for name in &tool_names {
            let result = server.dispatch_tool(name, Some(&serde_json::json!({})));
            // All tools should return something (not panic)
            assert!(
                !result.content.is_empty(),
                "Tool {name} returned empty content"
            );
        }
    }

    #[test]
    fn json_rpc_response_success() {
        let resp =
            JsonRpcResponse::success(Some(Value::Number(1.into())), serde_json::json!("ok"));
        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.error.is_none());
        assert_eq!(resp.result.unwrap(), "ok");
    }

    #[test]
    fn json_rpc_response_error() {
        let resp =
            JsonRpcResponse::error(Some(Value::Number(1.into())), -32600, "Bad request");
        assert!(resp.result.is_none());
        assert_eq!(resp.error.as_ref().unwrap().code, -32600);
    }

    #[test]
    fn tool_call_result_text() {
        let r = ToolCallResult::text("hello");
        assert_eq!(r.content[0].text, "hello");
        assert!(r.is_error.is_none());
    }

    #[test]
    fn tool_call_result_error() {
        let r = ToolCallResult::error("oops");
        assert_eq!(r.content[0].text, "oops");
        assert_eq!(r.is_error, Some(true));
    }

    #[test]
    fn tools_call_add_drawer_with_params() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "add_drawer",
                "arguments": { "content": "test content", "wing": "w1", "room": "r1" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        assert!(result["isError"].is_null());
    }

    #[test]
    fn tools_call_add_drawer_missing_params() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "add_drawer",
                "arguments": { "content": "test" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn tools_call_kg_query_with_entity() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "kg_query",
                "arguments": { "entity": "Einstein" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Einstein"));
    }

    #[test]
    fn initialized_notification() {
        let mut server = McpServer::new();
        let req = make_request("initialized", None);
        let resp = server.handle_message(&req);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn default_server() {
        let server = McpServer::default();
        assert!(!server.is_initialized());
        assert_eq!(server.tool_count(), 28);
    }

    #[test]
    fn tools_call_delete_drawer() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "delete_drawer",
                "arguments": { "drawer_id": "d123" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("d123"));
    }

    #[test]
    fn tools_call_diary_read() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "diary_read",
                "arguments": { "agent_id": "astro-scout" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("astro-scout"));
    }

    #[test]
    fn tools_call_kg_timeline() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "kg_timeline",
                "arguments": { "entity": "Earth" }
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Earth"));
    }

    #[test]
    fn graph_stats_returns_stats_json() {
        let mut server = McpServer::with_stats(PalaceStats {
            wing_count: 7,
            entity_count: 42,
            relationship_count: 100,
            ..PalaceStats::default()
        });
        let req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "graph_stats",
                "arguments": {}
            })),
        );
        let resp = server.handle_message(&req);
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        // Should contain the stats as JSON
        assert!(text.contains("42") || text.contains("entity_count"));
    }
}
