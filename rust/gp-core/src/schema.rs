//! Palace graph schema definitions as Cypher DDL.
//!
//! These statements initialize the GraphPalace schema in Kuzu.
//! All definitions map to the types in [`crate::types`].

/// Schema dialect for index creation syntax.
///
/// Different backends require different DDL syntax for index creation.
/// The core node/rel table DDL is shared, but index creation varies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchemaDialect {
    /// Standard Cypher DDL (used by InMemoryBackend and Kuzu).
    #[default]
    Cypher,
    /// LadybugDB stored-procedure syntax for index creation.
    LadybugDb,
}

/// All Cypher DDL statements for creating the GraphPalace schema.
///
/// Returns statements in the correct execution order:
/// 1. Node tables
/// 2. Relationship tables
/// 3. Vector indexes
/// 4. Full-text indexes
/// 5. Property indexes
pub fn schema_ddl() -> Vec<&'static str> {
    let mut stmts = Vec::new();
    stmts.extend(node_tables());
    stmts.extend(rel_tables());
    stmts.extend(vector_indexes());
    stmts.extend(fts_indexes());
    stmts.extend(property_indexes());
    stmts
}

/// CREATE NODE TABLE statements for all palace node types.
pub fn node_tables() -> Vec<&'static str> {
    vec![
        // Palace spatial hierarchy
        r#"CREATE NODE TABLE Palace(
    id STRING PRIMARY KEY,
    name STRING,
    description STRING,
    created_at TIMESTAMP DEFAULT current_timestamp()
)"#,
        r#"CREATE NODE TABLE Wing(
    id STRING PRIMARY KEY,
    name STRING,
    wing_type STRING,
    description STRING,
    embedding FLOAT[384],
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT current_timestamp()
)"#,
        r#"CREATE NODE TABLE Room(
    id STRING PRIMARY KEY,
    name STRING,
    hall_type STRING,
    description STRING,
    embedding FLOAT[384],
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT current_timestamp()
)"#,
        r#"CREATE NODE TABLE Closet(
    id STRING PRIMARY KEY,
    name STRING,
    summary STRING,
    embedding FLOAT[384],
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    drawer_count INT64 DEFAULT 0,
    created_at TIMESTAMP DEFAULT current_timestamp()
)"#,
        r#"CREATE NODE TABLE Drawer(
    id STRING PRIMARY KEY,
    content STRING,
    embedding FLOAT[384],
    source STRING,
    source_file STRING,
    importance FLOAT DEFAULT 0.5,
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT current_timestamp(),
    accessed_at TIMESTAMP DEFAULT current_timestamp()
)"#,
        // Knowledge graph entities
        r#"CREATE NODE TABLE Entity(
    id STRING PRIMARY KEY,
    name STRING,
    entity_type STRING,
    description STRING,
    embedding FLOAT[384],
    exploitation_pheromone FLOAT DEFAULT 0.0,
    exploration_pheromone FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT current_timestamp()
)"#,
        // Specialist agents
        r#"CREATE NODE TABLE Agent(
    id STRING PRIMARY KEY,
    name STRING,
    domain STRING,
    focus STRING,
    diary STRING,
    goal_embedding FLOAT[384],
    temperature FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT current_timestamp()
)"#,
    ]
}

/// CREATE REL TABLE statements for all palace relationship types.
pub fn rel_tables() -> Vec<&'static str> {
    vec![
        // Palace hierarchy (structural)
        "CREATE REL TABLE CONTAINS(FROM Palace TO Wing)",
        r#"CREATE REL TABLE HAS_ROOM(FROM Wing TO Room,
    base_cost FLOAT DEFAULT 0.3,
    current_cost FLOAT DEFAULT 0.3,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)"#,
        r#"CREATE REL TABLE HAS_CLOSET(FROM Room TO Closet,
    base_cost FLOAT DEFAULT 0.3,
    current_cost FLOAT DEFAULT 0.3,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)"#,
        r#"CREATE REL TABLE HAS_DRAWER(FROM Closet TO Drawer,
    base_cost FLOAT DEFAULT 0.3,
    current_cost FLOAT DEFAULT 0.3,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)"#,
        // Palace cross-connections (navigational)
        r#"CREATE REL TABLE HALL(FROM Room TO Room,
    hall_type STRING,
    base_cost FLOAT DEFAULT 0.5,
    current_cost FLOAT DEFAULT 0.5,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)"#,
        r#"CREATE REL TABLE TUNNEL(FROM Room TO Room,
    base_cost FLOAT DEFAULT 0.7,
    current_cost FLOAT DEFAULT 0.7,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)"#,
        // Knowledge graph (semantic)
        r#"CREATE REL TABLE RELATES_TO(FROM Entity TO Entity,
    predicate STRING,
    confidence FLOAT DEFAULT 0.5,
    base_cost FLOAT DEFAULT 1.0,
    current_cost FLOAT DEFAULT 1.0,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    observed_at TIMESTAMP DEFAULT current_timestamp()
)"#,
        // Memory ↔ Entity connections
        r#"CREATE REL TABLE REFERENCES(FROM Drawer TO Entity,
    relevance FLOAT DEFAULT 1.0,
    base_cost FLOAT DEFAULT 0.5,
    current_cost FLOAT DEFAULT 0.5,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)"#,
        // Semantic similarity (auto-computed)
        r#"CREATE REL TABLE SIMILAR_TO(FROM Drawer TO Drawer,
    similarity FLOAT,
    base_cost FLOAT,
    current_cost FLOAT,
    success_pheromone FLOAT DEFAULT 0.0,
    traversal_pheromone FLOAT DEFAULT 0.0,
    recency_pheromone FLOAT DEFAULT 0.0
)"#,
        // Agent ↔ Palace connections
        "CREATE REL TABLE MANAGES(FROM Agent TO Wing)",
        r#"CREATE REL TABLE INVESTIGATED(FROM Agent TO Drawer,
    result STRING,
    investigated_at TIMESTAMP DEFAULT current_timestamp()
)"#,
    ]
}

/// CREATE VECTOR INDEX statements for HNSW semantic search.
pub fn vector_indexes() -> Vec<&'static str> {
    vec![
        "CREATE VECTOR INDEX drawer_embedding_idx ON Drawer(embedding) WITH (metric='cosine', M=16, ef_construction=200)",
        "CREATE VECTOR INDEX entity_embedding_idx ON Entity(embedding) WITH (metric='cosine', M=16, ef_construction=200)",
        "CREATE VECTOR INDEX room_embedding_idx ON Room(embedding) WITH (metric='cosine', M=16, ef_construction=200)",
    ]
}

/// CREATE FTS INDEX statements for keyword search.
pub fn fts_indexes() -> Vec<&'static str> {
    vec![
        "CREATE FTS INDEX drawer_content_idx ON Drawer(content)",
        "CREATE FTS INDEX entity_name_idx ON Entity(name, description)",
    ]
}

/// CREATE INDEX statements for efficient property filtering.
pub fn property_indexes() -> Vec<&'static str> {
    vec![
        "CREATE INDEX wing_name_idx ON Wing(name)",
        "CREATE INDEX room_hall_idx ON Room(hall_type)",
        "CREATE INDEX entity_type_idx ON Entity(entity_type)",
        "CREATE INDEX rel_valid_idx ON RELATES_TO(valid_from, valid_to)",
    ]
}

/// Schema DDL for a specific backend dialect.
///
/// Node/rel table DDL is shared; only index creation syntax varies.
pub fn schema_ddl_for(dialect: SchemaDialect) -> Vec<String> {
    let mut stmts: Vec<String> = Vec::new();
    // Node and rel tables are dialect-independent
    stmts.extend(node_tables().iter().map(|s| s.to_string()));
    stmts.extend(rel_tables().iter().map(|s| s.to_string()));
    // Index syntax is dialect-dependent
    stmts.extend(vector_indexes_for(dialect));
    stmts.extend(fts_indexes_for(dialect));
    stmts.extend(property_indexes().iter().map(|s| s.to_string()));
    stmts
}

/// Vector index creation DDL for a specific dialect.
pub fn vector_indexes_for(dialect: SchemaDialect) -> Vec<String> {
    match dialect {
        SchemaDialect::Cypher => vector_indexes().iter().map(|s| s.to_string()).collect(),
        SchemaDialect::LadybugDb => vec![
            "CALL CREATE_VECTOR_INDEX('drawer_embedding_idx', 'Drawer', 'embedding', 'cosine', 16, 200)".to_string(),
            "CALL CREATE_VECTOR_INDEX('entity_embedding_idx', 'Entity', 'embedding', 'cosine', 16, 200)".to_string(),
            "CALL CREATE_VECTOR_INDEX('room_embedding_idx', 'Room', 'embedding', 'cosine', 16, 200)".to_string(),
        ],
    }
}

/// FTS index creation DDL for a specific dialect.
pub fn fts_indexes_for(dialect: SchemaDialect) -> Vec<String> {
    match dialect {
        SchemaDialect::Cypher => fts_indexes().iter().map(|s| s.to_string()).collect(),
        SchemaDialect::LadybugDb => vec![
            "CALL CREATE_FTS_INDEX('drawer_content_idx', 'Drawer', ['content'])".to_string(),
            "CALL CREATE_FTS_INDEX('entity_name_idx', 'Entity', ['name', 'description'])".to_string(),
        ],
    }
}

/// Cypher statements for bulk pheromone decay operations.
pub fn decay_statements() -> Vec<&'static str> {
    vec![
        // Decay all node pheromones
        r#"MATCH (n)
WHERE n.exploitation_pheromone > 0.001 OR n.exploration_pheromone > 0.001
SET n.exploitation_pheromone = n.exploitation_pheromone * (1.0 - $exploitation_rate),
    n.exploration_pheromone = n.exploration_pheromone * (1.0 - $exploration_rate)"#,
        // Decay all edge pheromones
        r#"MATCH ()-[e]->()
WHERE e.success_pheromone > 0.001 OR e.traversal_pheromone > 0.001 OR e.recency_pheromone > 0.001
SET e.success_pheromone = e.success_pheromone * (1.0 - $success_rate),
    e.traversal_pheromone = e.traversal_pheromone * (1.0 - $traversal_rate),
    e.recency_pheromone = e.recency_pheromone * (1.0 - $recency_rate)"#,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_ddl_count() {
        let stmts = schema_ddl();
        // 7 node tables + 11 rel tables + 3 vector + 2 fts + 4 property = 27
        assert_eq!(stmts.len(), 27);
    }

    #[test]
    fn test_node_tables_count() {
        assert_eq!(node_tables().len(), 7);
    }

    #[test]
    fn test_rel_tables_count() {
        assert_eq!(rel_tables().len(), 11);
    }

    #[test]
    fn test_vector_indexes_count() {
        assert_eq!(vector_indexes().len(), 3);
    }

    #[test]
    fn test_fts_indexes_count() {
        assert_eq!(fts_indexes().len(), 2);
    }

    #[test]
    fn test_property_indexes_count() {
        assert_eq!(property_indexes().len(), 4);
    }

    #[test]
    fn test_decay_statements_count() {
        assert_eq!(decay_statements().len(), 2);
    }

    #[test]
    fn test_schema_contains_palace() {
        let stmts = node_tables();
        assert!(stmts[0].contains("CREATE NODE TABLE Palace"));
    }

    #[test]
    fn test_schema_contains_embedding_dim() {
        let stmts = node_tables();
        // Wing should have FLOAT[384] for embedding
        assert!(stmts[1].contains("FLOAT[384]"));
    }

    #[test]
    fn test_rel_tables_contain_pheromones() {
        let stmts = rel_tables();
        // HAS_ROOM should have pheromone fields
        assert!(stmts[1].contains("success_pheromone"));
        assert!(stmts[1].contains("traversal_pheromone"));
        assert!(stmts[1].contains("recency_pheromone"));
    }

    // ─── Schema dialect ──────────────────────────────────────────────────

    #[test]
    fn test_schema_dialect_default_is_cypher() {
        assert_eq!(SchemaDialect::default(), SchemaDialect::Cypher);
    }

    #[test]
    fn test_vector_indexes_cypher_matches_original() {
        let original: Vec<String> = vector_indexes().iter().map(|s| s.to_string()).collect();
        let dialect = vector_indexes_for(SchemaDialect::Cypher);
        assert_eq!(original, dialect);
    }

    #[test]
    fn test_vector_indexes_ladybugdb() {
        let stmts = vector_indexes_for(SchemaDialect::LadybugDb);
        assert_eq!(stmts.len(), 3);
        assert!(stmts[0].starts_with("CALL CREATE_VECTOR_INDEX"));
        assert!(stmts[0].contains("'Drawer'"));
        assert!(stmts[0].contains("'cosine'"));
    }

    #[test]
    fn test_fts_indexes_ladybugdb() {
        let stmts = fts_indexes_for(SchemaDialect::LadybugDb);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].starts_with("CALL CREATE_FTS_INDEX"));
        assert!(stmts[1].contains("['name', 'description']"));
    }

    #[test]
    fn test_schema_ddl_for_cypher_count() {
        let stmts = schema_ddl_for(SchemaDialect::Cypher);
        // Same count as original: 7 + 11 + 3 + 2 + 4 = 27
        assert_eq!(stmts.len(), 27);
    }

    #[test]
    fn test_schema_ddl_for_ladybugdb_count() {
        let stmts = schema_ddl_for(SchemaDialect::LadybugDb);
        // Same count: 7 + 11 + 3 + 2 + 4 = 27
        assert_eq!(stmts.len(), 27);
    }
}
