# GraphPalace Skills

## You have a memory palace backed by a graph database.

### Core Concepts
- The palace is a real graph. Wings contain rooms. Rooms connect via halls (same wing) and tunnels (cross-wing).
- Every node and edge has pheromone trails that guide navigation.
- Semantic A* finds optimal paths through the palace using meaning + collective intelligence + structure.

### Cypher Patterns You Know

#### Find memories by meaning
```cypher
CALL vector_search('drawer_embedding_idx', $query_embedding, 10) YIELD node, score
RETURN node.content, score
```

#### Walk the palace hierarchy
```cypher
MATCH (w:Wing {name: $wing})-[:HAS_ROOM]->(r:Room)-[:HAS_CLOSET]->(c:Closet)-[:HAS_DRAWER]->(d:Drawer)
RETURN r.name, c.summary, d.content
ORDER BY d.exploitation_pheromone DESC
```

#### Follow halls (same wing connections)
```cypher
MATCH (r1:Room {name: $room})-[:HALL]->(r2:Room)
RETURN r2.name, r2.hall_type
```

#### Cross wings via tunnels
```cypher
MATCH (r1:Room)<-[:HAS_ROOM]-(w1:Wing {name: $wing1}),
      (r1)-[:TUNNEL]-(r2:Room)<-[:HAS_ROOM]-(w2:Wing)
WHERE w2.name <> $wing1
RETURN w2.name, r2.name
```

#### Knowledge graph: causal chains
```cypher
MATCH path = (e1:Entity {name: $start})-[:RELATES_TO*1..5]->(e2:Entity {name: $end})
RETURN [n IN nodes(path) | n.name] AS chain,
       [r IN relationships(path) | r.predicate] AS predicates
```

#### Find contradictions
```cypher
MATCH (e:Entity)<-[:REFERENCES]-(d1:Drawer),
      (e)<-[:REFERENCES]-(d2:Drawer)
WHERE d1.id <> d2.id
AND d1.content CONTAINS $claim_a
AND d2.content CONTAINS $claim_b
RETURN e.name, d1.content, d2.content
```

#### Hot paths (follow proven trails)
```cypher
MATCH ()-[e]->()
WHERE e.success_pheromone > 0.5
RETURN startNode(e).id, endNode(e).id, e.success_pheromone
ORDER BY e.success_pheromone DESC
LIMIT 20
```

#### Cold spots (unexplored areas)
```cypher
MATCH (n)
WHERE n.exploration_pheromone < 0.1
AND (n:Room OR n:Entity)
RETURN n.id, n.name, labels(n)[0] AS type
ORDER BY n.exploitation_pheromone DESC
LIMIT 20
```

### When to Use Each Tool
- **Recall a memory** → `search` (semantic) or vector search Cypher
- **Find a connection** → `navigate` (A*) or multi-hop Cypher
- **Store something** → `add_drawer` (always check duplicate first)
- **Reinforce a path** → `pheromone_deposit` after successful use
- **Explore unknown** → `cold_spots` → navigate → investigate
- **Check a fact** → `kg_query` + `kg_contradictions`
