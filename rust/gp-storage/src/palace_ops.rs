//! High-level palace CRUD operations using [`InMemoryBackend`].
//!
//! These functions provide a convenient facade over the backend's direct
//! methods, accepting the backend as the first parameter.

use crate::backend::Value;
use crate::memory::{InMemoryBackend, Relationship};
use gp_core::types::*;
use gp_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Taxonomy types
// ---------------------------------------------------------------------------

/// A snapshot of the complete palace taxonomy (hierarchy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceTaxonomy {
    pub wings: Vec<WingTaxonomy>,
    pub total_wings: usize,
    pub total_rooms: usize,
    pub total_closets: usize,
    pub total_drawers: usize,
    pub total_entities: usize,
}

/// Taxonomy entry for a single wing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WingTaxonomy {
    pub id: String,
    pub name: String,
    pub wing_type: String,
    pub rooms: Vec<RoomTaxonomy>,
}

/// Taxonomy entry for a single room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomTaxonomy {
    pub id: String,
    pub name: String,
    pub hall_type: String,
    pub closet_count: usize,
    pub drawer_count: usize,
}

// ---------------------------------------------------------------------------
// Palace operations
// ---------------------------------------------------------------------------

/// Create a new wing in the palace.
pub fn create_wing(
    backend: &InMemoryBackend,
    name: &str,
    wing_type: WingType,
    description: &str,
    embedding: Embedding,
) -> Result<String> {
    backend.create_wing(name, wing_type, description, embedding)
}

/// Create a new room in a wing.
pub fn create_room(
    backend: &InMemoryBackend,
    wing_id: &str,
    name: &str,
    hall_type: HallType,
    description: &str,
    embedding: Embedding,
) -> Result<String> {
    backend.create_room(wing_id, name, hall_type, description, embedding)
}

/// Create a new closet in a room.
pub fn create_closet(
    backend: &InMemoryBackend,
    room_id: &str,
    name: &str,
    summary: &str,
    embedding: Embedding,
) -> Result<String> {
    backend.create_closet(room_id, name, summary, embedding)
}

/// Create a new drawer in a closet.
pub fn create_drawer(
    backend: &InMemoryBackend,
    closet_id: &str,
    content: &str,
    embedding: Embedding,
    source: DrawerSource,
    source_file: Option<&str>,
    importance: f64,
) -> Result<String> {
    backend.create_drawer(closet_id, content, embedding, source, source_file, importance)
}

/// Get a wing by ID.
pub fn get_wing(backend: &InMemoryBackend, id: &str) -> Result<Wing> {
    backend.get_wing(id)
}

/// Search drawers by embedding similarity.
pub fn search_drawers(
    backend: &InMemoryBackend,
    query_embedding: &Embedding,
    k: usize,
) -> Result<Vec<(Drawer, f32)>> {
    Ok(backend.search_drawers(query_embedding, k, 0.0))
}

/// Get the complete palace taxonomy.
pub fn get_taxonomy(backend: &InMemoryBackend) -> Result<PalaceTaxonomy> {
    let d = backend.read_data();
    let mut wings_tax = Vec::new();
    for wing in d.wings.values() {
        let rooms: Vec<_> = d
            .parent_map
            .iter()
            .filter(|(_, parent)| *parent == &wing.id)
            .filter_map(|(child_id, _)| d.rooms.get(child_id))
            .map(|room| {
                let closet_ids: Vec<_> = d
                    .parent_map
                    .iter()
                    .filter(|(_, parent)| *parent == &room.id)
                    .filter_map(|(child_id, _)| {
                        if d.closets.contains_key(child_id) {
                            Some(child_id.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                let drawer_count: usize = closet_ids
                    .iter()
                    .filter_map(|cid| d.closets.get(cid))
                    .map(|c| c.drawer_count as usize)
                    .sum();
                RoomTaxonomy {
                    id: room.id.clone(),
                    name: room.name.clone(),
                    hall_type: format!("{:?}", room.hall_type),
                    closet_count: closet_ids.len(),
                    drawer_count,
                }
            })
            .collect();
        wings_tax.push(WingTaxonomy {
            id: wing.id.clone(),
            name: wing.name.clone(),
            wing_type: format!("{:?}", wing.wing_type),
            rooms,
        });
    }
    Ok(PalaceTaxonomy {
        total_wings: d.wings.len(),
        total_rooms: d.rooms.len(),
        total_closets: d.closets.len(),
        total_drawers: d.drawers.len(),
        total_entities: d.entities.len(),
        wings: wings_tax,
    })
}

/// Add an entity to the knowledge graph.
pub fn add_entity(
    backend: &InMemoryBackend,
    name: &str,
    entity_type: EntityType,
    description: &str,
    embedding: Embedding,
) -> Result<String> {
    backend.create_entity(name, entity_type, description, embedding)
}

/// Add a relationship triple to the knowledge graph.
pub fn add_relationship(
    backend: &InMemoryBackend,
    subject: &str,
    predicate: &str,
    object: &str,
    confidence: f64,
) -> Result<String> {
    backend.add_relationship(subject, predicate, object, confidence)
}

/// Query all relationships involving an entity.
pub fn query_relationships(
    backend: &InMemoryBackend,
    entity: &str,
) -> Vec<Relationship> {
    backend.query_relationships(entity)
}

/// Get a drawer by its ID.
pub fn get_drawer(backend: &InMemoryBackend, id: &str) -> Result<Drawer> {
    backend.get_drawer(id)
}

/// Convert backend query results to a simpler format.
pub fn rows_to_strings(
    rows: &[HashMap<String, Value>],
) -> Vec<HashMap<String, String>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_emb() -> Embedding {
        [0.0f32; 384]
    }

    fn make_emb(seed: u8) -> Embedding {
        let mut emb = [0.0f32; 384];
        for (i, v) in emb.iter_mut().enumerate() {
            *v = ((seed as f32 + i as f32) * 0.01).sin();
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in emb.iter_mut() {
                *v /= norm;
            }
        }
        emb
    }

    #[test]
    fn test_create_wing_op() {
        let b = InMemoryBackend::new();
        let id = create_wing(&b, "Science", WingType::Domain, "desc", zero_emb()).unwrap();
        assert!(id.starts_with("wing_"));
    }

    #[test]
    fn test_create_room_op() {
        let b = InMemoryBackend::new();
        let wid = create_wing(&b, "W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = create_room(&b, &wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        assert!(rid.starts_with("room_"));
    }

    #[test]
    fn test_create_closet_op() {
        let b = InMemoryBackend::new();
        let wid = create_wing(&b, "W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = create_room(&b, &wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = create_closet(&b, &rid, "C", "S", zero_emb()).unwrap();
        assert!(cid.starts_with("closet_"));
    }

    #[test]
    fn test_create_drawer_op() {
        let b = InMemoryBackend::new();
        let wid = create_wing(&b, "W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = create_room(&b, &wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = create_closet(&b, &rid, "C", "S", zero_emb()).unwrap();
        let did = create_drawer(
            &b, &cid, "content", make_emb(1), DrawerSource::Conversation, None, 0.5,
        ).unwrap();
        assert!(did.starts_with("drawer_"));
    }

    #[test]
    fn test_get_wing_op() {
        let b = InMemoryBackend::new();
        let id = create_wing(&b, "Science", WingType::Domain, "desc", zero_emb()).unwrap();
        let wing = get_wing(&b, &id).unwrap();
        assert_eq!(wing.name, "Science");
    }

    #[test]
    fn test_search_drawers_op() {
        let b = InMemoryBackend::new();
        let wid = create_wing(&b, "W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = create_room(&b, &wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = create_closet(&b, &rid, "C", "S", zero_emb()).unwrap();
        create_drawer(&b, &cid, "hello", make_emb(1), DrawerSource::Conversation, None, 0.5).unwrap();
        let results = search_drawers(&b, &make_emb(1), 5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_get_taxonomy() {
        let b = InMemoryBackend::new();
        let wid = create_wing(&b, "Science", WingType::Domain, "", zero_emb()).unwrap();
        let rid = create_room(&b, &wid, "Physics", HallType::Facts, "", zero_emb()).unwrap();
        let cid = create_closet(&b, &rid, "Mechanics", "Classical", zero_emb()).unwrap();
        create_drawer(&b, &cid, "F=ma", make_emb(1), DrawerSource::Conversation, None, 0.9).unwrap();

        let tax = get_taxonomy(&b).unwrap();
        assert_eq!(tax.total_wings, 1);
        assert_eq!(tax.total_rooms, 1);
        assert_eq!(tax.total_closets, 1);
        assert_eq!(tax.total_drawers, 1);
        assert_eq!(tax.wings.len(), 1);
        assert_eq!(tax.wings[0].rooms.len(), 1);
        assert_eq!(tax.wings[0].rooms[0].closet_count, 1);
        assert_eq!(tax.wings[0].rooms[0].drawer_count, 1);
    }

    #[test]
    fn test_add_entity_op() {
        let b = InMemoryBackend::new();
        let id = add_entity(&b, "Einstein", EntityType::Person, "Physicist", zero_emb()).unwrap();
        assert!(id.starts_with("entity_"));
    }

    #[test]
    fn test_add_relationship_op() {
        let b = InMemoryBackend::new();
        let e1 = add_entity(&b, "A", EntityType::Person, "", zero_emb()).unwrap();
        let e2 = add_entity(&b, "B", EntityType::Concept, "", zero_emb()).unwrap();
        let rid = add_relationship(&b, &e1, "knows", &e2, 0.9).unwrap();
        assert!(rid.starts_with("rel_"));
    }

    #[test]
    fn test_query_relationships_op() {
        let b = InMemoryBackend::new();
        let e1 = add_entity(&b, "A", EntityType::Person, "", zero_emb()).unwrap();
        let e2 = add_entity(&b, "B", EntityType::Concept, "", zero_emb()).unwrap();
        add_relationship(&b, &e1, "knows", &e2, 0.9).unwrap();
        let rels = query_relationships(&b, &e1);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].predicate, "knows");
    }

    #[test]
    fn test_get_drawer_op() {
        let b = InMemoryBackend::new();
        let wid = create_wing(&b, "W", WingType::Domain, "", zero_emb()).unwrap();
        let rid = create_room(&b, &wid, "R", HallType::Facts, "", zero_emb()).unwrap();
        let cid = create_closet(&b, &rid, "C", "S", zero_emb()).unwrap();
        let did = create_drawer(&b, &cid, "hello", zero_emb(), DrawerSource::Conversation, None, 0.5).unwrap();
        let drawer = get_drawer(&b, &did).unwrap();
        assert_eq!(drawer.content, "hello");
    }

    #[test]
    fn test_rows_to_strings() {
        let mut row = HashMap::new();
        row.insert("name".to_string(), Value::String("test".to_string()));
        row.insert("count".to_string(), Value::Int(42));
        let result = rows_to_strings(&[row]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("name").unwrap(), "test");
        assert_eq!(result[0].get("count").unwrap(), "42");
    }

    #[test]
    fn test_taxonomy_multiple_wings() {
        let b = InMemoryBackend::new();
        create_wing(&b, "A", WingType::Domain, "", zero_emb()).unwrap();
        create_wing(&b, "B", WingType::Project, "", zero_emb()).unwrap();
        let tax = get_taxonomy(&b).unwrap();
        assert_eq!(tax.total_wings, 2);
        assert_eq!(tax.wings.len(), 2);
    }

    #[test]
    fn test_taxonomy_empty_palace() {
        let b = InMemoryBackend::new();
        let tax = get_taxonomy(&b).unwrap();
        assert_eq!(tax.total_wings, 0);
        assert!(tax.wings.is_empty());
    }
}
