//! Search result types and pheromone boosting.

use serde::{Deserialize, Serialize};

/// A single result from a semantic search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// ID of the matched drawer.
    pub drawer_id: String,
    /// Content of the matched drawer.
    pub content: String,
    /// Cosine similarity score (potentially boosted by pheromones).
    pub score: f32,
    /// Name of the wing containing this drawer.
    pub wing_name: String,
    /// Name of the room containing this drawer.
    pub room_name: String,
}

/// A potential duplicate match found by `check_duplicate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateMatch {
    /// ID of the existing drawer that matches.
    pub drawer_id: String,
    /// Content of the existing drawer.
    pub content: String,
    /// Cosine similarity score between the candidate and this drawer.
    pub similarity: f32,
    /// Wing containing the matched drawer.
    pub wing: String,
    /// Room containing the matched drawer.
    pub room: String,
}

/// Boosts search results based on node pheromone levels.
///
/// `boosted_score = raw_score × (1.0 + boost_factor × exploitation)`
///
/// This makes frequently-accessed drawers rank slightly higher in search,
/// reflecting the "well-trodden path" signal from stigmergy.
pub struct PheromoneBooster {
    /// How much exploitation pheromone influences ranking. Default 0.3.
    pub boost_factor: f32,
}

impl Default for PheromoneBooster {
    fn default() -> Self {
        Self { boost_factor: 0.3 }
    }
}

impl PheromoneBooster {
    /// Create a new booster with the given factor.
    pub fn new(boost_factor: f32) -> Self {
        Self { boost_factor }
    }

    /// Compute the boosted score for a drawer.
    ///
    /// # Arguments
    /// - `raw_score`: cosine similarity in [-1, 1]
    /// - `exploitation`: node exploitation pheromone level (≥ 0)
    pub fn boost(&self, raw_score: f32, exploitation: f64) -> f32 {
        raw_score * (1.0 + self.boost_factor * exploitation as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_result_serialization() {
        let r = SearchResult {
            drawer_id: "drawer_1".into(),
            content: "The sky is blue".into(),
            score: 0.95,
            wing_name: "Science".into(),
            room_name: "Physics".into(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let deser: SearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.drawer_id, "drawer_1");
        assert_eq!(deser.content, "The sky is blue");
        assert!((deser.score - 0.95).abs() < 1e-5);
    }

    #[test]
    fn booster_default() {
        let b = PheromoneBooster::default();
        assert!((b.boost_factor - 0.3).abs() < 1e-6);
    }

    #[test]
    fn booster_no_pheromones() {
        let b = PheromoneBooster::default();
        let boosted = b.boost(0.8, 0.0);
        assert!((boosted - 0.8).abs() < 1e-6);
    }

    #[test]
    fn booster_with_pheromones() {
        let b = PheromoneBooster::new(0.1);
        // 0.8 × (1 + 0.1 × 2.0) = 0.8 × 1.2 = 0.96
        let boosted = b.boost(0.8, 2.0);
        assert!((boosted - 0.96).abs() < 1e-5);
    }

    #[test]
    fn booster_zero_factor() {
        let b = PheromoneBooster::new(0.0);
        let boosted = b.boost(0.5, 100.0);
        assert!((boosted - 0.5).abs() < 1e-6);
    }

    #[test]
    fn booster_high_exploitation() {
        let b = PheromoneBooster::new(0.1);
        // 0.9 × (1 + 0.1 × 10.0) = 0.9 × 2.0 = 1.8
        let boosted = b.boost(0.9, 10.0);
        assert!((boosted - 1.8).abs() < 1e-5);
    }
}
