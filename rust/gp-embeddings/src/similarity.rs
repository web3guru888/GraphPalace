//! Vector similarity functions for GraphPalace embeddings.

/// Compute cosine similarity between two vectors.
///
/// Returns a value in `[-1.0, 1.0]` where `1.0` means identical direction,
/// `0.0` means orthogonal, and `-1.0` means opposite direction.
///
/// If either vector has zero magnitude, returns `0.0`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    (dot / denom) as f32
}

/// Find the `k` most similar candidates to a query vector.
///
/// Only candidates with similarity ≥ `threshold` are returned.
/// Results are sorted descending by similarity score.
pub fn find_top_k(
    query: &[f32],
    candidates: &[(&str, &[f32])],
    k: usize,
    threshold: f32,
) -> Vec<(String, f32)> {
    let mut scored: Vec<(String, f32)> = candidates
        .iter()
        .map(|(label, vec)| {
            let sim = cosine_similarity(query, vec);
            (label.to_string(), sim)
        })
        .filter(|(_, sim)| *sim >= threshold)
        .collect();

    // Sort descending by similarity (use total_cmp for NaN-safety).
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let v = [1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "expected ~1.0, got {sim}");
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "expected ~0.0, got {sim}");
    }

    #[test]
    fn opposite_vectors_have_similarity_negative_one() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [-1.0f32, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6, "expected ~-1.0, got {sim}");
    }

    #[test]
    fn zero_vector_gives_zero_similarity() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&b, &a), 0.0);
        assert_eq!(cosine_similarity(&a, &a), 0.0);
    }

    #[test]
    fn known_cosine_value() {
        // cos(45°) ≈ 0.7071
        let a = [1.0f32, 0.0];
        let b = [1.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
            "expected ~0.7071, got {sim}"
        );
    }

    #[test]
    fn find_top_k_basic() {
        let query = [1.0f32, 0.0, 0.0];
        let a_vec = [1.0f32, 0.0, 0.0]; // sim = 1.0
        let b_vec = [0.0f32, 1.0, 0.0]; // sim = 0.0
        let c_vec = [0.7f32, 0.7, 0.0]; // sim ≈ 0.707

        let candidates: Vec<(&str, &[f32])> =
            vec![("a", &a_vec[..]), ("b", &b_vec[..]), ("c", &c_vec[..])];

        let results = find_top_k(&query, &candidates, 2, 0.5);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
        assert!((results[0].1 - 1.0).abs() < 1e-5);
        assert_eq!(results[1].0, "c");
    }

    #[test]
    fn find_top_k_threshold_filters() {
        let query = [1.0f32, 0.0];
        let a_vec = [0.0f32, 1.0]; // sim = 0.0
        let b_vec = [0.5f32, 0.5]; // sim ≈ 0.707

        let candidates: Vec<(&str, &[f32])> = vec![("a", &a_vec[..]), ("b", &b_vec[..])];

        let results = find_top_k(&query, &candidates, 10, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "b");
    }

    #[test]
    fn find_top_k_limits_to_k() {
        let query = [1.0f32, 0.0];
        let a_vec = [1.0f32, 0.0];
        let b_vec = [0.9f32, 0.1];
        let c_vec = [0.8f32, 0.2];

        let candidates: Vec<(&str, &[f32])> =
            vec![("a", &a_vec[..]), ("b", &b_vec[..]), ("c", &c_vec[..])];

        let results = find_top_k(&query, &candidates, 1, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn find_top_k_empty_candidates() {
        let query = [1.0f32, 0.0];
        let candidates: Vec<(&str, &[f32])> = vec![];
        let results = find_top_k(&query, &candidates, 5, 0.0);
        assert!(results.is_empty());
    }
}
