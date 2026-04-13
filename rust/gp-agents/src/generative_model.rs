//! Generative model with online statistics (Welford's algorithm).
//!
//! The generative model maintains running statistics for observed
//! quantities and provides predictions (mean, variance) for each key.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Online statistics using Welford's algorithm.
///
/// Computes running mean and variance in a single pass with O(1) memory,
/// numerically stable even for large datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelfordStats {
    /// Number of observations.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Sum of squared differences from the mean (M2 in Welford's notation).
    pub m2: f64,
}

impl WelfordStats {
    /// Create a new empty statistics tracker.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Update with a new observation using Welford's online algorithm.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Population variance (for count >= 2, otherwise 0.0).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Sample variance (Bessel-corrected, for count >= 2).
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sample standard deviation (Bessel-corrected).
    pub fn sample_std_dev(&self) -> f64 {
        self.sample_variance().sqrt()
    }
}

impl Default for WelfordStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Generative model for hierarchical prediction.
///
/// Maintains per-key statistics and provides predictions based on
/// observed data. The model makes no distributional assumptions beyond
/// tracking mean and variance.
pub struct GenerativeModel {
    /// Per-key running statistics.
    pub stats: HashMap<String, WelfordStats>,
}

impl GenerativeModel {
    /// Create a new empty generative model.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// Observe a value for a given key.
    pub fn observe(&mut self, key: &str, value: f64) {
        self.stats
            .entry(key.to_string())
            .or_default()
            .update(value);
    }

    /// Predict (mean, variance) for a key, or `None` if no data.
    pub fn predict(&self, key: &str) -> Option<(f64, f64)> {
        self.stats.get(key).map(|s| (s.mean, s.variance()))
    }

    /// Number of distinct keys being tracked.
    pub fn num_keys(&self) -> usize {
        self.stats.len()
    }
}

impl Default for GenerativeModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_single_value_variance_zero() {
        let mut w = WelfordStats::new();
        w.update(42.0);
        assert!((w.mean - 42.0).abs() < f64::EPSILON);
        assert!((w.variance() - 0.0).abs() < f64::EPSILON);
        assert_eq!(w.count, 1);
    }

    #[test]
    fn welford_known_sequence() {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5.0, Population variance = 4.0
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut w = WelfordStats::new();
        for &v in &values {
            w.update(v);
        }
        assert_eq!(w.count, 8);
        assert!((w.mean - 5.0).abs() < 1e-10, "mean={}", w.mean);
        assert!(
            (w.variance() - 4.0).abs() < 1e-10,
            "variance={}",
            w.variance()
        );
        // Sample variance = 4.0 * 8/7 ≈ 4.571
        let expected_sample_var = 4.0 * 8.0 / 7.0;
        assert!(
            (w.sample_variance() - expected_sample_var).abs() < 1e-10,
            "sample_variance={}",
            w.sample_variance()
        );
    }

    #[test]
    fn welford_std_dev() {
        let mut w = WelfordStats::new();
        for &v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.update(v);
        }
        assert!((w.std_dev() - 2.0).abs() < 1e-10, "std_dev={}", w.std_dev());
    }

    #[test]
    fn welford_empty_stats() {
        let w = WelfordStats::new();
        assert_eq!(w.count, 0);
        assert!((w.mean - 0.0).abs() < f64::EPSILON);
        assert!((w.variance() - 0.0).abs() < f64::EPSILON);
        assert!((w.std_dev() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn welford_constant_values_zero_variance() {
        let mut w = WelfordStats::new();
        for _ in 0..100 {
            w.update(7.0);
        }
        assert!((w.mean - 7.0).abs() < 1e-10);
        assert!(w.variance().abs() < 1e-10);
    }

    #[test]
    fn welford_two_values() {
        let mut w = WelfordStats::new();
        w.update(10.0);
        w.update(20.0);
        assert!((w.mean - 15.0).abs() < 1e-10);
        // Population variance: ((10-15)^2 + (20-15)^2) / 2 = 25
        assert!((w.variance() - 25.0).abs() < 1e-10);
        // Sample variance: 50 / 1 = 50
        assert!((w.sample_variance() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn welford_serialization_roundtrip() {
        let mut w = WelfordStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            w.update(v);
        }
        let json = serde_json::to_string(&w).unwrap();
        let w2: WelfordStats = serde_json::from_str(&json).unwrap();
        assert_eq!(w.count, w2.count);
        assert!((w.mean - w2.mean).abs() < f64::EPSILON);
        assert!((w.m2 - w2.m2).abs() < f64::EPSILON);
    }

    // --- GenerativeModel tests ---

    #[test]
    fn model_predict_unknown_key_returns_none() {
        let model = GenerativeModel::new();
        assert!(model.predict("unknown").is_none());
    }

    #[test]
    fn model_observe_and_predict() {
        let mut model = GenerativeModel::new();
        model.observe("temperature", 20.0);
        model.observe("temperature", 22.0);
        model.observe("temperature", 21.0);

        let (mean, variance) = model.predict("temperature").unwrap();
        assert!((mean - 21.0).abs() < 1e-10, "mean={mean}");
        // Pop variance: ((20-21)^2 + (22-21)^2 + (21-21)^2) / 3 = 2/3
        assert!(
            (variance - 2.0 / 3.0).abs() < 1e-10,
            "variance={variance}"
        );
    }

    #[test]
    fn model_multiple_keys() {
        let mut model = GenerativeModel::new();
        model.observe("a", 1.0);
        model.observe("b", 2.0);
        model.observe("a", 3.0);

        assert_eq!(model.num_keys(), 2);
        let (mean_a, _) = model.predict("a").unwrap();
        let (mean_b, _) = model.predict("b").unwrap();
        assert!((mean_a - 2.0).abs() < 1e-10);
        assert!((mean_b - 2.0).abs() < 1e-10);
    }

    #[test]
    fn model_single_observation_zero_variance() {
        let mut model = GenerativeModel::new();
        model.observe("x", 42.0);
        let (mean, var) = model.predict("x").unwrap();
        assert!((mean - 42.0).abs() < 1e-10);
        assert!(var.abs() < 1e-10);
    }
}
