//! Bayesian belief state for Active Inference agents (spec §6.3).
//!
//! Beliefs are represented as Gaussian distributions parameterized by
//! mean and precision (inverse variance). Updates use precision-weighted
//! Bayesian fusion.

use serde::{Deserialize, Serialize};

/// Default belief state prior mean.
pub const DEFAULT_PRIOR_MEAN: f64 = 20.0;

/// Default belief state prior precision (low confidence).
pub const DEFAULT_PRIOR_PRECISION: f64 = 0.1;

/// A Gaussian belief represented by mean and precision (1/variance).
///
/// Precision-weighted representation makes Bayesian updates a simple
/// addition in information space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    pub mean: f64,
    pub precision: f64,
}

impl Default for BeliefState {
    fn default() -> Self {
        Self {
            mean: DEFAULT_PRIOR_MEAN,
            precision: DEFAULT_PRIOR_PRECISION,
        }
    }
}

impl BeliefState {
    /// Create a new belief state with given mean and precision.
    pub fn new(mean: f64, precision: f64) -> Self {
        Self { mean, precision }
    }

    /// Precision-weighted Bayesian update.
    ///
    /// After observing `observation` with `observation_precision`, the
    /// posterior precision is the sum of prior and observation precisions,
    /// and the posterior mean is the precision-weighted average.
    pub fn update(&mut self, observation: f64, observation_precision: f64) {
        let prior_precision = self.precision;
        let prior_mean = self.mean;
        self.precision = prior_precision + observation_precision;
        self.mean =
            (prior_precision * prior_mean + observation_precision * observation) / self.precision;
    }

    /// Merge multiple belief states via precision-weighted averaging.
    ///
    /// The merged belief has precision equal to the sum of all input
    /// precisions, and mean equal to the precision-weighted average of
    /// all input means.
    ///
    /// # Panics
    ///
    /// Panics if `beliefs` is empty.
    pub fn merge(beliefs: &[&BeliefState]) -> BeliefState {
        assert!(!beliefs.is_empty(), "cannot merge zero beliefs");
        let total_precision: f64 = beliefs.iter().map(|b| b.precision).sum();
        let merged_mean: f64 =
            beliefs.iter().map(|b| b.precision * b.mean).sum::<f64>() / total_precision;
        BeliefState {
            mean: merged_mean,
            precision: total_precision,
        }
    }

    /// Variance (1/precision).
    pub fn variance(&self) -> f64 {
        1.0 / self.precision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_belief_state_values() {
        let b = BeliefState::default();
        assert!((b.mean - DEFAULT_PRIOR_MEAN).abs() < f64::EPSILON);
        assert!((b.precision - DEFAULT_PRIOR_PRECISION).abs() < f64::EPSILON);
    }

    #[test]
    fn default_variance_is_inverse_precision() {
        let b = BeliefState::default();
        assert!((b.variance() - 1.0 / DEFAULT_PRIOR_PRECISION).abs() < f64::EPSILON);
    }

    #[test]
    fn single_update_increases_precision() {
        let mut b = BeliefState::default();
        let old_precision = b.precision;
        b.update(25.0, 1.0);
        assert!(b.precision > old_precision);
        assert!((b.precision - (old_precision + 1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn single_update_shifts_mean_toward_observation() {
        let mut b = BeliefState::default(); // mean=20, prec=0.1
        b.update(30.0, 1.0); // strong observation at 30
        // Mean should shift toward 30
        assert!(b.mean > 20.0);
        assert!(b.mean < 30.0);
        // Exact: (0.1*20 + 1.0*30) / 1.1 = (2 + 30) / 1.1 = 29.0909...
        let expected = (0.1 * 20.0 + 1.0 * 30.0) / 1.1;
        assert!((b.mean - expected).abs() < 1e-10);
    }

    #[test]
    fn multiple_updates_converge() {
        let mut b = BeliefState::default(); // mean=20, prec=0.1
        // Many precise observations at 50
        for _ in 0..100 {
            b.update(50.0, 1.0);
        }
        // Should be very close to 50 with high precision
        assert!((b.mean - 50.0).abs() < 0.1);
        assert!(b.precision > 100.0);
    }

    #[test]
    fn merge_two_beliefs() {
        let a = BeliefState::new(10.0, 2.0);
        let b = BeliefState::new(20.0, 3.0);
        let merged = BeliefState::merge(&[&a, &b]);
        // Total precision: 2 + 3 = 5
        assert!((merged.precision - 5.0).abs() < f64::EPSILON);
        // Merged mean: (2*10 + 3*20) / 5 = (20 + 60) / 5 = 16
        assert!((merged.mean - 16.0).abs() < 1e-10);
    }

    #[test]
    fn merge_single_belief_is_identity() {
        let a = BeliefState::new(42.0, 7.0);
        let merged = BeliefState::merge(&[&a]);
        assert!((merged.mean - 42.0).abs() < f64::EPSILON);
        assert!((merged.precision - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn merge_equal_beliefs_averages() {
        let a = BeliefState::new(10.0, 1.0);
        let b = BeliefState::new(20.0, 1.0);
        let merged = BeliefState::merge(&[&a, &b]);
        assert!((merged.mean - 15.0).abs() < 1e-10);
        assert!((merged.precision - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "cannot merge zero beliefs")]
    fn merge_empty_panics() {
        let _ = BeliefState::merge(&[]);
    }

    #[test]
    fn serialization_roundtrip() {
        let b = BeliefState::new(3.14, 2.72);
        let json = serde_json::to_string(&b).unwrap();
        let b2: BeliefState = serde_json::from_str(&json).unwrap();
        assert!((b.mean - b2.mean).abs() < f64::EPSILON);
        assert!((b.precision - b2.precision).abs() < f64::EPSILON);
    }
}
