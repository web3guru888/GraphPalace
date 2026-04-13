//! Softmax action selection and temperature annealing (spec §6.4–6.5).
//!
//! Actions are selected by converting Expected Free Energy values to
//! probabilities via a softmax function, then sampling from the resulting
//! categorical distribution.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Compute softmax probabilities from EFE values.
///
/// For each candidate `(node_id, efe)`, the probability is:
///
/// ```text
/// weight_i = exp(-efe_i / temperature)
/// prob_i   = weight_i / Σ weight_j
/// ```
///
/// Since lower (more negative) EFE is better, negating gives a larger
/// exponent → higher probability for preferred actions.
///
/// Returns an empty vec if `candidates` is empty.
pub fn softmax_probabilities(
    candidates: &[(String, f64)],
    temperature: f64,
) -> Vec<(String, f64)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // For numerical stability, subtract max(-efe/T) before exp.
    let neg_efe_scaled: Vec<f64> = candidates
        .iter()
        .map(|(_, efe)| -efe / temperature)
        .collect();

    let max_val = neg_efe_scaled
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = neg_efe_scaled
        .iter()
        .map(|v| (v - max_val).exp())
        .collect();

    let total: f64 = weights.iter().sum();

    candidates
        .iter()
        .zip(weights.iter())
        .map(|((id, _), w)| (id.clone(), w / total))
        .collect()
}

/// Softmax action selection over EFE values.
///
/// Returns the `node_id` of the selected action, or `None` if candidates
/// is empty. Uses the provided RNG for stochastic sampling.
pub fn select_action(
    candidates: &[(String, f64)],
    temperature: f64,
    rng: &mut impl Rng,
) -> Option<String> {
    let probs = softmax_probabilities(candidates, temperature);
    if probs.is_empty() {
        return None;
    }

    // Sample from categorical distribution using inverse CDF.
    let u: f64 = rng.r#gen::<f64>();
    let mut cumulative = 0.0;
    for (id, p) in &probs {
        cumulative += p;
        if u <= cumulative {
            return Some(id.clone());
        }
    }

    // Floating-point edge case: return last element.
    probs.last().map(|(id, _)| id.clone())
}

/// Temperature annealing schedules from §6.5.
///
/// Controls how the exploration temperature changes over time,
/// typically decreasing to shift from exploration to exploitation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnealingSchedule {
    /// Linear interpolation from `start` to `end`.
    Linear { start: f64, end: f64 },
    /// Exponential decay: `start * exp(-decay * progress)`.
    Exponential { start: f64, decay: f64 },
    /// Cosine annealing: smooth transition from `start` to `end`.
    Cosine { start: f64, end: f64 },
}

impl AnnealingSchedule {
    /// Compute temperature at given progress ∈ [0.0, 1.0].
    ///
    /// - `progress = 0.0` → start of schedule
    /// - `progress = 1.0` → end of schedule
    pub fn anneal(&self, progress: f64) -> f64 {
        let progress = progress.clamp(0.0, 1.0);
        match self {
            AnnealingSchedule::Linear { start, end } => start - progress * (start - end),
            AnnealingSchedule::Exponential { start, decay } => {
                start * (-decay * progress).exp()
            }
            AnnealingSchedule::Cosine { start, end } => {
                end + 0.5 * (start - end)
                    * (1.0 + (std::f64::consts::PI * progress).cos())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_probabilities_sum_to_one() {
        let candidates = vec![
            ("a".into(), -2.0),
            ("b".into(), -1.0),
            ("c".into(), 0.0),
        ];
        let probs = softmax_probabilities(&candidates, 1.0);
        let sum: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn lower_efe_gets_higher_probability() {
        // EFE: a=-3 (best), b=-1, c=0 (worst)
        // Since weight = exp(-efe/T), more negative efe → -efe is more positive → higher weight
        let candidates = vec![
            ("a".into(), -3.0),
            ("b".into(), -1.0),
            ("c".into(), 0.0),
        ];
        let probs = softmax_probabilities(&candidates, 1.0);
        let p_a = probs.iter().find(|(id, _)| id == "a").unwrap().1;
        let p_b = probs.iter().find(|(id, _)| id == "b").unwrap().1;
        let p_c = probs.iter().find(|(id, _)| id == "c").unwrap().1;
        assert!(
            p_a > p_b && p_b > p_c,
            "lower EFE should get higher prob: p_a={p_a}, p_b={p_b}, p_c={p_c}"
        );
    }

    #[test]
    fn very_low_temperature_is_nearly_deterministic() {
        let candidates = vec![
            ("best".into(), -10.0),
            ("ok".into(), -1.0),
            ("bad".into(), 5.0),
        ];
        let probs = softmax_probabilities(&candidates, 0.01);
        let p_best = probs.iter().find(|(id, _)| id == "best").unwrap().1;
        assert!(
            p_best > 0.99,
            "at very low temperature, best action should dominate: p_best={p_best}"
        );
    }

    #[test]
    fn high_temperature_gives_uniform_ish_distribution() {
        let candidates = vec![
            ("a".into(), -3.0),
            ("b".into(), -1.0),
            ("c".into(), 0.0),
        ];
        let probs = softmax_probabilities(&candidates, 1000.0);
        // At very high temperature, all probs should be roughly 1/3
        for (_, p) in &probs {
            assert!(
                (*p - 1.0 / 3.0).abs() < 0.01,
                "at high temperature probs should be nearly uniform, got {p}"
            );
        }
    }

    #[test]
    fn empty_candidates_returns_none() {
        let mut rng = rand::thread_rng();
        assert!(select_action(&[], 1.0, &mut rng).is_none());
    }

    #[test]
    fn single_candidate_always_selected() {
        let candidates = vec![("only".into(), -1.0)];
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let selected = select_action(&candidates, 1.0, &mut rng);
            assert_eq!(selected, Some("only".into()));
        }
    }

    #[test]
    fn select_action_returns_valid_candidate() {
        let candidates = vec![
            ("a".into(), -2.0),
            ("b".into(), -1.0),
            ("c".into(), 0.0),
        ];
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let selected = select_action(&candidates, 1.0, &mut rng).unwrap();
            assert!(
                selected == "a" || selected == "b" || selected == "c",
                "unexpected selection: {selected}"
            );
        }
    }

    // --- Annealing schedule tests ---

    #[test]
    fn linear_annealing_start_and_end() {
        let schedule = AnnealingSchedule::Linear {
            start: 1.0,
            end: 0.1,
        };
        assert!((schedule.anneal(0.0) - 1.0).abs() < 1e-10);
        assert!((schedule.anneal(1.0) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn linear_annealing_midpoint() {
        let schedule = AnnealingSchedule::Linear {
            start: 1.0,
            end: 0.0,
        };
        assert!((schedule.anneal(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn exponential_annealing_start() {
        let schedule = AnnealingSchedule::Exponential {
            start: 1.0,
            decay: 3.0,
        };
        assert!((schedule.anneal(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn exponential_annealing_decreases() {
        let schedule = AnnealingSchedule::Exponential {
            start: 1.0,
            decay: 3.0,
        };
        let t0 = schedule.anneal(0.0);
        let t_mid = schedule.anneal(0.5);
        let t_end = schedule.anneal(1.0);
        assert!(t0 > t_mid && t_mid > t_end);
    }

    #[test]
    fn exponential_annealing_end_value() {
        let schedule = AnnealingSchedule::Exponential {
            start: 1.0,
            decay: 3.0,
        };
        let expected = (-3.0_f64).exp(); // e^(-3)
        assert!((schedule.anneal(1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn cosine_annealing_start_and_end() {
        let schedule = AnnealingSchedule::Cosine {
            start: 1.0,
            end: 0.1,
        };
        assert!((schedule.anneal(0.0) - 1.0).abs() < 1e-10);
        assert!((schedule.anneal(1.0) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn cosine_annealing_is_monotone_decreasing() {
        let schedule = AnnealingSchedule::Cosine {
            start: 1.0,
            end: 0.1,
        };
        let mut prev = schedule.anneal(0.0);
        for i in 1..=100 {
            let progress = i as f64 / 100.0;
            let current = schedule.anneal(progress);
            assert!(
                current <= prev + 1e-10,
                "cosine schedule should be non-increasing: prev={prev}, current={current}"
            );
            prev = current;
        }
    }

    #[test]
    fn annealing_clamps_progress() {
        let schedule = AnnealingSchedule::Linear {
            start: 1.0,
            end: 0.0,
        };
        // Out of range progress should be clamped
        assert!((schedule.anneal(-0.5) - 1.0).abs() < 1e-10);
        assert!((schedule.anneal(1.5) - 0.0).abs() < 1e-10);
    }
}
