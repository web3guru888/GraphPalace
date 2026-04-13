//! Periodic pheromone decay scheduling.
//!
//! Manages when decay operations should be applied during swarm operation.
//! Default: every 10 cycles (from spec §4.2).

use gp_core::config::PheromoneConfig;
use serde::{Deserialize, Serialize};

/// Tracks decay scheduling state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayScheduler {
    /// How many cycles between decay operations.
    pub interval: usize,
    /// Total number of decay operations performed.
    pub decay_count: usize,
    /// Cycle number of the last decay.
    pub last_decay_cycle: Option<usize>,
    /// Whether decay is enabled.
    pub enabled: bool,
}

impl DecayScheduler {
    /// Create a new scheduler with the given interval.
    pub fn new(interval: usize) -> Self {
        Self {
            interval,
            decay_count: 0,
            last_decay_cycle: None,
            enabled: true,
        }
    }

    /// Create from pheromone config.
    pub fn from_config(config: &PheromoneConfig) -> Self {
        Self::new(config.decay_interval_cycles)
    }

    /// Check if decay should be applied at the given cycle.
    pub fn should_decay(&self, cycle: usize) -> bool {
        if !self.enabled {
            return false;
        }
        if cycle == 0 {
            return false; // Never decay on first cycle
        }
        cycle.is_multiple_of(self.interval)
    }

    /// Record that a decay operation was performed.
    pub fn record_decay(&mut self, cycle: usize) {
        self.decay_count += 1;
        self.last_decay_cycle = Some(cycle);
    }

    /// Check and record: returns true if decay should happen, and records it.
    pub fn tick(&mut self, cycle: usize) -> bool {
        if self.should_decay(cycle) {
            self.record_decay(cycle);
            true
        } else {
            false
        }
    }

    /// Cycles since last decay (or None if never decayed).
    pub fn cycles_since_decay(&self, current_cycle: usize) -> Option<usize> {
        self.last_decay_cycle.map(|last| current_cycle - last)
    }

    /// Enable or disable the scheduler.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Reset the scheduler state.
    pub fn reset(&mut self) {
        self.decay_count = 0;
        self.last_decay_cycle = None;
    }
}

impl Default for DecayScheduler {
    fn default() -> Self {
        Self::new(10) // Default from spec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_interval_is_ten() {
        let s = DecayScheduler::default();
        assert_eq!(s.interval, 10);
        assert!(s.enabled);
        assert_eq!(s.decay_count, 0);
        assert!(s.last_decay_cycle.is_none());
    }

    #[test]
    fn from_config() {
        let config = PheromoneConfig {
            decay_interval_cycles: 5,
            ..PheromoneConfig::default()
        };
        let s = DecayScheduler::from_config(&config);
        assert_eq!(s.interval, 5);
    }

    #[test]
    fn no_decay_at_cycle_zero() {
        let s = DecayScheduler::new(10);
        assert!(!s.should_decay(0));
    }

    #[test]
    fn decay_at_interval() {
        let s = DecayScheduler::new(10);
        assert!(!s.should_decay(1));
        assert!(!s.should_decay(5));
        assert!(!s.should_decay(9));
        assert!(s.should_decay(10));
        assert!(!s.should_decay(11));
        assert!(s.should_decay(20));
        assert!(s.should_decay(30));
    }

    #[test]
    fn disabled_scheduler_never_decays() {
        let mut s = DecayScheduler::new(10);
        s.set_enabled(false);
        assert!(!s.should_decay(10));
        assert!(!s.should_decay(20));
    }

    #[test]
    fn record_decay_increments_count() {
        let mut s = DecayScheduler::new(10);
        s.record_decay(10);
        assert_eq!(s.decay_count, 1);
        assert_eq!(s.last_decay_cycle, Some(10));
        s.record_decay(20);
        assert_eq!(s.decay_count, 2);
        assert_eq!(s.last_decay_cycle, Some(20));
    }

    #[test]
    fn tick_combines_check_and_record() {
        let mut s = DecayScheduler::new(5);
        assert!(!s.tick(0));
        assert!(!s.tick(1));
        assert!(!s.tick(4));
        assert!(s.tick(5));
        assert_eq!(s.decay_count, 1);
        assert!(!s.tick(6));
        assert!(s.tick(10));
        assert_eq!(s.decay_count, 2);
    }

    #[test]
    fn cycles_since_decay() {
        let mut s = DecayScheduler::new(10);
        assert!(s.cycles_since_decay(5).is_none());
        s.record_decay(10);
        assert_eq!(s.cycles_since_decay(15), Some(5));
        assert_eq!(s.cycles_since_decay(10), Some(0));
    }

    #[test]
    fn reset_clears_state() {
        let mut s = DecayScheduler::new(10);
        s.record_decay(10);
        s.record_decay(20);
        s.reset();
        assert_eq!(s.decay_count, 0);
        assert!(s.last_decay_cycle.is_none());
        assert!(s.enabled); // enabled is preserved
        assert_eq!(s.interval, 10); // interval preserved
    }

    #[test]
    fn interval_one_decays_every_cycle() {
        let s = DecayScheduler::new(1);
        assert!(!s.should_decay(0));
        assert!(s.should_decay(1));
        assert!(s.should_decay(2));
        assert!(s.should_decay(100));
    }

    #[test]
    fn serialization_roundtrip() {
        let mut s = DecayScheduler::new(7);
        s.record_decay(7);
        s.record_decay(14);
        let json = serde_json::to_string(&s).unwrap();
        let s2: DecayScheduler = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.interval, 7);
        assert_eq!(s2.decay_count, 2);
        assert_eq!(s2.last_decay_cycle, Some(14));
    }
}
