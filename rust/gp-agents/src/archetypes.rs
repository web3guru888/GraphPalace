//! Agent archetype definitions from spec §6.6.
//!
//! Archetypes provide default configurations for agents with different
//! exploration/exploitation strategies and domain specialisations.

use serde::{Deserialize, Serialize};

/// Predefined agent archetypes with characteristic behaviours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentArchetype {
    /// High temperature (1.0): pure epistemic exploration, no goal bias.
    Explorer,
    /// Very low temperature (0.1): aggressively exploits known good paths.
    Exploiter,
    /// Low temperature (0.3): narrow domain focus, deep expertise.
    Specialist,
    /// Moderate temperature (0.5): broad search across many domains.
    Generalist,
    /// Low temperature (0.2): detects contradictions and inconsistencies.
    Critic,
}

impl AgentArchetype {
    /// Default temperature for this archetype.
    pub fn default_temperature(&self) -> f64 {
        match self {
            AgentArchetype::Explorer => 1.0,
            AgentArchetype::Exploiter => 0.1,
            AgentArchetype::Specialist => 0.3,
            AgentArchetype::Generalist => 0.5,
            AgentArchetype::Critic => 0.2,
        }
    }

    /// Human-readable description of the archetype's behaviour.
    pub fn description(&self) -> &'static str {
        match self {
            AgentArchetype::Explorer => {
                "High-temperature explorer: maximises epistemic value, no goal bias"
            }
            AgentArchetype::Exploiter => {
                "Low-temperature exploiter: aggressively follows known rewarding paths"
            }
            AgentArchetype::Specialist => {
                "Narrow-focus specialist: deep expertise in a single domain"
            }
            AgentArchetype::Generalist => {
                "Broad-search generalist: balanced exploration across domains"
            }
            AgentArchetype::Critic => {
                "Contradiction detector: identifies inconsistencies in the knowledge graph"
            }
        }
    }

    /// List of all archetypes.
    pub fn all() -> &'static [AgentArchetype] {
        &[
            AgentArchetype::Explorer,
            AgentArchetype::Exploiter,
            AgentArchetype::Specialist,
            AgentArchetype::Generalist,
            AgentArchetype::Critic,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_archetypes_have_positive_temperatures() {
        for archetype in AgentArchetype::all() {
            let temp = archetype.default_temperature();
            assert!(
                temp > 0.0,
                "{archetype:?} has non-positive temperature: {temp}"
            );
        }
    }

    #[test]
    fn explorer_has_highest_temperature() {
        let explorer_temp = AgentArchetype::Explorer.default_temperature();
        for archetype in AgentArchetype::all() {
            let temp = archetype.default_temperature();
            assert!(
                temp <= explorer_temp,
                "{archetype:?} (temp={temp}) exceeds Explorer (temp={explorer_temp})"
            );
        }
    }

    #[test]
    fn exploiter_has_lowest_temperature() {
        let exploiter_temp = AgentArchetype::Exploiter.default_temperature();
        for archetype in AgentArchetype::all() {
            let temp = archetype.default_temperature();
            assert!(
                temp >= exploiter_temp,
                "{archetype:?} (temp={temp}) is lower than Exploiter (temp={exploiter_temp})"
            );
        }
    }

    #[test]
    fn all_archetypes_have_descriptions() {
        for archetype in AgentArchetype::all() {
            let desc = archetype.description();
            assert!(
                !desc.is_empty(),
                "{archetype:?} has empty description"
            );
        }
    }

    #[test]
    fn temperature_ordering() {
        // Exploiter < Critic < Specialist < Generalist < Explorer
        assert!(
            AgentArchetype::Exploiter.default_temperature()
                < AgentArchetype::Critic.default_temperature()
        );
        assert!(
            AgentArchetype::Critic.default_temperature()
                < AgentArchetype::Specialist.default_temperature()
        );
        assert!(
            AgentArchetype::Specialist.default_temperature()
                < AgentArchetype::Generalist.default_temperature()
        );
        assert!(
            AgentArchetype::Generalist.default_temperature()
                < AgentArchetype::Explorer.default_temperature()
        );
    }

    #[test]
    fn serialization_roundtrip() {
        for archetype in AgentArchetype::all() {
            let json = serde_json::to_string(archetype).unwrap();
            let deserialized: AgentArchetype = serde_json::from_str(&json).unwrap();
            assert_eq!(*archetype, deserialized);
        }
    }

    #[test]
    fn all_returns_five_archetypes() {
        assert_eq!(AgentArchetype::all().len(), 5);
    }
}
