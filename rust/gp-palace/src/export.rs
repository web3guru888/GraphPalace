//! Palace export/import for persistence and migration.

use gp_storage::memory::PalaceData;
use serde::{Deserialize, Serialize};

/// How to handle conflicts during import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImportMode {
    /// Delete existing data and replace with imported data.
    Replace,
    /// Keep existing data; only add items whose IDs don't already exist.
    Merge,
    /// Import everything, overwriting existing items with the same ID.
    Overlay,
}

/// Statistics from an import operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportStats {
    pub wings_added: usize,
    pub rooms_added: usize,
    pub closets_added: usize,
    pub drawers_added: usize,
    pub entities_added: usize,
    pub relationships_added: usize,
    pub duplicates_skipped: usize,
}

/// A full, serializable snapshot of a palace's data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceExport {
    /// Format version for forward compatibility.
    pub version: u32,
    /// Timestamp of the export.
    pub exported_at: chrono::DateTime<chrono::Utc>,
    /// The raw palace data.
    pub data: PalaceData,
}

impl PalaceExport {
    /// Serialize this export to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize this export to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a palace export from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn import_mode_serialization() {
        let mode = ImportMode::Replace;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"replace\"");

        let mode2: ImportMode = serde_json::from_str("\"merge\"").unwrap();
        assert_eq!(mode2, ImportMode::Merge);

        let mode3: ImportMode = serde_json::from_str("\"overlay\"").unwrap();
        assert_eq!(mode3, ImportMode::Overlay);
    }

    #[test]
    fn import_stats_default() {
        let stats = ImportStats::default();
        assert_eq!(stats.wings_added, 0);
        assert_eq!(stats.drawers_added, 0);
        assert_eq!(stats.duplicates_skipped, 0);
    }

    #[test]
    fn import_stats_serialization() {
        let stats = ImportStats {
            wings_added: 2,
            rooms_added: 5,
            closets_added: 10,
            drawers_added: 20,
            entities_added: 3,
            relationships_added: 4,
            duplicates_skipped: 1,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deser: ImportStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.wings_added, 2);
        assert_eq!(deser.drawers_added, 20);
    }

    #[test]
    fn palace_export_round_trip() {
        let data = PalaceData::default();
        let export = PalaceExport {
            version: 1,
            exported_at: Utc::now(),
            data,
        };
        let json = export.to_json().unwrap();
        let deser = PalaceExport::from_json(&json).unwrap();
        assert_eq!(deser.version, 1);
        assert!(deser.data.wings.is_empty());
    }

    #[test]
    fn palace_export_pretty_json() {
        let data = PalaceData::default();
        let export = PalaceExport {
            version: 1,
            exported_at: Utc::now(),
            data,
        };
        let json = export.to_json_pretty().unwrap();
        assert!(json.contains('\n'));
    }

    #[test]
    fn palace_export_with_data() {
        use gp_core::types::*;

        let mut data = PalaceData::default();
        data.wings.insert(
            "wing_1".into(),
            Wing {
                id: "wing_1".into(),
                name: "Science".into(),
                wing_type: WingType::Domain,
                description: "Science wing".into(),
                embedding: [0.0; 384],
                pheromones: NodePheromones::default(),
                created_at: Utc::now(),
            },
        );
        let export = PalaceExport {
            version: 1,
            exported_at: Utc::now(),
            data,
        };
        let json = export.to_json().unwrap();
        let deser = PalaceExport::from_json(&json).unwrap();
        assert_eq!(deser.data.wings.len(), 1);
        assert_eq!(deser.data.wings["wing_1"].name, "Science");
    }
}
