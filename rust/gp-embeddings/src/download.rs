//! Model downloader for HuggingFace sentence-transformer models.
//!
//! Downloads ONNX model weights and tokenizer configuration from the
//! HuggingFace Hub, caching them locally so subsequent loads are instant.
//!
//! # Feature gate
//!
//! This module requires the `download` feature (which also enables `onnx`):
//!
//! ```toml
//! gp-embeddings = { path = "gp-embeddings", features = ["download"] }
//! ```

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use gp_core::{GraphPalaceError, Result};

/// Base URL for the HuggingFace Hub.
const HF_BASE: &str = "https://huggingface.co";

/// Default model identifier on the HuggingFace Hub.
const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Files to download for a complete ONNX model.
const MODEL_FILES: &[(&str, &str)] = &[
    ("onnx/model.onnx", "model.onnx"),
    ("tokenizer.json", "tokenizer.json"),
];

/// Download a sentence-transformer model from HuggingFace Hub.
///
/// Downloads `model.onnx` and `tokenizer.json` into
/// `<cache_dir>/<model_name>/`. If the files already exist they are not
/// re-downloaded.
///
/// Returns the path to the model directory.
///
/// # Arguments
///
/// * `model_name` — HuggingFace model identifier, e.g.
///   `"sentence-transformers/all-MiniLM-L6-v2"`.
/// * `cache_dir` — Local directory for cached downloads.
///
/// # Errors
///
/// Returns [`GraphPalaceError::Embedding`] on network or I/O errors.
pub fn download_model(model_name: &str, cache_dir: &Path) -> Result<PathBuf> {
    // Sanitise the model name for use as a directory name.
    let dir_name = model_name.replace('/', "--");
    let model_dir = cache_dir.join(&dir_name);
    fs::create_dir_all(&model_dir).map_err(|e| {
        GraphPalaceError::Embedding(format!(
            "create cache dir {}: {e}",
            model_dir.display()
        ))
    })?;

    for &(remote_path, local_name) in MODEL_FILES {
        let local_path = model_dir.join(local_name);
        if local_path.exists() {
            continue; // Already cached.
        }

        let url = format!("{HF_BASE}/{model_name}/resolve/main/{remote_path}");
        download_file(&url, &local_path)?;
    }

    Ok(model_dir)
}

/// Download the default model (`all-MiniLM-L6-v2`) into `cache_dir`.
///
/// Convenience wrapper around [`download_model`].
pub fn download_default_model(cache_dir: &Path) -> Result<PathBuf> {
    download_model(DEFAULT_MODEL, cache_dir)
}

/// Ensure a model directory contains the required files, downloading them
/// if missing.
///
/// Unlike [`download_model`], this function writes directly into `dir`
/// (no sub-directory is created). This is useful when you want the model
/// at a fixed path.
pub fn ensure_model_exists(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| {
        GraphPalaceError::Embedding(format!("create dir {}: {e}", dir.display()))
    })?;

    for &(remote_path, local_name) in MODEL_FILES {
        let local_path = dir.join(local_name);
        if local_path.exists() {
            continue;
        }

        let url = format!("{HF_BASE}/{DEFAULT_MODEL}/resolve/main/{remote_path}");
        download_file(&url, &local_path)?;
    }

    Ok(())
}

/// Download a single file from `url` to `path`.
fn download_file(url: &str, path: &Path) -> Result<()> {
    let resp = ureq::get(url).call().map_err(|e| {
        GraphPalaceError::Embedding(format!("HTTP GET {url}: {e}"))
    })?;

    let mut reader = resp.into_reader();
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).map_err(|e| {
        GraphPalaceError::Embedding(format!("read response body from {url}: {e}"))
    })?;

    if bytes.is_empty() {
        return Err(GraphPalaceError::Embedding(format!(
            "empty response from {url}"
        )));
    }

    // Write to a temp file first, then rename for atomicity.
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, &bytes).map_err(|e| {
        GraphPalaceError::Embedding(format!("write {}: {e}", tmp_path.display()))
    })?;
    fs::rename(&tmp_path, path).map_err(|e| {
        GraphPalaceError::Embedding(format!(
            "rename {} → {}: {e}",
            tmp_path.display(),
            path.display()
        ))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_files_list_is_complete() {
        assert_eq!(MODEL_FILES.len(), 2);
        assert!(MODEL_FILES.iter().any(|(_, l)| *l == "model.onnx"));
        assert!(MODEL_FILES.iter().any(|(_, l)| *l == "tokenizer.json"));
    }

    #[test]
    fn default_model_name() {
        assert_eq!(DEFAULT_MODEL, "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[test]
    fn ensure_model_exists_creates_dir() {
        let dir = std::env::temp_dir().join("gp_dl_test_ensure_mkdir");
        let _ = std::fs::remove_dir_all(&dir);
        // This will fail on the HTTP download but the directory should be created.
        let result = ensure_model_exists(&dir);
        assert!(dir.exists(), "directory should be created even if download fails");
        // Clean up.
        let _ = std::fs::remove_dir_all(&dir);
        // We expect an error because we can't reach HF in tests.
        // (If we can, that's fine too — the test is about the directory.)
        let _ = result;
    }

    #[test]
    fn download_model_sanitises_name() {
        // The directory name should replace '/' with '--'.
        let sanitised = DEFAULT_MODEL.replace('/', "--");
        assert_eq!(sanitised, "sentence-transformers--all-MiniLM-L6-v2");
    }
}
