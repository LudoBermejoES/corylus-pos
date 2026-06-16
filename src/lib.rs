//! POS tagging engine for Corylus.
//!
//! Mirrors the rust-rhyme architecture:
//!   - English: a downloaded averaged-perceptron model (SHA-256-pinned .tar.gz
//!     on the corylus-pos GitHub Release), verified, unpacked under the app-data
//!     dir, held in memory.
//!   - Spanish: heuristic only until the AnCora model ships; adjectives require
//!     the model; adverbs and gerunds use rule-based suffix matching.
//!
//! Two strategies behind one `tag_batch` API:
//!   1. Model-free suffix heuristic: adverbs (-ly / -mente) and gerunds
//!      (-ing / -ando / -iendo / -yendo) with exception lists. Works at first
//!      launch with no download.
//!   2. Contextual averaged-perceptron: loads JSON weights, tags with sentence
//!      context, normalises to UPOS. Needed for adjectives and higher-accuracy ADV.

mod error;
mod heuristic;
mod perceptron;
mod provision;
mod state;
mod tokenizer;
mod upos;

#[cfg(test)]
mod tests;

pub use error::PosError;
pub use heuristic::{is_en_adverb, is_en_gerund, is_es_adverb, is_es_gerund};
pub use tokenizer::{split_sentences, split_sentences_surface};
pub use upos::Upos;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub type Result<T> = std::result::Result<T, PosError>;

/// Configuration for one language's POS engine.
#[derive(Clone)]
pub struct EngineConfig {
    /// Directory where {lang}.pos.json and {lang}.pos.version.json live.
    pub data_dir: PathBuf,
    /// Language code: "en" or "es".
    pub lang: String,
    /// URL of the pinned gzipped tar artifact. Empty for rule-based (ES heuristic-only).
    pub source_url: String,
    /// Pinned SHA-256 hex string of the artifact. Empty for rule-based.
    pub source_sha256: String,
}

impl EngineConfig {
    pub fn default_en(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            lang: "en".into(),
            // Placeholder: filled in once the artifact is published (task 2.6).
            source_url: String::new(),
            source_sha256: String::new(),
        }
    }

    pub fn default_es(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            lang: "es".into(),
            // Heuristic-only until AnCora model is published (task 4.5).
            source_url: String::new(),
            source_sha256: String::new(),
        }
    }
}

/// Observable state of the POS engine for one language.
#[derive(Clone, Debug, PartialEq)]
pub enum PosState {
    NotInstalled,
    Downloading { downloaded: u64, total: Option<u64> },
    Indexing,
    Ready,
    Error { message: String },
}

pub(crate) struct Inner {
    pub config: EngineConfig,
    pub state: PosState,
    /// The loaded perceptron tagger; None until Ready or for heuristic-only configs.
    pub tagger: Option<perceptron::PerceptronTagger>,
}

/// Per-language POS engine. Cheap to clone (Arc-backed).
#[derive(Clone)]
pub struct PosEngine(Arc<Mutex<Inner>>);

impl PosEngine {
    pub fn new(config: EngineConfig) -> Self {
        let is_heuristic_only = config.source_url.is_empty();
        let initial_state = if is_heuristic_only {
            // Heuristics are always ready.
            PosState::Ready
        } else {
            // Check if model is already installed on disk.
            if state::is_installed_for(&config) {
                PosState::NotInstalled // will be upgraded to Ready in set_data_dir
            } else {
                PosState::NotInstalled
            }
        };
        Self(Arc::new(Mutex::new(Inner {
            config,
            state: initial_state,
            tagger: None,
        })))
    }

    /// Update the data directory. Called from `setup()` once the app data dir is known.
    /// For heuristic-only languages, this is a no-op state update. For model-backed
    /// languages, it also checks if the model is already installed.
    pub fn set_data_dir(&self, data_dir: PathBuf) {
        let mut inner = self.0.lock().unwrap();
        inner.config.data_dir = data_dir;
        if inner.config.source_url.is_empty() {
            // Heuristic-only: always Ready.
            inner.state = PosState::Ready;
            return;
        }
        if state::is_installed_for(&inner.config) {
            drop(inner);
            let _ = provision::try_load_model(self.0.clone());
        }
    }

    pub fn state(&self) -> PosState {
        self.0.lock().unwrap().state.clone()
    }

    pub fn data_dir(&self) -> PathBuf {
        self.0.lock().unwrap().config.data_dir.clone()
    }

    pub fn is_heuristic_only(&self) -> bool {
        self.0.lock().unwrap().config.source_url.is_empty()
    }

    /// Provision the model (no-op for heuristic-only configs).
    pub async fn provision(
        &self,
        on_progress: impl Fn(PosState) + Send + 'static,
    ) -> Result<()> {
        if self.is_heuristic_only() {
            on_progress(PosState::Ready);
            return Ok(());
        }
        provision::run(self.0.clone(), on_progress).await
    }

    /// Tag a batch of sentences. Returns one `Vec<TagResult>` per sentence.
    ///
    /// Each `TagResult` carries both the UPOS tag from the perceptron (if available)
    /// and the heuristic flags (adverb, gerund) that work without a model.
    ///
    /// - Adverb heuristic runs regardless of model availability.
    /// - Gerund heuristic runs regardless of model availability.
    /// - UPOS from the model is `None` when the model is not installed.
    pub fn tag_batch(&self, sentences: &[Vec<String>]) -> Vec<Vec<TagResult>> {
        let inner = self.0.lock().unwrap();
        let lang = inner.config.lang.clone();
        let tagger = inner.tagger.as_ref().map(|_| ());
        // We need to drop the lock before calling tagger methods; extract tagger ref carefully.
        drop(inner);

        sentences.iter().map(|sent| {
            // Re-acquire lock only for the tagger call.
            let upos_tags: Option<Vec<Upos>> = {
                let inner = self.0.lock().unwrap();
                inner.tagger.as_ref().map(|t| t.tag(sent))
            };

            sent.iter().enumerate().map(|(i, word)| {
                let upos = upos_tags.as_ref().map(|tags| tags[i]);
                let (is_adverb, is_gerund) = match lang.as_str() {
                    "es" => (is_es_adverb(word), is_es_gerund(word)),
                    _ => (is_en_adverb(word), is_en_gerund(word)),
                };
                // Heuristic gerund/adverb flags take precedence when model absent.
                let effective_upos = upos.or_else(|| {
                    if is_adverb { Some(Upos::Adv) }
                    else { None }
                });
                let _ = tagger; // suppress unused warning
                TagResult {
                    word: word.clone(),
                    upos: effective_upos,
                    is_adverb_heuristic: is_adverb,
                    is_gerund: is_gerund,
                }
            }).collect()
        }).collect()
    }

    /// Remove the installed model and reset to NotInstalled. No-op for heuristic-only.
    pub fn uninstall(&self) -> Result<()> {
        let mut inner = self.0.lock().unwrap();
        if inner.config.source_url.is_empty() {
            return Ok(());
        }
        let config = &inner.config;
        for path in [state::weights_path(config), state::version_path(config)] {
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
        }
        inner.tagger = None;
        inner.state = PosState::NotInstalled;
        Ok(())
    }
}

/// Result for a single token from `tag_batch`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TagResult {
    pub word: String,
    /// UPOS from the perceptron model. None when model not installed.
    pub upos: Option<Upos>,
    /// True when the adverb suffix heuristic fires (no model needed).
    pub is_adverb_heuristic: bool,
    /// True when the gerund suffix heuristic fires (no model needed).
    pub is_gerund: bool,
}
