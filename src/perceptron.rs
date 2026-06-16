//! Averaged-perceptron POS tagger.
//!
//! Implements contextual POS tagging using the averaged-perceptron algorithm
//! originally by Matthew Honnibal / NLTK. Weights are loaded from a
//! JSON file that mirrors the NLTK `averaged_perceptron_tagger` format.
//!
//! Source provenance: weights derived from
//!   NLTK averaged_perceptron_tagger_eng (MIT license)
//!   https://github.com/nltk/nltk_data/blob/gh-pages/packages/taggers/
//!   averaged_perceptron_tagger_eng.zip
//!   SHA-256 of the source weights recorded in scripts/build_model.py.

use std::collections::HashMap;
use std::path::Path;
use serde::Deserialize;

use crate::error::PosError;
use crate::upos::Upos;

pub type Result<T> = std::result::Result<T, PosError>;

/// JSON structure matching the NLTK perceptron weight export format.
#[derive(Deserialize)]
struct WeightsFile {
    /// tag_dict: token → most-frequent-tag (string)
    tags: HashMap<String, String>,
    /// weights: feature → {tag → weight}
    weights: HashMap<String, HashMap<String, f64>>,
    /// classes: sorted list of all tag strings
    classes: Vec<String>,
}

pub struct PerceptronTagger {
    tags: HashMap<String, String>,
    weights: HashMap<String, HashMap<String, f64>>,
    classes: Vec<String>,
    /// Language this tagger was loaded for ("en" or "es").
    lang: String,
}

impl PerceptronTagger {
    /// Load a tagger from a JSON weights file.
    pub fn load(path: &Path, lang: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let wf: WeightsFile = serde_json::from_str(&data)?;
        Ok(Self {
            tags: wf.tags,
            weights: wf.weights,
            classes: wf.classes,
            lang: lang.to_string(),
        })
    }

    /// Tag a sentence (list of lowercase tokens). Returns UPOS for each token.
    pub fn tag(&self, tokens: &[String]) -> Vec<Upos> {
        let mut prev = "-START-".to_string();
        let mut prev2 = "-START2-".to_string();
        let mut out = Vec::with_capacity(tokens.len());

        for (i, word) in tokens.iter().enumerate() {
            let tag = if let Some(t) = self.tags.get(word.as_str()) {
                t.clone()
            } else {
                let features = self.get_features(i, word, tokens, &prev, &prev2);
                self.predict(&features)
            };
            let upos = if self.lang == "en" {
                Upos::from_ptb(&tag)
            } else {
                Upos::from_ud(&tag)
            };
            out.push(upos);
            prev2 = prev;
            prev = tag;
        }
        out
    }

    fn predict(&self, features: &HashMap<String, f64>) -> String {
        let mut scores: HashMap<&str, f64> = HashMap::new();
        for (feat, val) in features {
            if let Some(weights) = self.weights.get(feat.as_str()) {
                for (label, weight) in weights {
                    *scores.entry(label.as_str()).or_insert(0.0) += val * weight;
                }
            }
        }
        self.classes
            .iter()
            .max_by(|a, b| {
                let sa = scores.get(a.as_str()).copied().unwrap_or(0.0);
                let sb = scores.get(b.as_str()).copied().unwrap_or(0.0);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.as_str())
            .unwrap_or("NN")
            .to_string()
    }

    fn get_features(
        &self,
        i: usize,
        word: &str,
        tokens: &[String],
        prev: &str,
        prev2: &str,
    ) -> HashMap<String, f64> {
        let mut f = HashMap::new();

        // Feature keys use space as separator to match NLTK's weight file format:
        // e.g. "i suffix ing", "i-1 tag+i word VBD running"
        macro_rules! add {
            ($name:expr, $val:expr) => {
                let key = format!("{} {}", $name, $val);
                *f.entry(key).or_insert(0.0) += 1.0;
            };
            ($name:expr, $v1:expr, $v2:expr) => {
                let key = format!("{} {} {}", $name, $v1, $v2);
                *f.entry(key).or_insert(0.0) += 1.0;
            };
        }

        let suf = |w: &str, n: usize| {
            let chars: Vec<char> = w.chars().collect();
            let start = chars.len().saturating_sub(n);
            chars[start..].iter().collect::<String>()
        };
        let pre = |w: &str, n: usize| {
            w.chars().take(n).collect::<String>()
        };
        let word_i = |idx: isize| -> &str {
            if idx < 0 {
                return match idx {
                    -1 => "-START-",
                    _ => "-START2-",
                };
            }
            let uidx = idx as usize;
            if uidx >= tokens.len() { "-END-" } else { tokens[uidx].as_str() }
        };

        *f.entry("bias".to_string()).or_insert(0.0) += 1.0;
        add!("i suffix", &suf(word, 3));
        add!("i pref1", &pre(word, 1));
        add!("i-1 tag", prev);
        add!("i-2 tag", prev2);
        add!("i tag+i-2 tag", prev, prev2);
        add!("i word", word);
        add!("i-1 tag+i word", prev, word);
        add!("i-1 word", word_i(i as isize - 1));
        add!("i-1 suffix", &suf(word_i(i as isize - 1), 3));
        add!("i-2 word", word_i(i as isize - 2));
        add!("i+1 word", word_i(i as isize + 1));
        add!("i+1 suffix", &suf(word_i(i as isize + 1), 3));
        add!("i+2 word", word_i(i as isize + 2));

        f
    }
}
