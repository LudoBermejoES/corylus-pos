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

    /// Tag a sentence of surface tokens. Returns UPOS for each token.
    pub fn tag(&self, surface: &[String]) -> Vec<Upos> {
        let es = self.lang == "es";
        // The Spanish model was trained on lowercased AnCora forms, so a
        // capitalised "Ella"/"La" is an unknown word to it unless lowercased.
        // English (NLTK) weights are cased and take the surface form.
        let lowered: Vec<String>;
        let tokens: &[String] = if es {
            lowered = surface.iter().map(|w| w.to_lowercase()).collect();
            &lowered
        } else {
            surface
        };
        let mut prev = "-START-".to_string();
        let mut prev2 = "-START2-".to_string();
        let mut out = Vec::with_capacity(tokens.len());

        for (i, word) in tokens.iter().enumerate() {
            // AnCora splits "del"/"al" into "de el"/"a el", so the model never
            // saw them: tag the preposition and give the next word the DET
            // context it was trained with.
            if es && (word == "del" || word == "al") {
                out.push(Upos::Adp);
                prev2 = "ADP".to_string();
                prev = "DET".to_string();
                continue;
            }
            // Lowercasing hid proper nouns ("Casa Blanca", "Miró") among common
            // words in training; a lowercase word in running Spanish text is
            // never a proper noun, so PROPN is ruled out for it.
            let no_propn = es && surface[i].chars().next().is_some_and(|c| c.is_lowercase());
            let tag = match self.tags.get(word.as_str()) {
                Some(t) if !(no_propn && t == "PROPN") => t.clone(),
                _ => {
                    let features = self.get_features(i, word, tokens, &prev, &prev2);
                    self.predict(&features, if no_propn { Some("PROPN") } else { None })
                }
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

    fn predict(&self, features: &HashMap<String, f64>, exclude: Option<&str>) -> String {
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
            .filter(|c| Some(c.as_str()) != exclude)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny Spanish model shaped like the AnCora one: trained on lowercase
    /// forms, with "casa" after a determiner leaning PROPN ("la Casa Blanca").
    fn es_tagger() -> PerceptronTagger {
        let json = r#"{
            "tags": {"ella": "PRON", "la": "DET", "el": "DET", "miró": "PROPN"},
            "weights": {
                "bias": {"PROPN": 1.0},
                "i-1 tag DET": {"NOUN": 2.0},
                "i-1 tag+i word DET casa": {"PROPN": 5.0},
                "i-1 tag NOUN": {"ADJ": 3.0},
                "i-1 tag PROPN": {"PROPN": 4.0},
                "i-1 tag PRON": {"VERB": 3.0},
                "i word mar": {"NOUN": 3.0}
            },
            "classes": ["ADJ", "ADP", "DET", "NOUN", "PRON", "PROPN", "VERB"]
        }"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("es.pos.json");
        std::fs::write(&path, json).unwrap();
        PerceptronTagger::load(&path, "es").unwrap()
    }

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn es_capitalised_sentence_start_is_looked_up_lowercase() {
        // "Ella" used to miss the lowercase tag dictionary and come out PROPN.
        let tags = es_tagger().tag(&words("Ella miró el mar"));
        assert_eq!(tags, vec![Upos::Pron, Upos::Verb, Upos::Det, Upos::Noun]);
    }

    #[test]
    fn es_lowercase_word_is_never_a_proper_noun() {
        // "casa" leaned PROPN and dragged "blanca" to PROPN with it.
        let tags = es_tagger().tag(&words("La casa blanca"));
        assert_eq!(tags, vec![Upos::Det, Upos::Noun, Upos::Adj]);
    }

    #[test]
    fn es_capitalised_name_can_still_be_a_proper_noun() {
        let tags = es_tagger().tag(&words("Tomás"));
        assert_eq!(tags, vec![Upos::Noun]); // PROPN maps to NOUN
    }

    #[test]
    fn es_contractions_are_prepositions() {
        let tags = es_tagger().tag(&words("la puerta del faro"));
        assert_eq!(tags[2], Upos::Adp);
        assert_eq!(tags[3], Upos::Noun, "word after \"del\" gets DET context");
    }
}
