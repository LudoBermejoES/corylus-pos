//! Universal POS tags (UPOS) — the common tag set exposed to consumers.
//!
//! Both Penn Treebank (NLTK English) and UD/AnCora (Spanish) tags are normalised
//! to this enum at the engine boundary. Consumers depend only on UPOS.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Upos {
    Adj,   // adjective
    Adv,   // adverb
    Noun,  // noun
    Verb,  // verb
    Pron,  // pronoun
    Det,   // determiner
    Adp,   // adposition (preposition/postposition)
    Aux,   // auxiliary verb
    Cconj, // coordinating conjunction
    Sconj, // subordinating conjunction
    Num,   // numeral
    Part,  // particle
    Intj,  // interjection
    Punct, // punctuation
    Sym,   // symbol
    X,     // other/unknown
    /// Gerund/present-participle — a morphological verb-form flag surfaced
    /// independently of the main UPOS tag because Universal POS has no
    /// GERUND category. A word tagged Gerund may also be ADV, NOUN, or VERB
    /// depending on context; consumers act on the Gerund flag separately.
    Gerund,
}

impl Upos {
    /// Map a Penn Treebank tag string to UPOS.
    pub fn from_ptb(tag: &str) -> Self {
        match tag {
            "JJ" | "JJR" | "JJS" => Self::Adj,
            "RB" | "RBR" | "RBS" | "WRB" => Self::Adv,
            "NN" | "NNS" | "NNP" | "NNPS" => Self::Noun,
            "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" => Self::Verb,
            "PRP" | "PRP$" | "WP" | "WP$" => Self::Pron,
            "DT" | "WDT" | "PDT" => Self::Det,
            "IN" => Self::Adp,
            "MD" => Self::Aux,
            "CC" => Self::Cconj,
            "RP" | "TO" => Self::Part,
            "UH" => Self::Intj,
            "CD" => Self::Num,
            "." | "," | ":" | "``" | "''" | "-LRB-" | "-RRB-" => Self::Punct,
            "$" | "SYM" => Self::Sym,
            _ => Self::X,
        }
    }

    /// Map a Universal Dependencies tag string to UPOS (used for Spanish AnCora).
    pub fn from_ud(tag: &str) -> Self {
        match tag {
            "ADJ" => Self::Adj,
            "ADV" => Self::Adv,
            "NOUN" | "PROPN" => Self::Noun,
            "VERB" => Self::Verb,
            "PRON" => Self::Pron,
            "DET" => Self::Det,
            "ADP" => Self::Adp,
            "AUX" => Self::Aux,
            "CCONJ" => Self::Cconj,
            "SCONJ" => Self::Sconj,
            "NUM" => Self::Num,
            "PART" => Self::Part,
            "INTJ" => Self::Intj,
            "PUNCT" => Self::Punct,
            "SYM" => Self::Sym,
            _ => Self::X,
        }
    }
}
