//! Model-free suffix heuristics: adverb and gerund detection.
//!
//! These work on first launch with no download — the perceptron model is only
//! needed for adjectives (and for higher-accuracy ADV tagging beyond the heuristic).

// ---- English adverb: -ly ----

static EN_ADV_EXCEPTIONS: &[&str] = &[
    "only", "family", "friendly", "early", "likely", "lonely", "lively",
    "lovely", "ugly", "silly", "holy", "belly", "bully", "curly", "burly",
    "gnarly", "surly", "twirly", "pearly", "barely", "fairly",
    "nearly", "really", "yearly",
    // common proper nouns / given names
    "nelly", "molly", "polly", "holly", "lily", "emily",
    // other false positives
    "rally", "bally", "dally", "tally", "sally", "valley",
];

pub fn is_en_adverb(word: &str) -> bool {
    let lower = word.to_lowercase();
    if lower.len() < 4 {
        return false;
    }
    if !lower.ends_with("ly") {
        return false;
    }
    if EN_ADV_EXCEPTIONS.contains(&lower.as_str()) {
        return false;
    }
    true
}

// ---- Spanish adverb: -mente ----

static ES_ADV_EXCEPTIONS: &[&str] = &[
    // false positives: words ending in -mente that are not adverbs
    "umente", "lamente", "clemente", "elemento",
];

pub fn is_es_adverb(word: &str) -> bool {
    let lower = word.to_lowercase();
    if lower.len() < 7 {
        return false;
    }
    if !lower.ends_with("mente") {
        return false;
    }
    if ES_ADV_EXCEPTIONS.iter().any(|e| lower == *e) {
        return false;
    }
    true
}

// ---- English gerund: -ing ----

static EN_GERUND_EXCEPTIONS: &[&str] = &[
    "thing", "king", "ring", "sing", "wing", "spring", "string", "bring",
    "during", "ceiling", "evening", "morning", "nothing", "something",
    "anything", "everything", "viking", "sibling", "pudding", "wedding",
    "offspring", "icing", "banking", "clothing", "funding", "housing",
    "farming", "booking", "camping", "coding", "cycling", "duckling",
    "ducking", "sting", "swing", "cling", "fling", "sling", "ding", "ping",
    "fitting", "heading", "shipping", "shopping", "sitting", "standing",
    // common nouns that happen to end in -ing
    "bearing", "building", "ceiling", "clearing", "coating", "covering",
    "crossing", "drawing", "finding", "flooring", "following", "gathering",
    "greeting", "handling", "hearing", "holding", "keeping", "landing",
    "leading", "learning", "leaving", "listing", "meaning", "meeting",
    "opening", "painting", "parking", "passing", "planning", "playing",
    "printing", "reading", "recording", "remaining", "roofing", "saving",
    "saying", "serving", "setting", "sharing", "siding", "siding",
    "skiing", "spelling", "storing", "stuffing", "surrounding", "training",
    "understanding", "warning", "washing", "welding", "wiring", "working",
    "writing",
];

pub fn is_en_gerund(word: &str) -> bool {
    let lower = word.to_lowercase();
    if lower.len() < 5 {
        return false;
    }
    if !lower.ends_with("ing") {
        return false;
    }
    if EN_GERUND_EXCEPTIONS.contains(&lower.as_str()) {
        return false;
    }
    true
}

// ---- Spanish gerund: -ando / -iendo / -yendo ----

static ES_GERUND_EXCEPTIONS: &[&str] = &[
    "bando", "blando", "comando", "mando", "cuando", "segundo", "mundo",
    "fondo", "redondo", "profundo", "rotundo", "Orlando", "Fernando",
    "Orlando", "Alejandro",
    // -iendo exceptions
    "cliente",
    // -yendo exceptions (very rare)
];

pub fn is_es_gerund(word: &str) -> bool {
    let lower = word.to_lowercase();
    if lower.len() < 5 {
        return false;
    }
    if !lower.ends_with("ando") && !lower.ends_with("iendo") && !lower.ends_with("yendo") {
        return false;
    }
    if ES_GERUND_EXCEPTIONS.iter().any(|e| lower == *e) {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_adverb_basic() {
        assert!(is_en_adverb("quickly"));
        assert!(is_en_adverb("slowly"));
        assert!(is_en_adverb("beautifully"));
    }

    #[test]
    fn en_adverb_exceptions() {
        assert!(!is_en_adverb("only"));
        assert!(!is_en_adverb("family"));
        assert!(!is_en_adverb("early"));
        assert!(!is_en_adverb("ugly"));
    }

    #[test]
    fn en_adverb_short_words() {
        assert!(!is_en_adverb("fly"));
        assert!(!is_en_adverb("sly"));
    }

    #[test]
    fn es_adverb_basic() {
        assert!(is_es_adverb("rápidamente"));
        assert!(is_es_adverb("claramente"));
        assert!(is_es_adverb("felizmente"));
    }

    #[test]
    fn es_adverb_short() {
        assert!(!is_es_adverb("mente"));
        assert!(!is_es_adverb("lente"));
    }

    #[test]
    fn en_gerund_basic() {
        assert!(is_en_gerund("running"));
        assert!(is_en_gerund("swimming"));
        assert!(is_en_gerund("eating"));
    }

    #[test]
    fn en_gerund_exceptions() {
        assert!(!is_en_gerund("thing"));
        assert!(!is_en_gerund("king"));
        assert!(!is_en_gerund("during"));
        assert!(!is_en_gerund("ring"));
        assert!(!is_en_gerund("ceiling"));
    }

    #[test]
    fn en_gerund_short() {
        assert!(!is_en_gerund("ing"));
        assert!(!is_en_gerund("sing"));
    }

    #[test]
    fn es_gerund_basic() {
        assert!(is_es_gerund("hablando"));
        assert!(is_es_gerund("comiendo"));
        assert!(is_es_gerund("yendo"));
    }

    #[test]
    fn es_gerund_exceptions() {
        assert!(!is_es_gerund("bando"));
        assert!(!is_es_gerund("blando"));
        assert!(!is_es_gerund("mando"));
    }

    #[test]
    fn gerund_flag_distinct_from_adv() {
        // A gerund is not tagged as ADV
        assert!(is_en_gerund("running"));
        assert!(!is_en_adverb("running"));
    }
}
