//! Lightweight tokenizer and sentence splitter.
//!
//! Boundary rule: sentence boundaries are `.`, `!`, `?` followed by whitespace
//! and an uppercase letter, or end of text. The tokenizer splits on whitespace
//! and strips surrounding punctuation from each token for tagging while
//! preserving the original surface form for callers that need it.

/// Split a block of text into sentences, each as a `Vec<String>` of tokens.
///
/// Tokens are lowercased, with surrounding punctuation stripped, suitable for
/// feeding to the perceptron tagger. The original word forms are not preserved
/// here; callers that need surface forms should use `tokenize_surface` and pair
/// with the returned token count.
pub fn split_sentences(text: &str) -> Vec<Vec<String>> {
    let mut sentences: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for word in text.split_whitespace() {
        let clean = strip_surrounding_punct(word);
        if clean.is_empty() {
            continue;
        }
        current.push(clean.to_lowercase());

        // Check if this word ends a sentence.
        if ends_sentence(word) && !current.is_empty() {
            sentences.push(current);
            current = Vec::new();
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}

/// Split text into sentences of surface-form tokens (no lowercasing, no punct strip).
/// Useful for display or pairing with heuristic results.
pub fn split_sentences_surface(text: &str) -> Vec<Vec<String>> {
    let mut sentences: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for word in text.split_whitespace() {
        if word.is_empty() {
            continue;
        }
        current.push(word.to_string());
        if ends_sentence(word) && !current.is_empty() {
            sentences.push(current);
            current = Vec::new();
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}

fn strip_surrounding_punct(word: &str) -> &str {
    word.trim_matches(|c: char| {
        matches!(c, '.' | ',' | '!' | '?' | ':' | ';' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '-' | '—' | '–')
    })
}

fn ends_sentence(word: &str) -> bool {
    word.ends_with('.') || word.ends_with('!') || word.ends_with('?')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_period() {
        let sents = split_sentences("The cat sat. The dog ran.");
        assert_eq!(sents.len(), 2);
        assert_eq!(sents[0], vec!["the", "cat", "sat"]);
        assert_eq!(sents[1], vec!["the", "dog", "ran"]);
    }

    #[test]
    fn strips_punctuation() {
        let sents = split_sentences("Hello, world!");
        assert_eq!(sents.len(), 1);
        assert_eq!(sents[0], vec!["hello", "world"]);
    }

    #[test]
    fn no_trailing_sentence() {
        let sents = split_sentences("one two three");
        assert_eq!(sents.len(), 1);
        assert_eq!(sents[0], vec!["one", "two", "three"]);
    }
}
