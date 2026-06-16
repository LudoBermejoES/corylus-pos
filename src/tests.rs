use super::*;

#[test]
fn engine_new_es_not_installed() {
    // Spanish now has the AnCora model — starts NotInstalled, not heuristic-only.
    let cfg = EngineConfig::default_es(std::path::PathBuf::from("/tmp/test-pos"));
    let engine = PosEngine::new(cfg);
    assert_eq!(engine.state(), PosState::NotInstalled);
    assert!(!engine.is_heuristic_only());
}

#[test]
fn engine_new_heuristic_only_explicit() {
    // A config with empty source_url is still heuristic-only (used in tests / future rule-based lang).
    let cfg = EngineConfig {
        data_dir: std::path::PathBuf::from("/tmp/test-pos-heuristic"),
        lang: "es".into(),
        source_url: String::new(),
        source_sha256: String::new(),
    };
    let engine = PosEngine::new(cfg);
    assert_eq!(engine.state(), PosState::Ready);
    assert!(engine.is_heuristic_only());
}

#[test]
fn engine_new_en_not_installed() {
    let cfg = EngineConfig {
        data_dir: std::path::PathBuf::from("/tmp/test-pos-en"),
        lang: "en".into(),
        source_url: "https://example.com/en.pos.tar.gz".into(),
        source_sha256: "abc123".into(),
    };
    let engine = PosEngine::new(cfg);
    assert_eq!(engine.state(), PosState::NotInstalled);
}

#[test]
fn tag_batch_adverb_heuristic_no_model() {
    let cfg = EngineConfig::default_en(std::path::PathBuf::from("/tmp/test-pos-en-h"));
    // No model installed, but heuristic should still fire for adverbs.
    // Override with a config that has no source_url (heuristic-only, always Ready).
    let cfg = EngineConfig {
        source_url: String::new(),
        source_sha256: String::new(),
        ..cfg
    };
    let engine = PosEngine::new(cfg);
    let sentences = vec![vec!["quickly".to_string(), "ran".to_string()]];
    let results = engine.tag_batch(&sentences);
    assert_eq!(results.len(), 1);
    assert!(results[0][0].is_adverb_heuristic);
    assert!(!results[0][1].is_adverb_heuristic);
}

#[test]
fn tag_batch_gerund_heuristic_no_model() {
    let cfg = EngineConfig {
        data_dir: std::path::PathBuf::from("/tmp/test-pos-en-g"),
        lang: "en".into(),
        source_url: String::new(),
        source_sha256: String::new(),
    };
    let engine = PosEngine::new(cfg);
    let sentences = vec![vec!["running".to_string(), "fast".to_string()]];
    let results = engine.tag_batch(&sentences);
    assert!(results[0][0].is_gerund);
    assert!(!results[0][1].is_gerund);
}

#[test]
fn tag_batch_es_adverb_heuristic() {
    let cfg = EngineConfig::default_es(std::path::PathBuf::from("/tmp/test-pos-es"));
    let engine = PosEngine::new(cfg);
    let sentences = vec![vec!["rápidamente".to_string(), "corrió".to_string()]];
    let results = engine.tag_batch(&sentences);
    assert!(results[0][0].is_adverb_heuristic);
    assert!(!results[0][1].is_adverb_heuristic);
}

#[test]
fn tag_batch_es_gerund_heuristic() {
    let cfg = EngineConfig::default_es(std::path::PathBuf::from("/tmp/test-pos-es-g"));
    let engine = PosEngine::new(cfg);
    let sentences = vec![vec!["hablando".to_string(), "corrió".to_string()]];
    let results = engine.tag_batch(&sentences);
    assert!(results[0][0].is_gerund);
    assert!(!results[0][1].is_gerund);
}

#[test]
fn gerund_flag_distinct_from_adv() {
    let cfg = EngineConfig {
        data_dir: std::path::PathBuf::from("/tmp/test-pos-en-ga"),
        lang: "en".into(),
        source_url: String::new(),
        source_sha256: String::new(),
    };
    let engine = PosEngine::new(cfg);
    let sentences = vec![vec!["running".to_string(), "quickly".to_string()]];
    let results = engine.tag_batch(&sentences);
    // "running" is gerund but not adverb
    assert!(results[0][0].is_gerund);
    assert!(!results[0][0].is_adverb_heuristic);
    // "quickly" is adverb but not gerund
    assert!(!results[0][1].is_gerund);
    assert!(results[0][1].is_adverb_heuristic);
}

// ── Integration tests (network required — run with `cargo test -- --ignored`) ──

/// Install the EN model, verify ADJ/ADV tagging on known sentences, then uninstall.
///
/// Sentence "the fast car" → "fast" is JJ (ADJ) in context.
/// Sentence "he ran fast" → "fast" is RB (ADV) in context (same word, different UPOS).
/// This is the canonical disambiguation test for the perceptron — it can only pass
/// with the contextual model loaded, not with heuristics alone.
#[tokio::test]
#[ignore]
async fn integration_en_install_tag_uninstall() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EngineConfig::default_en(dir.path().join("pos/en"));
    let engine = PosEngine::new(cfg);
    assert_eq!(engine.state(), PosState::NotInstalled);

    engine.provision(|s| println!("EN progress: {:?}", s)).await
        .expect("EN model download failed");
    assert_eq!(engine.state(), PosState::Ready);

    // "the fast car" — "fast" should be ADJ in this context
    let adj_sents = vec![vec!["the".to_string(), "fast".to_string(), "car".to_string()]];
    let adj_results = engine.tag_batch(&adj_sents);
    assert_eq!(adj_results[0][1].upos, Some(Upos::Adj),
        "\"fast\" in \"the fast car\" should be ADJ, got {:?}", adj_results[0][1].upos);

    // "he ran fast" — "fast" should be ADV in this context
    let adv_sents = vec![vec!["he".to_string(), "ran".to_string(), "fast".to_string()]];
    let adv_results = engine.tag_batch(&adv_sents);
    assert_eq!(adv_results[0][2].upos, Some(Upos::Adv),
        "\"fast\" in \"he ran fast\" should be ADV, got {:?}", adv_results[0][2].upos);

    // Adverb heuristic still fires for "-ly" words with model loaded
    let adv_sents2 = vec![vec!["quickly".to_string(), "ran".to_string()]];
    let adv_results2 = engine.tag_batch(&adv_sents2);
    assert!(adv_results2[0][0].is_adverb_heuristic, "quickly should be adverb heuristic");
    assert_eq!(adv_results2[0][0].upos, Some(Upos::Adv),
        "quickly should be ADV from model too, got {:?}", adv_results2[0][0].upos);

    // Uninstall returns to NotInstalled
    engine.uninstall().expect("uninstall failed");
    assert_eq!(engine.state(), PosState::NotInstalled);
}

/// Install the ES AnCora model, verify ADJ tagging on known Spanish sentences, then uninstall.
///
/// Adjectives tested:
///   "bonito" (JJ in NLTK-style / ADJ in UD) in "el coche bonito"
///   "grande"  in "un gran problema"
///   "rápido"  in "un coche rápido"
///
/// Adverb heuristic tested in parallel: "rápidamente" still fires is_adverb_heuristic
/// regardless of model.
#[tokio::test]
#[ignore]
async fn integration_es_install_tag_uninstall() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EngineConfig::default_es(dir.path().join("pos/es"));
    let engine = PosEngine::new(cfg);
    assert_eq!(engine.state(), PosState::NotInstalled);
    assert!(!engine.is_heuristic_only(), "ES must NOT be heuristic-only once AnCora model is pinned");

    engine.provision(|s| println!("ES progress: {:?}", s)).await
        .expect("ES model download failed");
    assert_eq!(engine.state(), PosState::Ready);

    // "el coche bonito" → "bonito" (index 2) should be ADJ
    let sents = vec![
        vec!["el".to_string(), "coche".to_string(), "bonito".to_string()],
        vec!["un".to_string(), "coche".to_string(), "rápido".to_string()],
        vec!["ella".to_string(), "corre".to_string(), "rápidamente".to_string()],
    ];
    let results = engine.tag_batch(&sents);

    // "bonito" in "el coche bonito"
    assert_eq!(results[0][2].upos, Some(Upos::Adj),
        "\"bonito\" in \"el coche bonito\" should be ADJ, got {:?}", results[0][2].upos);

    // "rápido" in "un coche rápido"
    assert_eq!(results[1][2].upos, Some(Upos::Adj),
        "\"rápido\" in \"un coche rápido\" should be ADJ, got {:?}", results[1][2].upos);

    // "rápidamente" — heuristic fires (no model needed) AND model agrees it's ADV
    assert!(results[2][2].is_adverb_heuristic, "rápidamente should fire adverb heuristic");
    assert_eq!(results[2][2].upos, Some(Upos::Adv),
        "rápidamente should be ADV from model, got {:?}", results[2][2].upos);

    // Uninstall returns to NotInstalled
    engine.uninstall().expect("uninstall failed");
    assert_eq!(engine.state(), PosState::NotInstalled);
}
