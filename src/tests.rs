use super::*;

#[test]
fn engine_new_heuristic_only() {
    let cfg = EngineConfig::default_es(std::path::PathBuf::from("/tmp/test-pos"));
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
