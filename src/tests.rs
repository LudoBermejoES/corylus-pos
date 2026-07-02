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

// Regression test: `new()` used to have a redundant
// `if is_installed_for(&config) { NotInstalled } else { NotInstalled }` branch
// (both arms identical) with a comment claiming set_data_dir would later
// upgrade the state. That branch never distinguished any real behavior since
// `new()` alone never calls `try_load_model` — only `set_data_dir` does. This
// exercises the exact case the dead branch checked (model files already
// present on disk at construction time) and confirms `new()` still reports
// `NotInstalled` immediately afterward.
#[test]
fn engine_new_reports_not_installed_even_when_model_files_preexist_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        data_dir: dir.path().to_path_buf(),
        lang: "en".into(),
        source_url: "https://example.com/en.pos.tar.gz".into(),
        source_sha256: "abc123".into(),
    };
    // Fabricate an "already installed" marker matching is_installed_for's checks.
    std::fs::write(state::weights_path(&cfg), "{}").unwrap();
    std::fs::write(
        state::version_path(&cfg),
        serde_json::to_string(&state::VersionFile {
            lang: cfg.lang.clone(),
            source_sha256: cfg.source_sha256.clone(),
            schema_version: state::SCHEMA_VERSION,
        }).unwrap(),
    ).unwrap();
    assert!(state::is_installed_for(&cfg), "fixture must satisfy is_installed_for");

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

/// Install the ES AnCora model, verify ADJ tagging across a broad set of Spanish
/// sentences, then uninstall.
///
/// Coverage:
///   Post-nominal ADJ  — most common Spanish position; must not confuse with NOUN
///   Pre-nominal ADJ   — apocope forms: gran, buen, mal
///   Gender/number agreement — feliz (invariant), felices (plural)
///   Predicate ADJ after ser/estar — "el cielo está azul", "la solución es perfecta"
///   Attributive after determiner — "muy bueno"
///   Ambiguous lemma — "libre"/"libres" depending on syntactic context
///   Non-ADJ sanity check — adverbs and pronouns must not be tagged ADJ
#[tokio::test]
#[ignore]
async fn integration_es_install_tag_uninstall() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EngineConfig::default_es(dir.path().join("pos/es"));
    let engine = PosEngine::new(cfg);
    assert_eq!(engine.state(), PosState::NotInstalled);
    assert!(!engine.is_heuristic_only(),
        "ES must NOT be heuristic-only once AnCora model is pinned");

    engine.provision(|s| println!("ES progress: {:?}", s)).await
        .expect("ES model download failed");
    assert_eq!(engine.state(), PosState::Ready);

    // ── Post-nominal adjectives ───────────────────────────────────────────────
    // These are the most common Spanish pattern: DET NOUN ADJ.
    // A model that only learned prefix/suffix patterns would fail here because
    // "bonito", "grande", "interesante", "inteligente" have no single shared
    // suffix — the model must use left-context (NOUN tag at i-1).
    let post_nominal = vec![
        vec!["el".to_string(), "coche".to_string(), "bonito".to_string()],
        vec!["la".to_string(), "casa".to_string(), "grande".to_string()],
        vec!["un".to_string(), "libro".to_string(), "interesante".to_string()],
        vec!["el".to_string(), "hombre".to_string(), "inteligente".to_string()],
    ];
    let r = engine.tag_batch(&post_nominal);
    assert_eq!(r[0][2].upos, Some(Upos::Adj),
        "\"bonito\" in DET NOUN __ should be ADJ");
    assert_eq!(r[1][2].upos, Some(Upos::Adj),
        "\"grande\" in DET NOUN __ should be ADJ");
    assert_eq!(r[2][2].upos, Some(Upos::Adj),
        "\"interesante\" in DET NOUN __ should be ADJ");
    assert_eq!(r[3][2].upos, Some(Upos::Adj),
        "\"inteligente\" in DET NOUN __ should be ADJ");

    // ── Pre-nominal (apocope) adjectives ─────────────────────────────────────
    // "gran", "buen", "mal" are shortened forms that appear before the noun.
    // The model must tag them ADJ using right-context (NOUN at i+1).
    let pre_nominal = vec![
        vec!["gran".to_string(), "problema".to_string()],
        vec!["buen".to_string(), "trabajo".to_string()],
        vec!["mal".to_string(), "tiempo".to_string()],
    ];
    let r = engine.tag_batch(&pre_nominal);
    assert_eq!(r[0][0].upos, Some(Upos::Adj), "\"gran\" before NOUN should be ADJ");
    assert_eq!(r[1][0].upos, Some(Upos::Adj), "\"buen\" before NOUN should be ADJ");
    assert_eq!(r[2][0].upos, Some(Upos::Adj), "\"mal\" before NOUN should be ADJ");

    // ── Gender/number invariance ──────────────────────────────────────────────
    // "feliz" is invariant for gender; "felices" is the plural form.
    // Both must be tagged ADJ regardless of the noun's gender.
    let agreement = vec![
        vec!["la".to_string(), "niña".to_string(), "feliz".to_string()],
        vec!["el".to_string(), "niño".to_string(), "feliz".to_string()],
        vec!["las".to_string(), "niñas".to_string(), "felices".to_string()],
    ];
    let r = engine.tag_batch(&agreement);
    assert_eq!(r[0][2].upos, Some(Upos::Adj), "\"feliz\" (fem.) should be ADJ");
    assert_eq!(r[1][2].upos, Some(Upos::Adj), "\"feliz\" (masc.) should be ADJ");
    assert_eq!(r[2][2].upos, Some(Upos::Adj), "\"felices\" (plural) should be ADJ");

    // ── Predicate adjectives after ser/estar ─────────────────────────────────
    // "azul" after "está" and "perfecta" after "es" must be ADJ, not NOUN.
    // A shallow model might be fooled by the verb-gap.
    let predicate = vec![
        vec!["el".to_string(), "cielo".to_string(), "está".to_string(), "azul".to_string()],
        vec!["la".to_string(), "solución".to_string(), "es".to_string(), "perfecta".to_string()],
    ];
    let r = engine.tag_batch(&predicate);
    assert_eq!(r[0][3].upos, Some(Upos::Adj),
        "\"azul\" after estar should be ADJ");
    assert_eq!(r[1][3].upos, Some(Upos::Adj),
        "\"perfecta\" after ser should be ADJ");

    // ── Ambiguous: "libre"/"libres" ───────────────────────────────────────────
    // In "tiempo libre" it is ADJ; in "somos libres" it is predicate ADJ.
    // Contrast: in "en libertad" it would be NOUN (not tested here).
    let ambiguous = vec![
        vec!["tiempo".to_string(), "libre".to_string()],
        vec!["somos".to_string(), "libres".to_string()],
    ];
    let r = engine.tag_batch(&ambiguous);
    assert_eq!(r[0][1].upos, Some(Upos::Adj), "\"libre\" modifying NOUN should be ADJ");
    assert_eq!(r[1][1].upos, Some(Upos::Adj), "\"libres\" after AUX should be ADJ");

    // ── Adverb heuristic alongside model ─────────────────────────────────────
    // "-mente" adverbs must fire the heuristic flag AND be tagged ADV by the model.
    let adv_sents = vec![
        vec!["ella".to_string(), "corre".to_string(), "rápidamente".to_string()],
        vec!["él".to_string(), "habla".to_string(), "claramente".to_string()],
    ];
    let r = engine.tag_batch(&adv_sents);
    assert!(r[0][2].is_adverb_heuristic, "rápidamente should fire adverb heuristic");
    assert_eq!(r[0][2].upos, Some(Upos::Adv),
        "rápidamente should be ADV from model");
    assert!(r[1][2].is_adverb_heuristic, "claramente should fire adverb heuristic");
    assert_eq!(r[1][2].upos, Some(Upos::Adv),
        "claramente should be ADV from model");

    // ── Non-ADJ sanity checks ─────────────────────────────────────────────────
    // Words that must NOT be tagged ADJ: adverbs and pronouns.
    let non_adj = vec![
        vec!["ayer".to_string(), "llegué".to_string()],   // ayer = ADV
        vec!["ellos".to_string(), "corren".to_string()],  // ellos = PRON
    ];
    let r = engine.tag_batch(&non_adj);
    assert_ne!(r[0][0].upos, Some(Upos::Adj), "\"ayer\" must not be ADJ");
    assert_ne!(r[1][0].upos, Some(Upos::Adj), "\"ellos\" must not be ADJ");

    // ── Uninstall ─────────────────────────────────────────────────────────────
    engine.uninstall().expect("uninstall failed");
    assert_eq!(engine.state(), PosState::NotInstalled);
}
