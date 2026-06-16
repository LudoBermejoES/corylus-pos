#!/usr/bin/env python3
"""
Train a Spanish averaged-perceptron POS tagger from UD Spanish AnCora and
export weights to es.pos.tar.gz for the corylus-pos GitHub Release.

Corpus provenance:
  UD Spanish AnCora (CC-BY 4.0)
  https://github.com/UniversalDependencies/UD_Spanish-AnCora
  Trained model redistributed with attribution per CC-BY 4.0 terms.
  Source corpus NOT redistributed — only the trained model artifact.

Usage:
  pip install conllu
  # Download corpus files:
  curl -sL https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu -o /tmp/ancora/es_ancora-ud-train.conllu
  curl -sL https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-dev.conllu   -o /tmp/ancora/es_ancora-ud-dev.conllu
  curl -sL https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-test.conllu  -o /tmp/ancora/es_ancora-ud-test.conllu
  python scripts/build_es_model.py

Output:
  es.pos.json      — weights in {tags, weights, classes} JSON format
                     (same format as en.pos.json / NLTK weights)
  es.pos.tar.gz    — tarball containing es.pos.json (for GitHub Release)
"""
import collections
import hashlib
import json
import random
import tarfile
from pathlib import Path

try:
    import conllu
except ImportError:
    raise SystemExit("pip install conllu")

CORPUS_DIR = Path("/tmp/ancora")
TRAIN_FILE = CORPUS_DIR / "es_ancora-ud-train.conllu"
DEV_FILE   = CORPUS_DIR / "es_ancora-ud-dev.conllu"
TEST_FILE  = CORPUS_DIR / "es_ancora-ud-test.conllu"


# ─── Data loading ─────────────────────────────────────────────────────────────

def read_tagged_sentences(path: Path):
    """Yield (tokens, tags) from a CoNLL-U file, using the UPOS column."""
    with open(path, encoding="utf-8") as fh:
        for sentence in conllu.parse_incr(fh):
            tokens, tags = [], []
            for token in sentence:
                if not isinstance(token["id"], int):
                    continue  # skip multiword tokens
                form = token["form"].lower()
                upos = token["upos"] or "X"
                tokens.append(form)
                tags.append(upos)
            if tokens:
                yield tokens, tags


# ─── Feature extraction ───────────────────────────────────────────────────────

def get_features(i, word, tokens, prev_tag, prev2_tag):
    """Return a dict of feature_key → 1.0 (indicator features, NLTK-style).

    Feature keys concatenate the feature name and its value so each unique
    (name, value) pair is a distinct indicator — exactly the format that the
    Rust PerceptronTagger's weight lookup expects.
    """
    def suffix(w, n):
        return w[-n:] if len(w) >= n else w

    def word_at(idx):
        if idx < 0:
            return "-START-" if idx == -1 else "-START2-"
        return tokens[idx] if idx < len(tokens) else "-END-"

    w_prev  = word_at(i - 1)
    w_prev2 = word_at(i - 2)
    w_next  = word_at(i + 1)
    w_next2 = word_at(i + 2)

    # Keys use space separator to match NLTK's format: "i suffix ing", "i-1 tag+i word VB run"
    features = {
        "bias": 1.0,
        f"i suffix {suffix(word, 3)}": 1.0,
        f"i pref1 {word[:1]}": 1.0,
        f"i-1 tag {prev_tag}": 1.0,
        f"i-2 tag {prev2_tag}": 1.0,
        f"i tag+i-2 tag {prev_tag} {prev2_tag}": 1.0,
        f"i word {word}": 1.0,
        f"i-1 tag+i word {prev_tag} {word}": 1.0,
        f"i-1 word {w_prev}": 1.0,
        f"i-1 suffix {suffix(w_prev, 3)}": 1.0,
        f"i-2 word {w_prev2}": 1.0,
        f"i+1 word {w_next}": 1.0,
        f"i+1 suffix {suffix(w_next, 3)}": 1.0,
        f"i+2 word {w_next2}": 1.0,
    }
    return features


# ─── Averaged perceptron ──────────────────────────────────────────────────────

class AveragedPerceptron:
    def __init__(self, classes):
        self.classes = classes
        # weights[feature][tag] → current weight
        self.weights = collections.defaultdict(lambda: collections.defaultdict(float))
        # For averaging: accumulated total and last-update step
        self._totals = collections.defaultdict(lambda: collections.defaultdict(float))
        self._timestamps = collections.defaultdict(lambda: collections.defaultdict(int))
        self.tagdict = {}
        self._step = 0

    def score(self, features):
        scores = {c: 0.0 for c in self.classes}
        for feat, val in features.items():
            if feat not in self.weights:
                continue
            for tag, weight in self.weights[feat].items():
                scores[tag] += val * weight
        return scores

    def predict(self, features):
        scores = self.score(features)
        return max(scores, key=scores.get)

    def update(self, truth, guess, features):
        self._step += 1
        if truth == guess:
            return
        for feat in features:
            w = self.weights[feat]
            t = self._timestamps[feat]
            tot = self._totals[feat]
            for label in (truth, guess):
                # Flush accumulated average up to now
                tot[label] += (self._step - t[label]) * w.get(label, 0.0)
                t[label] = self._step
            w[truth] = w.get(truth, 0.0) + 1.0
            w[guess] = w.get(guess, 0.0) - 1.0

    def average_weights(self):
        """Replace current weights with their time-averaged values."""
        averaged = {}
        for feat, weights in self.weights.items():
            t = self._timestamps[feat]
            tot = self._totals[feat]
            avg = {}
            for label, weight in weights.items():
                total = tot[label] + (self._step - t[label]) * weight
                v = round(total / self._step, 6)
                if v != 0.0:
                    avg[label] = v
            if avg:
                averaged[feat] = avg
        self.weights = averaged

    def build_tagdict(self, sentences, freq_threshold=20, certainty=0.97):
        """Pre-compute unambiguous tag assignments for frequent words."""
        counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for tokens, tags in sentences:
            for word, tag in zip(tokens, tags):
                counts[word][tag] += 1
        for word, tag_counts in counts.items():
            total = sum(tag_counts.values())
            best_tag, best_count = max(tag_counts.items(), key=lambda x: x[1])
            if best_count >= freq_threshold and best_count / total >= certainty:
                self.tagdict[word] = best_tag


# ─── Training loop ────────────────────────────────────────────────────────────

def train(sentences, n_iter=5, seed=42):
    all_classes = set()
    for _, tags in sentences:
        all_classes.update(tags)

    model = AveragedPerceptron(all_classes)
    model.build_tagdict(sentences)
    print(f"  Tag dict: {len(model.tagdict)} entries, {len(all_classes)} classes")

    rng = random.Random(seed)
    for it in range(n_iter):
        correct = total = 0
        rng.shuffle(sentences)
        for tokens, gold_tags in sentences:
            prev, prev2 = "-START-", "-START2-"
            for i, (word, truth) in enumerate(zip(tokens, gold_tags)):
                guess = model.tagdict.get(word)
                if guess is None:
                    feats = get_features(i, word, tokens, prev, prev2)
                    guess = model.predict(feats)
                    model.update(truth, guess, feats)
                prev2, prev = prev, truth
                correct += int(guess == truth)
                total += 1
        print(f"  Iter {it + 1}/{n_iter}: train acc {100 * correct / total:.2f}%")

    model.average_weights()
    return model


def evaluate(model, sentences):
    correct = total = 0
    for tokens, gold_tags in sentences:
        prev, prev2 = "-START-", "-START2-"
        for i, (word, truth) in enumerate(zip(tokens, gold_tags)):
            guess = model.tagdict.get(word)
            if guess is None:
                feats = get_features(i, word, tokens, prev, prev2)
                guess = model.predict(feats)
            prev2, prev = prev, guess
            correct += int(guess == truth)
            total += 1
    return 100 * correct / total if total else 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out_dir = Path(__file__).parent.parent
    json_path = out_dir / "es.pos.json"
    tar_path = out_dir / "es.pos.tar.gz"

    print("Reading corpus...")
    train_sents = list(read_tagged_sentences(TRAIN_FILE))
    dev_sents   = list(read_tagged_sentences(DEV_FILE))
    test_sents  = list(read_tagged_sentences(TEST_FILE))
    print(f"  train={len(train_sents)}, dev={len(dev_sents)}, test={len(test_sents)} sentences")

    train_sha256 = sha256_file(TRAIN_FILE)
    dev_sha256   = sha256_file(DEV_FILE)
    test_sha256  = sha256_file(TEST_FILE)
    source_sha256 = hashlib.sha256(
        (train_sha256 + dev_sha256 + test_sha256).encode()
    ).hexdigest()
    print(f"  corpus SHA-256: {source_sha256}")

    print("\nTraining...")
    model = train(train_sents, n_iter=5)

    dev_acc  = evaluate(model, dev_sents)
    test_acc = evaluate(model, test_sents)
    print(f"\nDev accuracy:  {dev_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")

    print(f"\nWriting {json_path} ...")
    payload = {
        "source": "UD Spanish AnCora (CC-BY 4.0) — trained model only, corpus not redistributed",
        "source_url": "https://github.com/UniversalDependencies/UD_Spanish-AnCora",
        "source_sha256": source_sha256,
        "dev_accuracy": round(dev_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "tags": model.tagdict,
        "weights": {k: dict(v) for k, v in model.weights.items()},
        "classes": sorted(model.classes),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    print(f"Writing {tar_path} ...")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(json_path, arcname="es.pos.json")

    tarball_sha256 = sha256_file(tar_path)
    print(f"\nTarball SHA-256: {tarball_sha256}")
    print()
    print("Pin in src/lib.rs EngineConfig::default_es():")
    print(f'  source_url: "https://github.com/LudoBermejoES/corylus-pos/releases/download/v1.0.1/es.pos.tar.gz".into(),')
    print(f'  source_sha256: "{tarball_sha256}".into(),')


if __name__ == "__main__":
    main()
