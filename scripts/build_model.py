#!/usr/bin/env python3
"""
Build en.pos.tar.gz from the NLTK averaged-perceptron tagger weights.

Source provenance:
  NLTK averaged_perceptron_tagger_eng (MIT license)
  https://github.com/nltk/nltk_data/blob/gh-pages/packages/taggers/
  averaged_perceptron_tagger_eng.zip

Usage:
  pip install nltk
  python -m nltk.downloader averaged_perceptron_tagger_eng
  python scripts/build_model.py

Output:
  en.pos.json      — weights in {tags, weights, classes} JSON format
  en.pos.tar.gz    — tarball containing en.pos.json (for GitHub Release)

Records the SHA-256 of the NLTK source weights files in the output, so the
pinned artifact is fully traceable back to the upstream source files.

Supports both old NLTK (pickle format) and new NLTK >=3.8 (JSON format).
"""
import hashlib
import json
import tarfile
from pathlib import Path

try:
    import nltk
    from nltk.tag.perceptron import PerceptronTagger
except ImportError:
    raise SystemExit("pip install nltk, then run: python -m nltk.downloader averaged_perceptron_tagger_eng")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out_dir = Path(__file__).parent.parent
    json_path = out_dir / "en.pos.json"
    tar_path = out_dir / "en.pos.tar.gz"

    # Locate the NLTK tagger data directory.
    import nltk.data
    tagger_dir = Path(nltk.data.find("taggers/averaged_perceptron_tagger_eng"))
    print(f"NLTK tagger data: {tagger_dir}")

    # New NLTK (>=3.8) stores three separate JSON files.
    weights_file = tagger_dir / "averaged_perceptron_tagger_eng.weights.json"
    tagdict_file = tagger_dir / "averaged_perceptron_tagger_eng.tagdict.json"
    classes_file = tagger_dir / "averaged_perceptron_tagger_eng.classes.json"

    if weights_file.exists():
        print("Detected new-style NLTK JSON format.")
        weights_sha256 = sha256_file(weights_file)
        tagdict_sha256 = sha256_file(tagdict_file)
        classes_sha256 = sha256_file(classes_file)
        print(f"  weights SHA-256: {weights_sha256}")
        print(f"  tagdict SHA-256: {tagdict_sha256}")
        print(f"  classes SHA-256: {classes_sha256}")
        source_sha256 = hashlib.sha256(
            (weights_sha256 + tagdict_sha256 + classes_sha256).encode()
        ).hexdigest()

        with open(weights_file, encoding="utf-8") as fh:
            raw_weights = json.load(fh)
        with open(tagdict_file, encoding="utf-8") as fh:
            tagdict = json.load(fh)
        with open(classes_file, encoding="utf-8") as fh:
            classes = json.load(fh)

        # raw_weights maps feature → {tag: weight}; values may be plain dicts
        weights = {k: dict(v) for k, v in raw_weights.items()}
        tags = dict(tagdict)

    else:
        # Old NLTK: single pickle file.
        pickle_path = tagger_dir / "averaged_perceptron_tagger_eng.pickle"
        if not pickle_path.exists():
            raise SystemExit(f"Could not find NLTK tagger data in {tagger_dir}")
        print("Detected old-style NLTK pickle format.")
        source_sha256 = sha256_file(pickle_path)
        print(f"  pickle SHA-256: {source_sha256}")

        print("Loading NLTK perceptron tagger...")
        tagger = PerceptronTagger()
        model = tagger.model
        weights = {k: dict(v) for k, v in model.weights.items()}
        tags = dict(model.tagdict)
        classes = sorted(model.classes)

    payload = {
        "source": "NLTK averaged_perceptron_tagger_eng (MIT license)",
        "source_sha256": source_sha256,
        "tags": tags,
        "weights": weights,
        "classes": sorted(classes),
    }

    print(f"Writing {json_path} ...")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    print(f"Writing {tar_path} ...")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(json_path, arcname="en.pos.json")

    tarball_sha256 = sha256_file(tar_path)
    print(f"\nTarball SHA-256 (for EngineConfig): {tarball_sha256}")
    print()
    print("Pin these in src/lib.rs EngineConfig::default_en():")
    print(f'  source_url: "https://github.com/LudoBermejoES/corylus-pos/releases/download/v1.0.0/en.pos.tar.gz".into(),')
    print(f'  source_sha256: "{tarball_sha256}".into(),')


if __name__ == "__main__":
    main()
