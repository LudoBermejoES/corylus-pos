#!/usr/bin/env python3
"""
Build en.pos.tar.gz from the NLTK averaged-perceptron tagger weights.

Source provenance:
  NLTK averaged_perceptron_tagger_eng (MIT license)
  https://github.com/nltk/nltk_data/blob/gh-pages/packages/taggers/
  averaged_perceptron_tagger_eng.zip

Usage:
  pip install nltk
  python scripts/build_model.py

Output:
  en.pos.json      — weights in {tags, weights, classes} JSON format
  en.pos.tar.gz    — tarball containing en.pos.json (for GitHub Release)

Records the SHA-256 of the NLTK source weights file in the output, so the
pinned artifact is fully traceable back to the upstream source commit.
"""
import hashlib
import json
import os
import pickle
import struct
import tarfile
import tempfile
from pathlib import Path

try:
    import nltk
    from nltk.tag.perceptron import PerceptronTagger
except ImportError:
    raise SystemExit("pip install nltk, then run: python -m nltk.downloader averaged_perceptron_tagger_eng")


def main():
    out_dir = Path(__file__).parent.parent
    json_path = out_dir / "en.pos.json"
    tar_path = out_dir / "en.pos.tar.gz"

    print("Loading NLTK perceptron tagger...")
    tagger = PerceptronTagger()

    # Locate the pickle file so we can record its SHA-256.
    import nltk.data
    pickle_path = nltk.data.find("taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.pickle")
    with open(pickle_path, "rb") as fh:
        raw = fh.read()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    print(f"Source pickle SHA-256: {source_sha256}")

    # Export weights in our JSON format.
    model = tagger.model
    payload = {
        "source": "NLTK averaged_perceptron_tagger_eng (MIT license)",
        "source_sha256": source_sha256,
        "tags": dict(model.tagdict),
        "weights": {k: dict(v) for k, v in model.weights.items()},
        "classes": sorted(model.classes),
    }

    print(f"Writing {json_path} ...")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    artifact_sha256 = hashlib.sha256(open(json_path, "rb").read()).hexdigest()
    print(f"Artifact SHA-256: {artifact_sha256}")

    print(f"Writing {tar_path} ...")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(json_path, arcname="en.pos.json")

    tarball_sha256 = hashlib.sha256(open(tar_path, "rb").read()).hexdigest()
    print(f"Tarball SHA-256 (for EngineConfig): {tarball_sha256}")
    print()
    print("Pin this in src/lib.rs EngineConfig::default_en():")
    print(f'  source_sha256: "{tarball_sha256}".into(),')


if __name__ == "__main__":
    main()
