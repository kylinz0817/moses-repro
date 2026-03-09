"""
scripts/download_data.py
------------------------
Downloads all required datasets for MoSEs reproduction study.

Original MoSEs datasets (CMV, SciXGen, WP, XSum, CNN, DialogSum, IMDB, PubMedQA)
should be copied from the original authors' repo:
    git clone https://github.com/creator-xi/MoSEs
    cp -r MoSEs/data/* data/raw/

Additional datasets (TruthfulQA, HC3) are downloaded automatically
from HuggingFace.

Usage:
    # Verify original datasets are present
    python scripts/download_data.py --datasets original

    # Download HC3 + TruthfulQA from HuggingFace
    python scripts/download_data.py --datasets additional

    # Do both
    python scripts/download_data.py --datasets all
"""

import os
import json
import argparse
import random
from pathlib import Path

ORIGINAL_DATASETS = ["CMV", "SciXGen", "WP", "XSum", "CNN", "DialogSum", "IMDB", "PubMedQA"]
ADDITIONAL_DATASETS = ["TruthfulQA", "HC3"]

SPLIT_SIZES = {
    "CMV":        {"ref": 1800, "test": 200},
    "SciXGen":    {"ref": 1800, "test": 200},
    "WP":         {"ref": 1800, "test": 200},
    "XSum":       {"ref": 1800, "test": 200},
    "CNN":        {"ref": 200,  "test": 200},
    "DialogSum":  {"ref": 200,  "test": 200},
    "IMDB":       {"ref": 200,  "test": 200},
    "PubMedQA":   {"ref": 200,  "test": 200},
    "TruthfulQA": {"ref": 200,  "test": 200},
    "HC3":        {"ref": 500,  "test": 200},
}


def verify_original_datasets(data_dir: Path) -> bool:
    """Check that all original dataset files exist."""
    missing = []
    for ds in ORIGINAL_DATASETS:
        for split in ["ref.json", "test.json"]:
            p = data_dir / ds / split
            if not p.exists():
                missing.append(str(p))

    if missing:
        print("\n[WARNING] Missing original dataset files:")
        for m in missing:
            print(f"  - {m}")
        print("\nFix: clone the original repo and copy data:")
        print("  git clone https://github.com/creator-xi/MoSEs")
        print("  cp -r MoSEs/data/* data/raw/")
        return False

    print("[OK] All 8 original datasets found.")
    return True


def download_hc3(data_dir: Path, seed: int = 42):
    """Download HC3 (Human-ChatGPT Comparison Corpus) from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    random.seed(seed)
    out_dir = data_dir / "HC3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "ref.json").exists() and (out_dir / "test.json").exists():
        print("[HC3] Already downloaded, skipping.")
        return

    print("[HC3] Downloading from HuggingFace (Hello-SimpleAI/HC3)...")
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")

    samples = []
    for item in ds:
        for ans in item.get("human_answers", []):
            if ans.strip():
                samples.append({"text": ans.strip(), "label": 0})
        for ans in item.get("chatgpt_answers", []):
            if ans.strip():
                samples.append({"text": ans.strip(), "label": 1})

    random.shuffle(samples)
    split = SPLIT_SIZES["HC3"]
    ref_data  = samples[: split["ref"]]
    test_data = samples[split["ref"] : split["ref"] + split["test"]]

    with open(out_dir / "ref.json", "w") as f:
        json.dump(ref_data, f, indent=2)
    with open(out_dir / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"[HC3] Saved {len(ref_data)} ref + {len(test_data)} test samples.")


def download_truthfulqa(data_dir: Path, seed: int = 42):
    """Download TruthfulQA from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    random.seed(seed)
    out_dir = data_dir / "TruthfulQA"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "ref.json").exists() and (out_dir / "test.json").exists():
        print("[TruthfulQA] Already downloaded, skipping.")
        return

    print("[TruthfulQA] Downloading from HuggingFace...")
    ds = load_dataset("truthful_qa", "generation", split="validation")

    samples = []
    for item in ds:
        # Correct/best answer = human label (0)
        if item["best_answer"].strip():
            samples.append({"text": item["best_answer"].strip(), "label": 0})
        # Incorrect answers = AI-hallucinated proxy (label 1)
        for ans in item.get("incorrect_answers", [])[:1]:
            if ans.strip():
                samples.append({"text": ans.strip(), "label": 1})

    random.shuffle(samples)
    split = SPLIT_SIZES["TruthfulQA"]
    ref_data  = samples[: split["ref"]]
    test_data = samples[split["ref"] : split["ref"] + split["test"]]

    with open(out_dir / "ref.json", "w") as f:
        json.dump(ref_data, f, indent=2)
    with open(out_dir / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"[TruthfulQA] Saved {len(ref_data)} ref + {len(test_data)} test samples.")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for MoSEs reproduction.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Which datasets: 'all', 'original', 'additional', 'HC3', 'TruthfulQA'",
    )
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = args.datasets
    if "all" in targets:
        targets = ["original", "additional"]

    if "original" in targets:
        print("\n--- Verifying original datasets ---")
        verify_original_datasets(data_dir)

    if "additional" in targets or "HC3" in targets:
        print("\n--- Downloading HC3 ---")
        download_hc3(data_dir, seed=args.seed)

    if "additional" in targets or "TruthfulQA" in targets:
        print("\n--- Downloading TruthfulQA ---")
        download_truthfulqa(data_dir, seed=args.seed)

    print("\nDone. data/raw/ contents:")
    for p in sorted(data_dir.iterdir()):
        if p.is_dir():
            files = list(p.glob("*.json"))
            print(f"  {p.name}/  ({len(files)} json files)")


if __name__ == "__main__":
    main()
