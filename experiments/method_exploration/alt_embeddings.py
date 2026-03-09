"""
experiments/method_exploration/alt_embeddings.py
-------------------------------------------------
Additional Experiment 2: Method Exploration — Alternative Embedding Models

Research question:
  MoSEs uses BGE-M3 (1024D) as the semantic embedding backbone for SAR.
  How sensitive is MoSEs to the choice of embedding model?
  Can a smaller/faster model (MiniLM, 384D) or a different large model
  (E5-large, 1024D) achieve comparable performance?

Models compared:
  1. BGE-M3 (1024D) — original paper choice
  2. E5-large-v2 (1024D) — alternative large embedding model
  3. all-MiniLM-L6-v2 (384D) — lightweight, fast alternative
  4. all-mpnet-base-v2 (768D) — strong general-purpose model

This experiment isolates the embedding component while keeping
SAR and CTE identical.

Usage:
    python experiments/method_exploration/alt_embeddings.py \
        --dataset CMV SciXGen \
        --detector roberta \
        --embedding_models bge-m3 e5-large minilm mpnet \
        --output_dir results/alt_embeddings/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

import json
import pickle
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_INFO = {
    "bge-m3": {"name": "BGE-M3", "dim": 1024, "params": "570M", "type": "flagembedding"},
    "e5-large": {"name": "E5-large-v2", "dim": 1024, "params": "335M", "type": "sentence_transformer"},
    "minilm": {"name": "all-MiniLM-L6-v2", "dim": 384, "params": "22M", "type": "sentence_transformer"},
    "mpnet": {"name": "all-mpnet-base-v2", "dim": 768, "params": "110M", "type": "sentence_transformer"},
}


def run_embedding_comparison(
    datasets: List[str],
    detectors: List[str],
    embedding_models: List[str],
    data_dir: Path,
    processed_dir: Path,
    sar_dir: Path,
    output_dir: Path,
    pca_dims: int = 32,
    device: str = "cuda",
    seed: int = 42,
) -> Dict:
    """
    For each embedding model, run the full MoSEs pipeline and collect results.
    """
    from preprocess import preprocess_dataset
    from train_sar import train_sar_for_dataset
    from train_cte import evaluate_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for emb_model in embedding_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Embedding model: {emb_model} ({EMBEDDING_MODEL_INFO[emb_model]['name']})")
        logger.info(f"{'='*60}")

        emb_dir = processed_dir / f"emb_{emb_model}"
        emb_sar_dir = sar_dir / f"emb_{emb_model}"

        t_start = time.time()

        for dataset in datasets:
            for detector in detectors:
                logger.info(f"\n  {dataset} / {detector} / {emb_model}")

                # Preprocess with this embedding model
                try:
                    preprocess_dataset(
                        dataset=dataset,
                        detector=detector,
                        data_dir=data_dir,
                        output_dir=emb_dir,
                        pca_dims=pca_dims,
                        device=device,
                        embedding_model=emb_model,
                        seed=seed,
                    )
                except FileNotFoundError as e:
                    logger.warning(f"  Skipping: {e}")
                    continue

                # Train SAR
                try:
                    train_sar_for_dataset(
                        dataset=dataset,
                        detector=detector,
                        processed_dir=emb_dir,
                        output_dir=emb_sar_dir,
                        embedding_model=emb_model,
                        epochs=100,
                        epsilon=0.05,
                        seed=seed,
                    )
                except Exception as e:
                    logger.error(f"  SAR error: {e}")

                # Evaluate
                try:
                    r = evaluate_dataset(
                        dataset=dataset,
                        detector=detector,
                        processed_dir=emb_dir,
                        sar_dir=emb_sar_dir,
                        output_dir=output_dir / emb_model,
                        embedding_model=emb_model,
                        cte_types=["lr", "xgb"],
                        seed=seed,
                    )
                    r["embedding_model"] = emb_model
                    r["embedding_dim"] = EMBEDDING_MODEL_INFO[emb_model]["dim"]
                    r["embedding_params"] = EMBEDDING_MODEL_INFO[emb_model]["params"]
                    all_results.append(r)
                except Exception as e:
                    logger.error(f"  Eval error: {e}")

        emb_time = time.time() - t_start
        logger.info(f"  Total time for {emb_model}: {emb_time:.1f}s")

    # Save results
    results_path = output_dir / "alt_embeddings_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    _print_embedding_comparison_table(all_results, datasets, detectors, embedding_models)

    # Plot
    _plot_embedding_comparison(all_results, output_dir)

    return all_results


def _print_embedding_comparison_table(results, datasets, detectors, embedding_models):
    print("\n" + "=" * 100)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 100)
    print(f"{'Dataset':<12} {'Detector':<15}", end="")
    for emb in embedding_models:
        name = EMBEDDING_MODEL_INFO[emb]["name"]
        print(f" {name[:12]:>12}", end="")
    print()
    print("-" * 100)

    # Organize by dataset/detector
    result_map = {}
    for r in results:
        key = (r["dataset"], r["detector"], r["embedding_model"])
        result_map[key] = r

    for dataset in datasets:
        for detector in detectors:
            print(f"{dataset:<12} {detector:<15}", end="")
            for emb in embedding_models:
                r = result_map.get((dataset, detector, emb), {})
                acc = r.get("moses_lr", {}).get("accuracy", float("nan"))
                print(f" {acc:>12.4f}", end="")
            print()
    print("=" * 100)


def _plot_embedding_comparison(results, output_dir):
    """Plot accuracy vs embedding model as bar chart."""
    try:
        from collections import defaultdict
        import numpy as np

        # Average accuracy per embedding model across datasets
        emb_accs = defaultdict(list)
        for r in results:
            lr_acc = r.get("moses_lr", {}).get("accuracy")
            if lr_acc is not None:
                emb_accs[r["embedding_model"]].append(lr_acc)

        models = list(emb_accs.keys())
        means = [np.mean(emb_accs[m]) for m in models]
        stds = [np.std(emb_accs[m]) for m in models]
        labels = [EMBEDDING_MODEL_INFO.get(m, {}).get("name", m) for m in models]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                      color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Accuracy (MoSEs-lr)")
        ax.set_title("MoSEs-lr Accuracy vs. Embedding Model\n(mean ± std across datasets)")
        ax.set_ylim(0.5, 1.0)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        fig.savefig(output_dir / "embedding_comparison.png", dpi=150)
        plt.close()
        logger.info(f"Plot saved: {output_dir / 'embedding_comparison.png'}")
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["CMV", "SciXGen", "WP", "XSum"])
    parser.add_argument("--detector", nargs="+", default=["roberta"])
    parser.add_argument(
        "--embedding_models",
        nargs="+",
        default=list(EMBEDDING_MODEL_INFO.keys()),
        choices=list(EMBEDDING_MODEL_INFO.keys()),
    )
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--sar_dir", default="results/sar_models")
    parser.add_argument("--output_dir", default="results/alt_embeddings")
    parser.add_argument("--pca_dims", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_embedding_comparison(
        datasets=args.dataset,
        detectors=args.detector,
        embedding_models=args.embedding_models,
        data_dir=Path(args.data_dir),
        processed_dir=Path(args.processed_dir),
        sar_dir=Path(args.sar_dir),
        output_dir=Path(args.output_dir),
        pca_dims=args.pca_dims,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
