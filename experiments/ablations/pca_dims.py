"""
experiments/ablations/pca_dims.py
----------------------------------
Ablation Study 2: Effect of PCA Dimensionality

Research question:
  The paper uses 32D PCA projection of BGE-M3 (1024D) embeddings.
  How sensitive is MoSEs to this dimensionality?
  Is 32D sufficient, or would higher dimensions improve performance?

We test: [8, 16, 32, 64, 128, 256] PCA dimensions.
Since PCA projection happens during preprocessing, this requires
re-running preprocessing with different pca_dims.
To save compute, we refit PCA from already-computed full embeddings
stored in the reference pickle files.

Usage:
    python experiments/ablations/pca_dims.py \
        --dataset CMV SciXGen WP XSum \
        --detector roberta \
        --pca_dims 8 16 32 64 128 256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def reproject_with_pca(records: List[Dict], pca: PCA) -> List[Dict]:
    """Replace pca_embedding in each record with a new projection."""
    updated = []
    for r in records:
        r_copy = dict(r)
        r_copy["pca_embedding"] = pca.transform(r["embedding"].reshape(1, -1))[0].astype(np.float32)
        updated.append(r_copy)
    return updated


def run_pca_dims_ablation(
    datasets: List[str],
    detectors: List[str],
    pca_dims_list: List[int],
    processed_dir: Path,
    output_dir: Path,
    sar_epochs: int = 100,
    sar_epsilon: float = 0.05,
    neighborhood_k: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """
    For each PCA dimension, refit PCA from full embeddings and re-run MoSEs.
    """
    from train_sar import StylisticsAwareRouter
    from train_cte import CTELogisticRegression, CTEXGBoost, build_cte_features, fit_static_threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for dataset in datasets:
        for detector in detectors:
            suffix = f"{dataset}_{detector}_bge-m3"
            ref_path = processed_dir / f"{suffix}_ref.pkl"
            test_path = processed_dir / f"{suffix}_test.pkl"

            if not ref_path.exists() or not test_path.exists():
                logger.warning(f"Skipping {dataset}/{detector}: data not found")
                continue

            with open(ref_path, "rb") as f:
                ref_data = pickle.load(f)
            with open(test_path, "rb") as f:
                test_data = pickle.load(f)

            ref_records_orig = ref_data["records"]
            test_records_orig = test_data["records"]
            test_labels = np.array([r["label"] for r in test_records_orig])
            test_scores = np.array([r["score"] for r in test_records_orig])
            ref_labels = np.array([r["label"] for r in ref_records_orig])

            # Check if full embeddings are stored
            if "embedding" not in ref_records_orig[0]:
                logger.warning(
                    f"Full embeddings not stored in {ref_path}. "
                    "Re-run preprocessing with full embeddings saved."
                )
                continue

            # Get full-dim embeddings
            ref_full_embs = np.stack([r["embedding"] for r in ref_records_orig])
            test_full_embs = np.stack([r["embedding"] for r in test_records_orig])
            max_dims = ref_full_embs.shape[1]

            for pca_dim in pca_dims_list:
                if pca_dim >= max_dims:
                    logger.warning(f"  pca_dim={pca_dim} >= embedding_dim={max_dims}, skipping")
                    continue

                logger.info(f"  {dataset}/{detector} pca_dim={pca_dim}")

                # Fit PCA on reference set
                pca = PCA(n_components=pca_dim, random_state=seed)
                ref_pca = pca.fit_transform(ref_full_embs).astype(np.float32)
                test_pca = pca.transform(test_full_embs).astype(np.float32)

                # Update records with new PCA embeddings
                ref_records = [dict(r, pca_embedding=ref_pca[i]) for i, r in enumerate(ref_records_orig)]
                test_records = [dict(r, pca_embedding=test_pca[i]) for i, r in enumerate(test_records_orig)]

                # Static threshold baseline
                ref_scores = np.array([r["score"] for r in ref_records])
                thresh = fit_static_threshold(ref_scores, ref_labels)
                static_preds = (test_scores >= thresh).astype(int)
                static_acc = accuracy_score(test_labels, static_preds)

                # Train SAR
                sar = StylisticsAwareRouter(
                    n_prototypes=10,
                    epsilon=sar_epsilon,
                    epochs=sar_epochs,
                    neighborhood_k=neighborhood_k,
                    seed=seed,
                )
                sar.fit(ref_pca, ref_records)

                # Build CTE features
                ref_X = np.stack([
                    build_cte_features(r, sar.route(r["pca_embedding"])[2])
                    for r in ref_records
                ])
                test_X = np.stack([
                    build_cte_features(r, sar.route(r["pca_embedding"])[2])
                    for r in test_records
                ])

                # CTE-lr
                cte_lr = CTELogisticRegression(seed=seed)
                cte_lr.fit(ref_X, ref_labels)
                lr_acc = accuracy_score(test_labels, cte_lr.predict(test_X))

                # CTE-xg
                cte_xg = CTEXGBoost(seed=seed)
                cte_xg.fit(ref_X, ref_labels)
                xg_acc = accuracy_score(test_labels, cte_xg.predict(test_X))

                # Explained variance
                explained_var = float(np.sum(pca.explained_variance_ratio_))

                result = {
                    "dataset": dataset,
                    "detector": detector,
                    "pca_dim": pca_dim,
                    "explained_variance_ratio": explained_var,
                    "static_acc": float(static_acc),
                    "moses_lr_acc": float(lr_acc),
                    "moses_xg_acc": float(xg_acc),
                }
                all_results.append(result)
                logger.info(
                    f"    pca={pca_dim}d ({explained_var:.1%} var), "
                    f"lr={lr_acc:.4f}, xg={xg_acc:.4f}"
                )

    results_path = output_dir / "pca_dims_ablation.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved: {results_path}")

    _plot_pca_dims(all_results, pca_dims_list, output_dir)
    return all_results


def _plot_pca_dims(results, pca_dims_list, output_dir):
    """Plot accuracy vs PCA dimensions."""
    try:
        from collections import defaultdict
        ds_results = defaultdict(list)
        for r in results:
            ds_results[r["dataset"]].append(r)

        datasets = list(ds_results.keys())
        fig, axes = plt.subplots(1, max(len(datasets), 1), figsize=(5 * len(datasets), 4), sharey=True)
        if len(datasets) == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            rs = sorted(ds_results[dataset], key=lambda x: x["pca_dim"])
            dims = [r["pca_dim"] for r in rs]
            lr_accs = [r["moses_lr_acc"] for r in rs]
            xg_accs = [r["moses_xg_acc"] for r in rs]
            static_accs = [r["static_acc"] for r in rs]

            ax.semilogx(dims, lr_accs, "b-o", label="MoSEs-lr", linewidth=2, base=2)
            ax.semilogx(dims, xg_accs, "g-s", label="MoSEs-xg", linewidth=2, base=2)
            ax.axhline(static_accs[0], color="red", linestyle="--", alpha=0.7, label="Static threshold")
            ax.set_xlabel("PCA dimensions (log scale)")
            ax.set_ylabel("Test Accuracy")
            ax.set_title(f"{dataset}")
            ax.legend(fontsize=8)
            ax.set_ylim(0.5, 1.0)
            ax.grid(True, alpha=0.3, which="both")
            # Mark paper default
            if 32 in dims:
                idx = dims.index(32)
                ax.axvline(32, color="orange", linestyle=":", alpha=0.8, label="Paper default (32)")

        plt.suptitle("Effect of PCA Dimensionality on MoSEs Accuracy", fontsize=13)
        plt.tight_layout()
        fig.savefig(output_dir / "pca_dims_ablation.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["CMV", "SciXGen", "WP", "XSum"])
    parser.add_argument("--detector", nargs="+", default=["roberta"])
    parser.add_argument("--pca_dims", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/ablations/pca_dims")
    parser.add_argument("--sar_epochs", type=int, default=100)
    parser.add_argument("--sar_epsilon", type=float, default=0.05)
    parser.add_argument("--neighborhood_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pca_dims_ablation(
        datasets=args.dataset,
        detectors=args.detector,
        pca_dims_list=args.pca_dims,
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        sar_epochs=args.sar_epochs,
        sar_epsilon=args.sar_epsilon,
        neighborhood_k=args.neighborhood_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
