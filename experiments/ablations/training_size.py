"""
experiments/ablations/training_size.py
---------------------------------------
Ablation Study 1: Effect of Reference Set Size

Research question:
  How does the size of the reference set affect MoSEs performance?
  This is critical for the low-resource setting where only 200 reference
  samples are available vs. 1800 in the main setting.

We subsample the reference set to fractions: [0.10, 0.25, 0.50, 0.75, 1.00]
and train SAR + CTE at each size. The test set is always the full held-out set.

This tests Claim H3: whether MoSEs is effective even with small reference sets.
High accuracy at small reference sizes would support the practical utility claim.

Usage:
    python experiments/ablations/training_size.py \
        --dataset CMV SciXGen WP XSum \
        --detector roberta \
        --sizes 0.1 0.25 0.5 0.75 1.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

import json
import pickle
import argparse
import logging
import time
import copy
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_training_size_ablation(
    datasets: List[str],
    detectors: List[str],
    sizes: List[float],
    processed_dir: Path,
    output_dir: Path,
    pca_dims: int = 32,
    sar_epochs: int = 100,
    sar_epsilon: float = 0.05,
    neighborhood_k: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """
    For each (dataset, detector, size), subsample reference set and run MoSEs.
    """
    from train_sar import StylisticsAwareRouter
    from train_cte import CTELogisticRegression, CTEXGBoost, build_cte_features

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
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

            all_ref_records = ref_data["records"]
            test_records = test_data["records"]
            test_labels = np.array([r["label"] for r in test_records])
            test_embs = np.stack([r["pca_embedding"] for r in test_records])
            test_scores = np.array([r["score"] for r in test_records])

            for size_frac in sizes:
                n_samples = max(10, int(len(all_ref_records) * size_frac))
                # Stratified subsample to maintain class balance
                ai_indices = [i for i, r in enumerate(all_ref_records) if r["label"] == 1]
                human_indices = [i for i, r in enumerate(all_ref_records) if r["label"] == 0]

                n_ai = min(n_samples // 2, len(ai_indices))
                n_human = min(n_samples - n_ai, len(human_indices))
                n_ai = n_samples - n_human  # rebalance

                sampled_ai = rng.choice(ai_indices, size=min(n_ai, len(ai_indices)), replace=False)
                sampled_human = rng.choice(human_indices, size=min(n_human, len(human_indices)), replace=False)
                sampled_indices = np.concatenate([sampled_ai, sampled_human])
                rng.shuffle(sampled_indices)

                sub_records = [all_ref_records[i] for i in sampled_indices]
                sub_embs = np.stack([r["pca_embedding"] for r in sub_records])
                sub_labels = np.array([r["label"] for r in sub_records])
                sub_scores = np.array([r["score"] for r in sub_records])

                logger.info(f"  {dataset}/{detector} size={size_frac:.0%} n={len(sub_records)}")

                # --- Static threshold baseline ---
                from train_cte import fit_static_threshold
                thresh = fit_static_threshold(sub_scores, sub_labels)
                static_preds = (test_scores >= thresh).astype(int)
                static_acc = accuracy_score(test_labels, static_preds)

                # --- Train SAR on subsampled reference ---
                sar = StylisticsAwareRouter(
                    n_prototypes=min(10, len(sub_records) // 5),
                    epsilon=sar_epsilon,
                    epochs=sar_epochs,
                    neighborhood_k=min(neighborhood_k, len(sub_records) // 2),
                    seed=seed,
                )
                sar.fit(sub_embs, sub_records)

                # Build CTE features
                ref_cte_feats = []
                for record in sub_records:
                    _, _, nbrs = sar.route(record["pca_embedding"])
                    feat = build_cte_features(record, nbrs)
                    ref_cte_feats.append(feat)
                ref_X = np.stack(ref_cte_feats)

                test_cte_feats = []
                for record in test_records:
                    _, _, nbrs = sar.route(record["pca_embedding"])
                    feat = build_cte_features(record, nbrs)
                    test_cte_feats.append(feat)
                test_X = np.stack(test_cte_feats)

                # Train CTE-lr
                cte_lr = CTELogisticRegression(seed=seed)
                cte_lr.fit(ref_X, sub_labels)
                lr_preds = cte_lr.predict(test_X)
                lr_acc = accuracy_score(test_labels, lr_preds)

                # Train CTE-xg
                cte_xg = CTEXGBoost(seed=seed)
                cte_xg.fit(ref_X, sub_labels)
                xg_preds = cte_xg.predict(test_X)
                xg_acc = accuracy_score(test_labels, xg_preds)

                result = {
                    "dataset": dataset,
                    "detector": detector,
                    "size_fraction": size_frac,
                    "n_ref_samples": len(sub_records),
                    "static_acc": float(static_acc),
                    "moses_lr_acc": float(lr_acc),
                    "moses_xg_acc": float(xg_acc),
                }
                all_results.append(result)
                logger.info(
                    f"    static={static_acc:.4f}, moses_lr={lr_acc:.4f}, moses_xg={xg_acc:.4f}"
                )

    # Save results
    results_path = output_dir / "training_size_ablation.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved: {results_path}")

    # Plot
    _plot_training_size(all_results, sizes, output_dir)

    return all_results


def _plot_training_size(results, sizes, output_dir):
    """Plot accuracy vs reference set size for each dataset."""
    try:
        from collections import defaultdict

        # Group by dataset
        ds_results = defaultdict(lambda: defaultdict(list))
        for r in results:
            ds = r["dataset"]
            sz = r["size_fraction"]
            ds_results[ds][sz].append(r)

        datasets = list(ds_results.keys())
        n_ds = len(datasets)
        fig, axes = plt.subplots(1, max(n_ds, 1), figsize=(5 * n_ds, 4), sharey=True)
        if n_ds == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            sz_vals = sorted(sizes)
            lr_accs = []
            xg_accs = []
            static_accs = []

            for sz in sz_vals:
                rs = ds_results[dataset].get(sz, [])
                if rs:
                    lr_accs.append(np.mean([r["moses_lr_acc"] for r in rs]))
                    xg_accs.append(np.mean([r["moses_xg_acc"] for r in rs]))
                    static_accs.append(np.mean([r["static_acc"] for r in rs]))
                else:
                    lr_accs.append(float("nan"))
                    xg_accs.append(float("nan"))
                    static_accs.append(float("nan"))

            x = [s * 100 for s in sz_vals]
            ax.plot(x, lr_accs, "b-o", label="MoSEs-lr", linewidth=2)
            ax.plot(x, xg_accs, "g-s", label="MoSEs-xg", linewidth=2)
            ax.plot(x, static_accs, "r--^", label="Static threshold", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Reference set size (%)")
            ax.set_ylabel("Test Accuracy")
            ax.set_title(f"{dataset}")
            ax.legend(fontsize=8)
            ax.set_ylim(0.5, 1.0)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Effect of Reference Set Size on MoSEs Accuracy", fontsize=13)
        plt.tight_layout()
        fig.savefig(output_dir / "training_size_ablation.png", dpi=150)
        plt.close()
        logger.info(f"Plot saved: {output_dir / 'training_size_ablation.png'}")
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["CMV", "SciXGen", "WP", "XSum"])
    parser.add_argument("--detector", nargs="+", default=["roberta"])
    parser.add_argument("--sizes", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/ablations/training_size")
    parser.add_argument("--pca_dims", type=int, default=32)
    parser.add_argument("--sar_epochs", type=int, default=100)
    parser.add_argument("--sar_epsilon", type=float, default=0.05)
    parser.add_argument("--neighborhood_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training_size_ablation(
        datasets=args.dataset,
        detectors=args.detector,
        sizes=args.sizes,
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        sar_epochs=args.sar_epochs,
        sar_epsilon=args.sar_epsilon,
        neighborhood_k=args.neighborhood_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
