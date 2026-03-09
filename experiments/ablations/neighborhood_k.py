"""
experiments/ablations/neighborhood_k.py
-----------------------------------------
Ablation Study 3: Effect of SAR Neighborhood Size k

Research question:
  MoSEs routes each test sample to k nearest reference neighbors.
  Larger k means more stable estimates but less style-specific adaptation.
  Smaller k means more precise but potentially noisier estimates.

We test k ∈ [5, 10, 20, 50, 100, 200] while holding all other params fixed.

This ablation tests Claim H4 (routing matters): if MoSEs is
significantly better at a specific k than at k=N (full pool, equivalent
to no routing), then the routing mechanism is genuinely contributing.

Usage:
    python experiments/ablations/neighborhood_k.py \
        --dataset CMV SciXGen WP XSum \
        --detector roberta \
        --k_values 5 10 20 50 100 200
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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_neighborhood_k_ablation(
    datasets: List[str],
    detectors: List[str],
    k_values: List[int],
    processed_dir: Path,
    output_dir: Path,
    sar_epochs: int = 100,
    sar_epsilon: float = 0.05,
    seed: int = 42,
) -> List[Dict]:
    """
    For each k, re-route the test samples and re-train CTE.
    SAR prototypes are trained once (with max k).
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

            ref_records = ref_data["records"]
            test_records = test_data["records"]
            ref_labels = np.array([r["label"] for r in ref_records])
            test_labels = np.array([r["label"] for r in test_records])
            ref_scores = np.array([r["score"] for r in ref_records])
            test_scores = np.array([r["score"] for r in test_records])

            # Train SAR once with max k
            max_k = max(k_values)
            sar = StylisticsAwareRouter(
                n_prototypes=10,
                epsilon=sar_epsilon,
                epochs=sar_epochs,
                neighborhood_k=max_k,
                seed=seed,
            )
            ref_embs = np.stack([r["pca_embedding"] for r in ref_records])
            sar.fit(ref_embs, ref_records)

            # Static threshold baseline (k-independent)
            thresh = fit_static_threshold(ref_scores, ref_labels)
            static_acc = accuracy_score(test_labels, (test_scores >= thresh).astype(int))

            # No-routing baseline: use all reference samples as neighborhood
            no_route_ref_X = np.stack([
                build_cte_features(r, ref_records)  # use all ref as neighbors
                for r in ref_records
            ])
            no_route_test_X = np.stack([
                build_cte_features(r, ref_records)
                for r in test_records
            ])
            cte_lr_nr = CTELogisticRegression(seed=seed)
            cte_lr_nr.fit(no_route_ref_X, ref_labels)
            no_route_acc = accuracy_score(test_labels, cte_lr_nr.predict(no_route_test_X))
            logger.info(f"  {dataset}/{detector} no-routing lr acc: {no_route_acc:.4f}")

            for k in k_values:
                actual_k = min(k, len(ref_records))
                logger.info(f"  {dataset}/{detector} k={k}")

                # Override SAR's k by patching neighborhood_k
                sar.neighborhood_k = actual_k

                ref_X = np.stack([
                    build_cte_features(r, sar.route(r["pca_embedding"])[2])
                    for r in ref_records
                ])
                test_X = np.stack([
                    build_cte_features(r, sar.route(r["pca_embedding"])[2])
                    for r in test_records
                ])

                cte_lr = CTELogisticRegression(seed=seed)
                cte_lr.fit(ref_X, ref_labels)
                lr_acc = accuracy_score(test_labels, cte_lr.predict(test_X))

                cte_xg = CTEXGBoost(seed=seed)
                cte_xg.fit(ref_X, ref_labels)
                xg_acc = accuracy_score(test_labels, cte_xg.predict(test_X))

                result = {
                    "dataset": dataset,
                    "detector": detector,
                    "k": k,
                    "actual_k": actual_k,
                    "static_acc": float(static_acc),
                    "no_routing_lr_acc": float(no_route_acc),
                    "moses_lr_acc": float(lr_acc),
                    "moses_xg_acc": float(xg_acc),
                }
                all_results.append(result)
                logger.info(f"    lr={lr_acc:.4f}, xg={xg_acc:.4f}")

    results_path = output_dir / "neighborhood_k_ablation.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved: {results_path}")

    _plot_neighborhood_k(all_results, k_values, output_dir)
    return all_results


def _plot_neighborhood_k(results, k_values, output_dir):
    """Plot accuracy vs k."""
    try:
        from collections import defaultdict
        ds_results = defaultdict(list)
        for r in results:
            ds_results[(r["dataset"], r["detector"])].append(r)

        keys = list(ds_results.keys())
        fig, axes = plt.subplots(1, max(len(keys), 1), figsize=(5 * len(keys), 4), sharey=True)
        if len(keys) == 1:
            axes = [axes]

        for ax, key in zip(axes, keys):
            rs = sorted(ds_results[key], key=lambda x: x["k"])
            ks = [r["k"] for r in rs]
            lr_accs = [r["moses_lr_acc"] for r in rs]
            xg_accs = [r["moses_xg_acc"] for r in rs]
            no_route_acc = rs[0]["no_routing_lr_acc"]  # same for all k
            static_acc = rs[0]["static_acc"]

            ax.plot(ks, lr_accs, "b-o", label="MoSEs-lr (routed)", linewidth=2)
            ax.plot(ks, xg_accs, "g-s", label="MoSEs-xg (routed)", linewidth=2)
            ax.axhline(no_route_acc, color="purple", linestyle="-.", alpha=0.8, label="No routing (lr)")
            ax.axhline(static_acc, color="red", linestyle="--", alpha=0.7, label="Static threshold")
            ax.set_xlabel("Neighborhood size k")
            ax.set_ylabel("Test Accuracy")
            ax.set_title(f"{key[0]} / {key[1]}")
            ax.legend(fontsize=8)
            ax.set_ylim(0.5, 1.0)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Effect of SAR Neighborhood Size k", fontsize=13)
        plt.tight_layout()
        fig.savefig(output_dir / "neighborhood_k_ablation.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["CMV", "SciXGen", "WP", "XSum"])
    parser.add_argument("--detector", nargs="+", default=["roberta"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 20, 50, 100, 200])
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/ablations/neighborhood_k")
    parser.add_argument("--sar_epochs", type=int, default=100)
    parser.add_argument("--sar_epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_neighborhood_k_ablation(
        datasets=args.dataset,
        detectors=args.detector,
        k_values=args.k_values,
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        sar_epochs=args.sar_epochs,
        sar_epsilon=args.sar_epsilon,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
