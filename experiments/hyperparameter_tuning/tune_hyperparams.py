"""
experiments/hyperparameter_tuning/tune_hyperparams.py
------------------------------------------------------
Hyperparameter Tuning Experiments

We systematically vary key MoSEs hyperparameters and measure their
effect on validation accuracy. This helps us understand:
  1. How robust MoSEs is to its default hyperparameters
  2. Whether the paper's defaults are near-optimal

Parameters explored:
  A. SAR Sinkhorn epsilon (ε): controls routing sharpness
     Range: [0.01, 0.05, 0.1, 0.2, 0.5]
     Paper default: 0.05

  B. SAR epochs: training duration
     Range: [10, 25, 50, 100, 200]
     Paper default: 100

  C. SAR n_prototypes: number of style clusters
     Range: [5, 10, 20, 30, 50]
     Paper default: 10

  D. Neighborhood k: routing neighborhood size
     Range: [5, 10, 20, 50, 100]
     Paper default: 20 (inferred)

  E. CTE LR regularization C: logistic regression strength
     Range: [0.01, 0.1, 1.0, 10.0, 100.0]
     Paper default: 1.0

We use a one-at-a-time (OAT) strategy: vary one parameter while fixing others
at their default values.

Usage:
    python experiments/hyperparameter_tuning/tune_hyperparams.py \
        --dataset CMV \
        --detector roberta \
        --param epsilon \
        --output_dir results/hparam_tuning/
    
    # Run all params
    python experiments/hyperparameter_tuning/tune_hyperparams.py \
        --dataset CMV SciXGen \
        --detector roberta \
        --param all \
        --output_dir results/hparam_tuning/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default hyperparameters (from paper)
DEFAULTS = {
    "epsilon": 0.05,
    "epochs": 100,
    "n_prototypes": 10,
    "neighborhood_k": 20,
    "lr_C": 1.0,
}

# Parameter search grids
PARAM_GRIDS = {
    "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
    "epochs": [10, 25, 50, 100, 200],
    "n_prototypes": [5, 10, 20, 30, 50],
    "neighborhood_k": [5, 10, 20, 50, 100],
    "lr_C": [0.01, 0.1, 1.0, 10.0, 100.0],
}


def run_single_config(
    ref_records: List[Dict],
    test_records: List[Dict],
    epsilon: float,
    epochs: int,
    n_prototypes: int,
    neighborhood_k: int,
    lr_C: float,
    seed: int = 42,
) -> Dict[str, float]:
    """Run one MoSEs configuration and return accuracy dict."""
    from train_sar import StylisticsAwareRouter
    from train_cte import CTELogisticRegression, CTEXGBoost, build_cte_features, fit_static_threshold

    ref_embs = np.stack([r["pca_embedding"] for r in ref_records])
    ref_labels = np.array([r["label"] for r in ref_records])
    ref_scores = np.array([r["score"] for r in ref_records])
    test_labels = np.array([r["label"] for r in test_records])
    test_scores = np.array([r["score"] for r in test_records])

    # Static threshold
    thresh = fit_static_threshold(ref_scores, ref_labels)
    static_acc = accuracy_score(test_labels, (test_scores >= thresh).astype(int))

    # SAR
    sar = StylisticsAwareRouter(
        n_prototypes=min(n_prototypes, len(ref_records) // 5),
        epsilon=epsilon,
        epochs=epochs,
        neighborhood_k=min(neighborhood_k, len(ref_records) // 2),
        seed=seed,
    )
    sar.fit(ref_embs, ref_records)

    # CTE features
    ref_X = np.stack([
        build_cte_features(r, sar.route(r["pca_embedding"])[2])
        for r in ref_records
    ])
    test_X = np.stack([
        build_cte_features(r, sar.route(r["pca_embedding"])[2])
        for r in test_records
    ])

    # CTE-lr
    cte_lr = CTELogisticRegression(C=lr_C, seed=seed)
    cte_lr.fit(ref_X, ref_labels)
    lr_acc = accuracy_score(test_labels, cte_lr.predict(test_X))

    # CTE-xg (doesn't depend on lr_C)
    cte_xg = CTEXGBoost(seed=seed)
    cte_xg.fit(ref_X, ref_labels)
    xg_acc = accuracy_score(test_labels, cte_xg.predict(test_X))

    return {
        "static_acc": float(static_acc),
        "moses_lr_acc": float(lr_acc),
        "moses_xg_acc": float(xg_acc),
    }


def run_oat_tuning(
    param_name: str,
    param_values: List[Any],
    datasets: List[str],
    detectors: List[str],
    processed_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> List[Dict]:
    """One-at-a-time hyperparameter tuning."""
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

            logger.info(f"\n{dataset}/{detector} — sweeping {param_name}")

            for val in param_values:
                # Build config with this param overridden
                cfg = dict(DEFAULTS)
                cfg[param_name] = val

                logger.info(f"  {param_name}={val}")

                try:
                    accs = run_single_config(
                        ref_records=ref_records,
                        test_records=test_records,
                        epsilon=cfg["epsilon"],
                        epochs=cfg["epochs"],
                        n_prototypes=cfg["n_prototypes"],
                        neighborhood_k=cfg["neighborhood_k"],
                        lr_C=cfg["lr_C"],
                        seed=seed,
                    )
                except Exception as e:
                    logger.error(f"  Error: {e}")
                    accs = {"static_acc": float("nan"), "moses_lr_acc": float("nan"), "moses_xg_acc": float("nan")}

                result = {
                    "dataset": dataset,
                    "detector": detector,
                    "param_name": param_name,
                    "param_value": val,
                    "is_default": val == DEFAULTS.get(param_name),
                    **accs,
                }
                all_results.append(result)
                logger.info(f"    lr={accs['moses_lr_acc']:.4f}, xg={accs['moses_xg_acc']:.4f}")

    # Save
    results_path = output_dir / f"hparam_{param_name}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved: {results_path}")

    # Plot
    _plot_hparam_sensitivity(all_results, param_name, param_values, output_dir)
    return all_results


def _plot_hparam_sensitivity(results, param_name, param_values, output_dir):
    """Plot accuracy vs hyperparameter value."""
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
            rs = sorted(ds_results[key], key=lambda x: x["param_value"])
            vals = [r["param_value"] for r in rs]
            lr_accs = [r["moses_lr_acc"] for r in rs]
            xg_accs = [r["moses_xg_acc"] for r in rs]

            use_log = param_name in ("epsilon", "lr_C")
            plot_fn = ax.semilogx if use_log else ax.plot

            plot_fn(vals, lr_accs, "b-o", label="MoSEs-lr", linewidth=2)
            plot_fn(vals, xg_accs, "g-s", label="MoSEs-xg", linewidth=2)

            # Mark paper default
            default_val = DEFAULTS.get(param_name)
            if default_val in vals:
                idx = vals.index(default_val)
                ax.axvline(default_val, color="orange", linestyle=":", alpha=0.8,
                           label=f"Default ({default_val})")

            ax.set_xlabel(param_name)
            ax.set_ylabel("Test Accuracy")
            ax.set_title(f"{key[0]} / {key[1]}")
            ax.legend(fontsize=8)
            ax.set_ylim(0.5, 1.0)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Hyperparameter Sensitivity: {param_name}", fontsize=13)
        plt.tight_layout()
        fig.savefig(output_dir / f"hparam_{param_name}.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["CMV", "SciXGen"])
    parser.add_argument("--detector", nargs="+", default=["roberta"])
    parser.add_argument(
        "--param",
        nargs="+",
        default=["epsilon"],
        choices=list(PARAM_GRIDS.keys()) + ["all"],
    )
    parser.add_argument("--values", nargs="+", type=float, default=None,
                        help="Custom values (overrides grid for single param)")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/hparam_tuning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = args.param
    if "all" in params:
        params = list(PARAM_GRIDS.keys())

    for param in params:
        values = args.values if args.values and len(params) == 1 else PARAM_GRIDS[param]
        logger.info(f"\n{'='*60}")
        logger.info(f"Tuning: {param} over {values}")
        logger.info(f"{'='*60}")

        run_oat_tuning(
            param_name=param,
            param_values=values,
            datasets=args.dataset,
            detectors=args.detector,
            processed_dir=Path(args.processed_dir),
            output_dir=Path(args.output_dir),
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
