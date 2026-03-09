"""
notebooks/analysis.py
----------------------
Result visualization and analysis script.
Run after completing all experiments to generate paper-quality tables and figures.

Usage:
    python notebooks/analysis.py --results_dir results/ --output_dir figures/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAIN_DATASETS = ["CMV", "SciXGen", "WP", "XSum"]
LOW_DATASETS = ["CNN", "DialogSum", "IMDB", "PubMedQA"]
DETECTORS = ["roberta", "fastdetectgpt", "lastde"]
DETECTOR_LABELS = {
    "roberta": "RoBERTa",
    "fastdetectgpt": "Fast-DetectGPT",
    "lastde": "Lastde",
}
METHOD_COLORS = {
    "static_threshold": "#E74C3C",
    "nearest_voting": "#F39C12",
    "moses_lr": "#2ECC71",
    "moses_xg": "#3498DB",
}
METHOD_LABELS = {
    "static_threshold": "Static Thresh.",
    "nearest_voting": "Nearest Voting",
    "moses_lr": "MoSEs-lr",
    "moses_xg": "MoSEs-xg",
}


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load all *_results.json files."""
    results = []
    for f in sorted(results_dir.glob("*_results.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def print_latex_table(results: List[Dict], datasets: List[str], detector: str):
    """Print LaTeX-formatted accuracy table for a given detector."""
    methods = ["static_threshold", "nearest_voting", "moses_lr", "moses_xg"]
    method_names = ["Static", "Nearest Vote", "MoSEs-lr", "MoSEs-xg"]

    result_map = {}
    for r in results:
        if r["detector"] == detector:
            result_map[r["dataset"]] = r

    print(f"\n% Results for {DETECTOR_LABELS.get(detector, detector)}")
    print("\\begin{tabular}{l" + "r" * len(methods) + "}")
    print("\\toprule")
    print("Dataset & " + " & ".join(method_names) + " \\\\")
    print("\\midrule")

    for ds in datasets:
        r = result_map.get(ds, {})
        row = [ds]
        for m in methods:
            acc = r.get(m, {}).get("accuracy", float("nan"))
            if not np.isnan(acc):
                row.append(f"{acc:.4f}")
            else:
                row.append("--")
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")

    # Average row
    print("\n% Average:")
    avg_row = ["Avg"]
    for m in methods:
        accs = [result_map[ds].get(m, {}).get("accuracy", float("nan"))
                for ds in datasets if ds in result_map]
        accs = [a for a in accs if not np.isnan(a)]
        avg_row.append(f"{np.mean(accs):.4f}" if accs else "--")
    print(" & ".join(avg_row) + " \\\\")


def plot_main_results(results: List[Dict], output_dir: Path):
    """Create main results figure: grouped bar chart per detector."""
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["static_threshold", "nearest_voting", "moses_lr", "moses_xg"]
    n_methods = len(methods)
    x = np.arange(len(MAIN_DATASETS))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, detector in zip(axes, DETECTORS):
        result_map = {r["dataset"]: r for r in results if r["detector"] == detector}

        for i, method in enumerate(methods):
            accs = [
                result_map.get(ds, {}).get(method, {}).get("accuracy", float("nan"))
                for ds in MAIN_DATASETS
            ]
            offset = (i - n_methods / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                accs,
                width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(MAIN_DATASETS, rotation=15, ha="right")
        ax.set_ylabel("Test Accuracy" if ax == axes[0] else "")
        ax.set_title(DETECTOR_LABELS[detector])
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("MoSEs vs Baselines — Main Datasets", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "main_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'main_results.png'}")


def plot_improvement_heatmap(results: List[Dict], output_dir: Path):
    """Heatmap showing MoSEs-lr improvement over static threshold."""
    all_datasets = MAIN_DATASETS + LOW_DATASETS
    improvements = np.full((len(DETECTORS), len(all_datasets)), np.nan)

    for i, det in enumerate(DETECTORS):
        for j, ds in enumerate(all_datasets):
            r = next((r for r in results if r["dataset"] == ds and r["detector"] == det), None)
            if r:
                static = r.get("static_threshold", {}).get("accuracy", np.nan)
                lr = r.get("moses_lr", {}).get("accuracy", np.nan)
                if not np.isnan(static) and not np.isnan(lr):
                    improvements[i, j] = lr - static

    fig, ax = plt.subplots(figsize=(14, 4))
    mask = np.isnan(improvements)
    sns.heatmap(
        improvements,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        xticklabels=all_datasets,
        yticklabels=[DETECTOR_LABELS[d] for d in DETECTORS],
        ax=ax,
        mask=mask,
        vmin=-0.05,
        vmax=0.15,
    )
    ax.set_title("MoSEs-lr Accuracy Improvement over Static Threshold\n(green=improvement, red=degradation)")

    # Add divider between main and low-resource
    ax.axvline(len(MAIN_DATASETS), color="black", linewidth=2)
    ax.text(len(MAIN_DATASETS) / 2, -0.5, "Main Datasets", ha="center", fontsize=9)
    ax.text(len(MAIN_DATASETS) + len(LOW_DATASETS) / 2, -0.5, "Low-Resource", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "improvement_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'improvement_heatmap.png'}")


def plot_ablation_summary(results_dir: Path, output_dir: Path):
    """Summary figure for all ablations."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

    # (1) Training size
    training_path = results_dir / "ablations/training_size/training_size_ablation.json"
    if training_path.exists():
        with open(training_path) as f:
            ts_data = json.load(f)
        # Average across datasets
        from collections import defaultdict
        sz_map = defaultdict(list)
        for r in ts_data:
            sz_map[r["size_fraction"]].append(r["moses_lr_acc"])
        sizes = sorted(sz_map.keys())
        means = [np.mean(sz_map[s]) for s in sizes]
        axes[0].plot([s * 100 for s in sizes], means, "b-o", linewidth=2)
        axes[0].set_xlabel("Reference set size (%)")
        axes[0].set_ylabel("Avg. Accuracy")
        axes[0].set_title("Effect of Training Set Size")
        axes[0].set_ylim(0.5, 1.0)
        axes[0].grid(True, alpha=0.3)

    # (2) PCA dimensions
    pca_path = results_dir / "ablations/pca_dims/pca_dims_ablation.json"
    if pca_path.exists():
        with open(pca_path) as f:
            pca_data = json.load(f)
        from collections import defaultdict
        dim_map = defaultdict(list)
        for r in pca_data:
            dim_map[r["pca_dim"]].append(r["moses_lr_acc"])
        dims = sorted(dim_map.keys())
        means = [np.mean(dim_map[d]) for d in dims]
        axes[1].semilogx(dims, means, "g-s", linewidth=2, base=2)
        axes[1].axvline(32, color="orange", linestyle=":", alpha=0.8, label="Default (32)")
        axes[1].set_xlabel("PCA dimensions")
        axes[1].set_ylabel("Avg. Accuracy")
        axes[1].set_title("Effect of PCA Dimensions")
        axes[1].set_ylim(0.5, 1.0)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which="both")

    # (3) Neighborhood k
    k_path = results_dir / "ablations/neighborhood_k/neighborhood_k_ablation.json"
    if k_path.exists():
        with open(k_path) as f:
            k_data = json.load(f)
        from collections import defaultdict
        k_map = defaultdict(list)
        no_route_accs = []
        for r in k_data:
            k_map[r["k"]].append(r["moses_lr_acc"])
            no_route_accs.append(r["no_routing_lr_acc"])
        ks = sorted(k_map.keys())
        means = [np.mean(k_map[k]) for k in ks]
        axes[2].plot(ks, means, "r-o", linewidth=2, label="Routed (MoSEs-lr)")
        if no_route_accs:
            axes[2].axhline(np.mean(no_route_accs), color="purple", linestyle="-.",
                            alpha=0.8, label="No routing")
        axes[2].axvline(20, color="orange", linestyle=":", alpha=0.8, label="Default k=20")
        axes[2].set_xlabel("Neighborhood k")
        axes[2].set_ylabel("Avg. Accuracy")
        axes[2].set_title("Effect of Neighborhood Size k")
        axes[2].set_ylim(0.5, 1.0)
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

    # (4) Hyperparameter: epsilon
    eps_path = results_dir / "hparam_tuning/hparam_epsilon.json"
    if eps_path.exists():
        with open(eps_path) as f:
            eps_data = json.load(f)
        from collections import defaultdict
        eps_map = defaultdict(list)
        for r in eps_data:
            eps_map[r["param_value"]].append(r["moses_lr_acc"])
        epsilons = sorted(eps_map.keys())
        means = [np.mean(eps_map[e]) for e in epsilons]
        axes[3].semilogx(epsilons, means, "m-^", linewidth=2)
        axes[3].axvline(0.05, color="orange", linestyle=":", alpha=0.8, label="Default ε=0.05")
        axes[3].set_xlabel("Sinkhorn ε")
        axes[3].set_ylabel("Avg. Accuracy")
        axes[3].set_title("Sensitivity to Sinkhorn ε")
        axes[3].set_ylim(0.5, 1.0)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3, which="both")

    # (5) Hyperparameter: n_prototypes
    nproto_path = results_dir / "hparam_tuning/hparam_n_prototypes.json"
    if nproto_path.exists():
        with open(nproto_path) as f:
            np_data = json.load(f)
        from collections import defaultdict
        np_map = defaultdict(list)
        for r in np_data:
            np_map[r["param_value"]].append(r["moses_lr_acc"])
        nps = sorted(np_map.keys())
        means = [np.mean(np_map[n]) for n in nps]
        axes[4].plot(nps, means, "c-D", linewidth=2)
        axes[4].axvline(10, color="orange", linestyle=":", alpha=0.8, label="Default K=10")
        axes[4].set_xlabel("Number of prototypes K")
        axes[4].set_ylabel("Avg. Accuracy")
        axes[4].set_title("Sensitivity to # Prototypes")
        axes[4].set_ylim(0.5, 1.0)
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    # (6) Alternative embeddings
    emb_path = results_dir / "alt_embeddings/alt_embeddings_results.json"
    if emb_path.exists():
        with open(emb_path) as f:
            emb_data = json.load(f)
        from collections import defaultdict
        emb_map = defaultdict(list)
        for r in emb_data:
            lr_acc = r.get("moses_lr", {}).get("accuracy")
            if lr_acc is not None:
                emb_map[r["embedding_model"]].append(lr_acc)
        models = list(emb_map.keys())
        means = [np.mean(emb_map[m]) for m in models]
        axes[5].bar(models, means, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(models)], alpha=0.8)
        axes[5].set_xlabel("Embedding Model")
        axes[5].set_ylabel("Avg. Accuracy (MoSEs-lr)")
        axes[5].set_title("Embedding Model Comparison")
        axes[5].set_ylim(0.5, 1.0)
        axes[5].tick_params(axis="x", rotation=15)
        axes[5].grid(True, alpha=0.3, axis="y")

    plt.suptitle("MoSEs Ablation & Sensitivity Analysis", fontsize=15)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'ablation_summary.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load main results
    results = load_all_results(results_dir)

    if results:
        logger.info(f"Loaded {len(results)} result files.")

        # LaTeX tables
        for detector in DETECTORS:
            print_latex_table(results, MAIN_DATASETS, detector)
            print_latex_table(results, LOW_DATASETS, detector)

        # Figures
        plot_main_results(results, output_dir)
        plot_improvement_heatmap(results, output_dir)
    else:
        logger.warning("No main results found. Run the pipeline first.")

    # Ablation plots
    plot_ablation_summary(results_dir, output_dir)

    logger.info(f"Analysis complete. Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
