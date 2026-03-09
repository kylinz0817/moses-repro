"""
experiments/additional_datasets/run_additional.py
--------------------------------------------------
Additional Experiment 1: Evaluate MoSEs on new datasets not in the original paper.

Datasets:
  - HC3 (Human-ChatGPT Comparison Corpus): real-world human vs ChatGPT answers
  - TruthfulQA: correct vs hallucinated answers (proxy for AI-generated)

Motivation:
  The original paper evaluates on 8 datasets. We ask: does MoSEs generalize
  to newer, out-of-distribution data? HC3 is particularly interesting because
  it contains real ChatGPT responses (not sampled from a single generator model),
  which may have different stylistic properties than the original training distribution.

Usage:
    python experiments/additional_datasets/run_additional.py \
        --detector roberta --output_dir results/additional/
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
from sklearn.metrics import accuracy_score, classification_report

from preprocess import (
    preprocess_dataset,
    EMBEDDING_MODEL_NAMES,
)
from train_sar import StylisticsAwareRouter, train_sar_for_dataset
from train_cte import evaluate_dataset, summarize_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ADDITIONAL_DATASETS = ["HC3", "TruthfulQA"]
DETECTORS = ["roberta", "fastdetectgpt", "lastde"]


def run_additional_experiments(
    datasets: List[str],
    detectors: List[str],
    data_dir: Path,
    processed_dir: Path,
    sar_dir: Path,
    output_dir: Path,
    pca_dims: int = 32,
    device: str = "cuda",
    fp16: bool = False,
    seed: int = 42,
):
    """
    Run the full MoSEs pipeline on additional datasets and report results.
    
    Also runs cross-dataset transfer: train SAR/CTE on original datasets,
    evaluate on additional datasets (zero-shot transfer).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # ---- Part 1: Train and evaluate on additional datasets directly ----
    logger.info("=" * 60)
    logger.info("Part 1: Direct evaluation on additional datasets")
    logger.info("=" * 60)

    for dataset in datasets:
        for detector in detectors:
            logger.info(f"\n--- {dataset} / {detector} ---")

            # Preprocess
            try:
                preprocess_dataset(
                    dataset=dataset,
                    detector=detector,
                    data_dir=data_dir,
                    output_dir=processed_dir,
                    pca_dims=pca_dims,
                    device=device,
                    fp16=fp16,
                    seed=seed,
                )
            except FileNotFoundError as e:
                logger.warning(f"Skipping preprocessing: {e}")
                continue

            # Train SAR
            try:
                train_sar_for_dataset(
                    dataset=dataset,
                    detector=detector,
                    processed_dir=processed_dir,
                    output_dir=sar_dir,
                    epochs=100,
                    epsilon=0.05,
                    neighborhood_k=20,
                    seed=seed,
                )
            except Exception as e:
                logger.error(f"SAR training error: {e}")

            # Evaluate
            try:
                r = evaluate_dataset(
                    dataset=dataset,
                    detector=detector,
                    processed_dir=processed_dir,
                    sar_dir=sar_dir,
                    output_dir=output_dir / "direct",
                    cte_types=["lr", "xgb"],
                    seed=seed,
                )
                r["experiment"] = "direct"
                all_results.append(r)
            except Exception as e:
                logger.error(f"Evaluation error: {e}")

    # ---- Part 2: Zero-shot transfer from CMV (domain generalization) ----
    logger.info("\n" + "=" * 60)
    logger.info("Part 2: Zero-shot transfer (CMV SAR -> additional datasets)")
    logger.info("=" * 60)

    source_dataset = "CMV"  # Use CMV as source for transfer
    detector = detectors[0]  # Use first detector for transfer experiment

    source_suffix = f"{source_dataset}_{detector}_bge-m3"
    source_sar_path = sar_dir / f"{source_suffix}_sar.pkl"

    if source_sar_path.exists():
        source_sar = StylisticsAwareRouter.load(source_sar_path)

        for dataset in datasets:
            test_suffix = f"{dataset}_{detector}_bge-m3"
            test_path = processed_dir / f"{test_suffix}_test.pkl"

            if not test_path.exists():
                logger.warning(f"Test data not found for transfer: {test_path}")
                continue

            with open(test_path, "rb") as f:
                test_data = pickle.load(f)
            test_records = test_data["records"]
            test_labels = np.array([r["label"] for r in test_records])

            # Route using source SAR, classify with nearest voting
            from train_cte import build_cte_features
            preds = []
            for record in test_records:
                _, _, neighbors = source_sar.route(record["pca_embedding"])
                # Majority vote from neighbors
                if neighbors:
                    nbr_labels = np.array([n["label"] for n in neighbors])
                    pred = int(np.round(nbr_labels.mean()))
                else:
                    pred = int(record["score"] > 0.5)
                preds.append(pred)

            transfer_acc = accuracy_score(test_labels, preds)
            logger.info(f"  Zero-shot transfer {source_dataset}→{dataset}: acc={transfer_acc:.4f}")

            r = {
                "dataset": dataset,
                "detector": detector,
                "source_dataset": source_dataset,
                "experiment": "zero_shot_transfer",
                "transfer_accuracy": float(transfer_acc),
            }
            all_results.append(r)
    else:
        logger.warning(f"Source SAR not found: {source_sar_path}")

    # Save all results
    results_path = output_dir / "additional_datasets_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {results_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("ADDITIONAL DATASETS RESULTS")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Detector':<18} {'Static':>8} {'MoSEs-lr':>10} {'MoSEs-xg':>10}")
    print("-" * 80)
    for r in all_results:
        if r.get("experiment") == "direct":
            ds = r["dataset"]
            det = r["detector"]
            static = r.get("static_threshold", {}).get("accuracy", float("nan"))
            lr = r.get("moses_lr", {}).get("accuracy", float("nan"))
            xg = r.get("moses_xg", {}).get("accuracy", float("nan"))
            print(f"{ds:<15} {det:<18} {static:>8.4f} {lr:>10.4f} {xg:>10.4f}")
    print("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ADDITIONAL_DATASETS)
    parser.add_argument("--detector", nargs="+", default=["roberta"])
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--sar_dir", default="results/sar_models")
    parser.add_argument("--output_dir", default="results/additional")
    parser.add_argument("--pca_dims", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_additional_experiments(
        datasets=args.datasets,
        detectors=args.detector,
        data_dir=Path(args.data_dir),
        processed_dir=Path(args.processed_dir),
        sar_dir=Path(args.sar_dir),
        output_dir=Path(args.output_dir),
        pca_dims=args.pca_dims,
        device=args.device,
        fp16=args.fp16,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
