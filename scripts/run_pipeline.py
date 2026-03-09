"""
scripts/run_pipeline.py
-----------------------
Python orchestrator for the full MoSEs pipeline.
Runs preprocessing -> SAR -> CTE for all specified dataset/detector combos
and records wall-clock timing for each step (used in paper's Table 6).

Usage:
    # Full reproduction (all datasets, all detectors)
    python scripts/run_pipeline.py --dataset all --detector all

    # Quick smoke test (one dataset, RoBERTa only)
    python scripts/run_pipeline.py --dataset CMV --detector roberta --quick

    # RoBERTa only, all datasets
    python scripts/run_pipeline.py --dataset all --detector roberta
"""

import sys
import os
import json
import time
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAIN_DATASETS    = ["CMV", "SciXGen", "WP", "XSum"]
LOW_DATASETS     = ["CNN", "DialogSum", "IMDB", "PubMedQA"]
ALL_DATASETS     = MAIN_DATASETS + LOW_DATASETS
DETECTORS        = ["roberta", "fastdetectgpt", "lastde"]


def run_step(cmd: list, step_name: str) -> Tuple[int, float]:
    """Run a shell command, log output, return (returncode, elapsed_seconds)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"CMD:  {' '.join(str(c) for c in cmd)}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error(f"Step FAILED (code {result.returncode}): {step_name}")
    else:
        logger.info(f"Step done in {elapsed:.1f}s: {step_name}")
    return result.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(description="End-to-end MoSEs pipeline runner.")
    parser.add_argument("--dataset",        nargs="+", default=["all"])
    parser.add_argument("--detector",       nargs="+", default=["roberta"])
    parser.add_argument("--data_dir",       default="data/raw")
    parser.add_argument("--processed_dir",  default="data/processed")
    parser.add_argument("--sar_dir",        default="results/sar_models")
    parser.add_argument("--results_dir",    default="results")
    parser.add_argument("--embedding_model",default="bge-m3")
    parser.add_argument("--pca_dims",       type=int,   default=32)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--epsilon",        type=float, default=0.05)
    parser.add_argument("--neighborhood_k", type=int,   default=20)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--fp16",           action="store_true")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 SAR epochs, skip low-resource datasets",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 10

    datasets  = ALL_DATASETS if "all" in args.dataset  else args.dataset
    detectors = DETECTORS    if "all" in args.detector else args.detector

    if args.quick:
        datasets = [d for d in datasets if d in MAIN_DATASETS]

    python   = sys.executable
    timing   = {}
    t_total  = time.time()

    # ------------------------------------------------------------------ #
    # Step 1 — Preprocessing (one run per detector, covers all datasets)  #
    # ------------------------------------------------------------------ #
    for detector in detectors:
        cmd = [
            python, "scripts/preprocess.py",
            "--dataset",        *datasets,
            "--detector",       detector,
            "--data_dir",       args.data_dir,
            "--output_dir",     args.processed_dir,
            "--pca_dims",       str(args.pca_dims),
            "--device",         args.device,
            "--embedding_model",args.embedding_model,
            "--seed",           str(args.seed),
        ]
        if args.fp16:
            cmd.append("--fp16")
        rc, elapsed = run_step(cmd, f"preprocess/{detector}")
        timing[f"preprocess_{detector}"] = {"elapsed_s": elapsed, "rc": rc}

    # ------------------------------------------------------------------ #
    # Step 2 — SAR training                                               #
    # ------------------------------------------------------------------ #
    for detector in detectors:
        cmd = [
            python, "scripts/train_sar.py",
            "--dataset",        *datasets,
            "--detector",       detector,
            "--processed_dir",  args.processed_dir,
            "--output_dir",     args.sar_dir,
            "--embedding_model",args.embedding_model,
            "--epochs",         str(args.epochs),
            "--epsilon",        str(args.epsilon),
            "--neighborhood_k", str(args.neighborhood_k),
            "--seed",           str(args.seed),
        ]
        rc, elapsed = run_step(cmd, f"train_sar/{detector}")
        timing[f"sar_{detector}"] = {"elapsed_s": elapsed, "rc": rc}

    # ------------------------------------------------------------------ #
    # Step 3 — CTE training + evaluation                                  #
    # ------------------------------------------------------------------ #
    for detector in detectors:
        cmd = [
            python, "scripts/train_cte.py",
            "--dataset",        *datasets,
            "--detector",       detector,
            "--cte_type",       "lr", "xgb",
            "--processed_dir",  args.processed_dir,
            "--sar_dir",        args.sar_dir,
            "--output_dir",     args.results_dir,
            "--embedding_model",args.embedding_model,
            "--seed",           str(args.seed),
        ]
        rc, elapsed = run_step(cmd, f"train_cte/{detector}")
        timing[f"cte_{detector}"] = {"elapsed_s": elapsed, "rc": rc}

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    total_elapsed = time.time() - t_total
    timing["total"] = {"elapsed_s": total_elapsed}

    logger.info(f"\n{'='*60}")
    logger.info("Pipeline complete!")
    logger.info(f"Total wall time: {total_elapsed/60:.1f} minutes")
    logger.info("Per-step timing:")
    for step, info in timing.items():
        logger.info(f"  {step:<35} {info['elapsed_s']:>7.1f}s")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    timing_path = Path(args.results_dir) / "pipeline_timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    logger.info(f"Timing saved: {timing_path}")

    # Print results table
    subprocess.run([python, "scripts/train_cte.py",
                    "--summarize", "--output_dir", args.results_dir])


if __name__ == "__main__":
    main()
