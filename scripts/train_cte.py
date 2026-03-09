"""
train_cte.py
------------
Step 3 of the MoSEs pipeline.

Trains the Conditional Threshold Estimator (CTE) and evaluates on the test set.
Implements:
  - MoSEs-lr: Logistic regression with class-weighted NLL
  - MoSEs-xg: XGBoost (depth=6, 100 estimators)

Also evaluates baselines:
  - Static threshold: single global threshold fit on reference set
  - Nearest voting: majority vote among nearest reference neighbors

Usage:
    python scripts/train_cte.py --dataset CMV --detector roberta --cte_type lr
    python scripts/train_cte.py --dataset all --detector all --cte_type both
"""

import os
import json
import pickle
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_DATASETS = ["CMV", "SciXGen", "WP", "XSum", "CNN", "DialogSum", "IMDB", "PubMedQA", "TruthfulQA", "HC3"]
DETECTORS = ["roberta", "fastdetectgpt", "lastde"]


# ---------------------------------------------------------------------------
# Baseline: Static threshold
# ---------------------------------------------------------------------------

def fit_static_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the optimal global threshold on the reference set.
    Sweeps all unique score values and picks threshold maximizing accuracy.
    """
    best_thresh = 0.5
    best_acc = 0.0
    thresholds = np.unique(scores)
    for t in thresholds:
        preds = (scores >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh


# ---------------------------------------------------------------------------
# Baseline: Nearest-neighbor voting
# ---------------------------------------------------------------------------

def nearest_voting(
    query_emb: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    k: int = 5,
) -> int:
    """Predict label by majority vote among k nearest reference neighbors."""
    from sklearn.preprocessing import normalize
    q = normalize(query_emb.reshape(1, -1))[0]
    ref_norm = normalize(ref_embeddings)
    sims = ref_norm @ q
    top_k = np.argsort(sims)[-k:]
    votes = ref_labels[top_k]
    return int(np.round(votes.mean()))


# ---------------------------------------------------------------------------
# CTE: feature construction for CTE input
# ---------------------------------------------------------------------------

def build_cte_features(
    query_record: Dict,
    neighbor_records: List[Dict],
) -> np.ndarray:
    """
    Build the CTE input feature vector for one sample.

    Features:
        - query discrimination score (scalar)
        - query conditional features (6 scalars)
        - neighbor stats: mean/std/min/max of neighbor scores (4 scalars)
        - neighbor stats: fraction of AI-labeled neighbors (scalar)
        - query PCA embedding (pca_dims scalars)

    Total: 1 + 6 + 4 + 1 + pca_dims = 12 + pca_dims features
    """
    query_score = query_record["score"]
    query_cond = query_record["cond_features"]   # (6,)
    query_pca = query_record["pca_embedding"]     # (pca_dims,)

    if len(neighbor_records) > 0:
        nbr_scores = np.array([r["score"] for r in neighbor_records])
        nbr_labels = np.array([r["label"] for r in neighbor_records])
        nbr_mean = float(np.mean(nbr_scores))
        nbr_std = float(np.std(nbr_scores))
        nbr_min = float(np.min(nbr_scores))
        nbr_max = float(np.max(nbr_scores))
        nbr_ai_frac = float(np.mean(nbr_labels))
    else:
        nbr_mean = nbr_std = nbr_min = nbr_max = nbr_ai_frac = 0.0

    feat = np.concatenate([
        [query_score],
        query_cond,
        [nbr_mean, nbr_std, nbr_min, nbr_max, nbr_ai_frac],
        query_pca,
    ])
    return feat.astype(np.float32)


# ---------------------------------------------------------------------------
# CTE models
# ---------------------------------------------------------------------------

class CTELogisticRegression:
    """CTE-lr: logistic regression with class-weighted NLL."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class CTEXGBoost:
    """CTE-xg: XGBoost with depth=6, 100 estimators."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Install xgboost: pip install xgboost")

        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset: str,
    detector: str,
    processed_dir: Path,
    sar_dir: Path,
    output_dir: Path,
    embedding_model: str = "bge-m3",
    cte_types: List[str] = ("lr", "xgb"),
    seed: int = 42,
) -> Dict:
    """
    Full CTE training + evaluation for one dataset/detector combination.
    Returns dict of results for all methods.
    """
    suffix = f"{dataset}_{detector}_{embedding_model}"
    ref_path = processed_dir / f"{suffix}_ref.pkl"
    test_path = processed_dir / f"{suffix}_test.pkl"
    sar_path = sar_dir / f"{suffix}_sar.pkl"

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference data not found: {ref_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    # Load data
    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    ref_records = ref_data["records"]
    test_records = test_data["records"]

    ref_scores = np.array([r["score"] for r in ref_records])
    ref_labels = np.array([r["label"] for r in ref_records])
    test_scores = np.array([r["score"] for r in test_records])
    test_labels = np.array([r["label"] for r in test_records])

    results = {
        "dataset": dataset,
        "detector": detector,
        "embedding_model": embedding_model,
        "n_ref": len(ref_records),
        "n_test": len(test_records),
    }

    # ---- Baseline 1: Static threshold ----
    t0 = time.time()
    best_thresh = fit_static_threshold(ref_scores, ref_labels)
    static_preds = (test_scores >= best_thresh).astype(int)
    static_acc = accuracy_score(test_labels, static_preds)
    results["static_threshold"] = {
        "accuracy": float(static_acc),
        "threshold": float(best_thresh),
        "time_s": time.time() - t0,
    }
    logger.info(f"  Static threshold acc: {static_acc:.4f}")

    # ---- Baseline 2: Nearest voting ----
    ref_embs = np.stack([r["pca_embedding"] for r in ref_records], axis=0)
    test_embs = np.stack([r["pca_embedding"] for r in test_records], axis=0)

    t0 = time.time()
    nn_preds = np.array([
        nearest_voting(test_embs[i], ref_embs, ref_labels, k=5)
        for i in range(len(test_records))
    ])
    nn_acc = accuracy_score(test_labels, nn_preds)
    results["nearest_voting"] = {
        "accuracy": float(nn_acc),
        "time_s": time.time() - t0,
    }
    logger.info(f"  Nearest voting acc: {nn_acc:.4f}")

    # ---- MoSEs methods (require SAR) ----
    if not sar_path.exists():
        logger.warning(f"SAR model not found: {sar_path}. Skipping MoSEs methods.")
        return results

    from train_sar import StylisticsAwareRouter
    sar = StylisticsAwareRouter.load(sar_path)

    # Route all test samples
    logger.info(f"  Routing {len(test_records)} test samples through SAR...")
    t0 = time.time()
    test_cte_features = []
    for record in test_records:
        _, _, neighbor_records = sar.route(record["pca_embedding"])
        feat = build_cte_features(record, neighbor_records)
        test_cte_features.append(feat)
    test_X = np.stack(test_cte_features, axis=0)
    routing_time = time.time() - t0

    # Build training features using leave-one-out routing on reference set
    logger.info(f"  Building CTE training features on reference set...")
    ref_cte_features = []
    for i, record in enumerate(ref_records):
        # Exclude self: temporarily mask out this sample
        # Simplified: just route normally (self may be included as neighbor)
        _, _, neighbor_records = sar.route(record["pca_embedding"])
        # Filter out self if present (match by text)
        neighbor_records = [nr for nr in neighbor_records if nr.get("text") != record.get("text")]
        feat = build_cte_features(record, neighbor_records)
        ref_cte_features.append(feat)
    ref_X = np.stack(ref_cte_features, axis=0)
    ref_y = ref_labels

    # ---- MoSEs-lr ----
    if "lr" in cte_types:
        t0 = time.time()
        cte_lr = CTELogisticRegression(seed=seed)
        cte_lr.fit(ref_X, ref_y)
        train_time = time.time() - t0

        t1 = time.time()
        lr_preds = cte_lr.predict(test_X)
        infer_time = (time.time() - t1) / len(test_records) * 1000  # ms per sample

        lr_acc = accuracy_score(test_labels, lr_preds)
        results["moses_lr"] = {
            "accuracy": float(lr_acc),
            "train_time_s": float(train_time),
            "infer_time_ms_per_sample": float(infer_time),
            "routing_time_s": float(routing_time),
        }
        logger.info(f"  MoSEs-lr acc: {lr_acc:.4f} (train={train_time:.3f}s, infer={infer_time:.3f}ms/sample)")

        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{suffix}_cte_lr.pkl", "wb") as f:
            pickle.dump(cte_lr, f)

    # ---- MoSEs-xg ----
    if "xgb" in cte_types:
        t0 = time.time()
        cte_xg = CTEXGBoost(seed=seed)
        cte_xg.fit(ref_X, ref_y)
        train_time = time.time() - t0

        t1 = time.time()
        xg_preds = cte_xg.predict(test_X)
        infer_time = (time.time() - t1) / len(test_records) * 1000

        xg_acc = accuracy_score(test_labels, xg_preds)
        results["moses_xg"] = {
            "accuracy": float(xg_acc),
            "train_time_s": float(train_time),
            "infer_time_ms_per_sample": float(infer_time),
            "routing_time_s": float(routing_time),
        }
        logger.info(f"  MoSEs-xg acc: {xg_acc:.4f} (train={train_time:.3f}s, infer={infer_time:.3f}ms/sample)")

        with open(output_dir / f"{suffix}_cte_xg.pkl", "wb") as f:
            pickle.dump(cte_xg, f)

    # Save results
    result_path = output_dir / f"{suffix}_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved: {result_path}")

    return results


def summarize_results(results_dir: Path) -> Dict:
    """Aggregate all result JSONs and print a summary table."""
    all_results = []
    for f in sorted(results_dir.glob("*_results.json")):
        with open(f) as fp:
            all_results.append(json.load(fp))

    print("\n" + "=" * 100)
    print(f"{'Dataset':<15} {'Detector':<18} {'Static':>8} {'NNVote':>8} {'MoSEs-lr':>10} {'MoSEs-xg':>10}")
    print("=" * 100)

    summary = {}
    for r in all_results:
        ds = r["dataset"]
        det = r["detector"]
        static = r.get("static_threshold", {}).get("accuracy", float("nan"))
        nn = r.get("nearest_voting", {}).get("accuracy", float("nan"))
        lr = r.get("moses_lr", {}).get("accuracy", float("nan"))
        xg = r.get("moses_xg", {}).get("accuracy", float("nan"))
        print(f"{ds:<15} {det:<18} {static:>8.4f} {nn:>8.4f} {lr:>10.4f} {xg:>10.4f}")
        key = f"{ds}_{det}"
        summary[key] = {"static": static, "nn_voting": nn, "moses_lr": lr, "moses_xg": xg}

    print("=" * 100)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CTE and evaluate MoSEs.")
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--detector", nargs="+", default=["roberta"])
    parser.add_argument(
        "--cte_type",
        nargs="+",
        default=["lr", "xgb"],
        choices=["lr", "xgb", "both"],
    )
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--sar_dir", type=str, default="results/sar_models")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--embedding_model", type=str, default="bge-m3")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.summarize:
        summarize_results(Path(args.output_dir))
        return

    datasets = args.dataset
    if "all" in datasets:
        datasets = ALL_DATASETS

    detectors = args.detector
    if "all" in detectors:
        detectors = DETECTORS

    cte_types = args.cte_type
    if "both" in cte_types:
        cte_types = ["lr", "xgb"]

    processed_dir = Path(args.processed_dir)
    sar_dir = Path(args.sar_dir)
    output_dir = Path(args.output_dir)

    all_results = []
    for dataset in datasets:
        for detector in detectors:
            logger.info(f"\n--- {dataset} / {detector} ---")
            try:
                r = evaluate_dataset(
                    dataset=dataset,
                    detector=detector,
                    processed_dir=processed_dir,
                    sar_dir=sar_dir,
                    output_dir=output_dir,
                    embedding_model=args.embedding_model,
                    cte_types=cte_types,
                    seed=args.seed,
                )
                all_results.append(r)
            except FileNotFoundError as e:
                logger.warning(f"Skipping: {e}")
            except Exception as e:
                logger.error(f"Error {dataset}/{detector}: {e}", exc_info=True)

    # Summary
    if all_results:
        summarize_results(output_dir)


if __name__ == "__main__":
    main()
