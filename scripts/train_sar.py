"""
train_sar.py
------------
Step 2 of the MoSEs pipeline.

Trains the Stylistics-Aware Router (SAR) on the reference set.
SAR groups texts by stylistic similarity using prototype-based
nearest-neighbor retrieval with Sinkhorn-Knopp regularization.

Usage:
    python scripts/train_sar.py --dataset CMV --detector roberta
    python scripts/train_sar.py --dataset all --detector all --epochs 100 --epsilon 0.05
"""

import os
import json
import pickle
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_DATASETS = ["CMV", "SciXGen", "WP", "XSum", "CNN", "DialogSum", "IMDB", "PubMedQA", "TruthfulQA", "HC3"]
DETECTORS = ["roberta", "fastdetectgpt", "lastde"]


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp optimal transport
# ---------------------------------------------------------------------------

def sinkhorn_knopp(cost_matrix: np.ndarray, epsilon: float = 0.05, n_iters: int = 100) -> np.ndarray:
    """
    Sinkhorn-Knopp algorithm for optimal transport regularization.

    Args:
        cost_matrix: (N, K) assignment cost matrix (lower = better match)
        epsilon: regularization strength
        n_iters: number of Sinkhorn iterations

    Returns:
        transport_plan: (N, K) soft assignment matrix (rows sum to 1/N)
    """
    N, K = cost_matrix.shape
    # Convert costs to log-probabilities
    log_K = -cost_matrix / epsilon

    # Initialize uniform marginals
    u = np.zeros(N)
    v = np.zeros(K)

    log_alpha = np.log(np.ones(N) / N)   # uniform over samples
    log_beta = np.log(np.ones(K) / K)    # uniform over prototypes

    for _ in range(n_iters):
        # u-update
        u = log_alpha - np.log(np.sum(np.exp(log_K + v[None, :]), axis=1) + 1e-8)
        # v-update
        v = log_beta - np.log(np.sum(np.exp(log_K + u[:, None]), axis=0) + 1e-8)

    log_T = log_K + u[:, None] + v[None, :]
    T = np.exp(log_T)
    return T  # (N, K)


# ---------------------------------------------------------------------------
# SAR model
# ---------------------------------------------------------------------------

class StylisticsAwareRouter:
    """
    Prototype-based Stylistics-Aware Router (SAR).

    Learns K prototype embeddings that represent distinct stylistic groups
    in the reference set. Uses Sinkhorn-Knopp OT for soft assignments.

    At inference, routes each input sample to its nearest style prototype,
    then retrieves a neighborhood of reference samples with the same style.
    """

    def __init__(
        self,
        n_prototypes: int = 10,
        epsilon: float = 0.05,
        epochs: int = 100,
        lr: float = 0.01,
        neighborhood_k: int = 20,
        seed: int = 42,
    ):
        self.n_prototypes = n_prototypes
        self.epsilon = epsilon
        self.epochs = epochs
        self.lr = lr
        self.neighborhood_k = neighborhood_k
        self.seed = seed

        self.prototypes: Optional[np.ndarray] = None  # (K, D)
        self.ref_embeddings: Optional[np.ndarray] = None
        self.ref_assignments: Optional[np.ndarray] = None  # (N,) hard assignment
        self.ref_records: Optional[List[Dict]] = None

    def _init_prototypes(self, embeddings: np.ndarray) -> np.ndarray:
        """K-means++ initialization for prototypes."""
        rng = np.random.RandomState(self.seed)
        N, D = embeddings.shape
        idx = [rng.randint(N)]
        for _ in range(1, self.n_prototypes):
            # Compute distances to nearest chosen prototype
            chosen = embeddings[idx]  # (k, D)
            dists = np.min(
                np.sum((embeddings[:, None, :] - chosen[None, :, :]) ** 2, axis=-1),
                axis=1,
            )  # (N,)
            probs = dists / (dists.sum() + 1e-8)
            idx.append(rng.choice(N, p=probs))
        return embeddings[idx].copy()  # (K, D)

    def fit(self, embeddings: np.ndarray, records: List[Dict]) -> "StylisticsAwareRouter":
        """
        Train SAR prototypes on reference set embeddings.

        Args:
            embeddings: (N, D) normalized semantic embeddings
            records: list of reference sample dicts

        Returns:
            self
        """
        np.random.seed(self.seed)
        embeddings = normalize(embeddings, norm="l2")
        N, D = embeddings.shape

        # Initialize prototypes
        self.prototypes = self._init_prototypes(embeddings)
        self.prototypes = normalize(self.prototypes, norm="l2")

        loss_history = []

        for epoch in range(self.epochs):
            # Compute cosine distance matrix (1 - cosine_sim = distance)
            # embeddings: (N, D), prototypes: (K, D)
            sim = embeddings @ self.prototypes.T  # (N, K)
            cost = 1.0 - sim  # (N, K) cost matrix

            # Soft assignment via Sinkhorn-Knopp
            T = sinkhorn_knopp(cost, epsilon=self.epsilon)  # (N, K)

            # Update prototypes: weighted mean of assigned embeddings
            # New prototype k = sum_i T[i,k] * emb[i] / sum_i T[i,k]
            weights = T / (T.sum(axis=0, keepdims=True) + 1e-8)  # (N, K) normalized
            new_prototypes = (embeddings[:, None, :] * weights[:, :, None]).sum(axis=0)  # (K, D)
            new_prototypes = normalize(new_prototypes, norm="l2")

            # Compute loss (total transport cost)
            loss = float((T * cost).sum())
            loss_history.append(loss)

            # Gradient step (blend toward new prototypes)
            self.prototypes = (1 - self.lr) * self.prototypes + self.lr * new_prototypes
            self.prototypes = normalize(self.prototypes, norm="l2")

            if (epoch + 1) % 20 == 0:
                logger.debug(f"  SAR epoch {epoch+1}/{self.epochs}, loss={loss:.4f}")

        # Compute final hard assignments
        sim = embeddings @ self.prototypes.T  # (N, K)
        self.ref_assignments = np.argmax(sim, axis=1)  # (N,)
        self.ref_embeddings = embeddings
        self.ref_records = records

        cluster_sizes = np.bincount(self.ref_assignments, minlength=self.n_prototypes)
        logger.info(f"  SAR cluster sizes: {cluster_sizes.tolist()}")
        logger.info(f"  Final loss: {loss_history[-1]:.4f}")

        return self

    def route(self, query_embedding: np.ndarray) -> Tuple[int, np.ndarray, List[Dict]]:
        """
        Route a single query to the nearest prototype and retrieve neighbors.

        Args:
            query_embedding: (D,) query semantic embedding

        Returns:
            prototype_idx: assigned prototype index
            neighbor_indices: indices into ref_records of k nearest neighbors
            neighbor_records: list of k nearest reference records
        """
        q = normalize(query_embedding.reshape(1, -1), norm="l2")[0]

        # Assign to nearest prototype
        sim_proto = q @ self.prototypes.T  # (K,)
        proto_idx = int(np.argmax(sim_proto))

        # Retrieve k nearest neighbors from the same prototype cluster
        cluster_mask = self.ref_assignments == proto_idx  # (N,)
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            # Fallback: use all reference samples
            cluster_indices = np.arange(len(self.ref_records))

        cluster_embs = self.ref_embeddings[cluster_indices]  # (|cluster|, D)
        sims = cluster_embs @ q  # (|cluster|,)
        k = min(self.neighborhood_k, len(cluster_indices))
        top_k = np.argsort(sims)[-k:][::-1]
        neighbor_indices = cluster_indices[top_k]
        neighbor_records = [self.ref_records[i] for i in neighbor_indices]

        return proto_idx, neighbor_indices, neighbor_records

    def route_batch(self, embeddings: np.ndarray) -> List[Tuple[int, np.ndarray, List[Dict]]]:
        """Route a batch of query embeddings."""
        results = []
        for emb in embeddings:
            results.append(self.route(emb))
        return results

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SAR model saved: {path}")

    @staticmethod
    def load(path: Path) -> "StylisticsAwareRouter":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

def train_sar_for_dataset(
    dataset: str,
    detector: str,
    processed_dir: Path,
    output_dir: Path,
    embedding_model: str = "bge-m3",
    n_prototypes: int = 10,
    epsilon: float = 0.05,
    epochs: int = 100,
    lr: float = 0.01,
    neighborhood_k: int = 20,
    seed: int = 42,
) -> StylisticsAwareRouter:
    """Load preprocessed reference data and train SAR."""
    suffix = f"{dataset}_{detector}_{embedding_model}"
    ref_path = processed_dir / f"{suffix}_ref.pkl"
    sar_out = output_dir / f"{suffix}_sar.pkl"

    if sar_out.exists():
        logger.info(f"[{suffix}] SAR already trained. Loading from {sar_out}")
        return StylisticsAwareRouter.load(sar_out)

    if not ref_path.exists():
        raise FileNotFoundError(f"Preprocessed reference data not found: {ref_path}")

    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)

    records = ref_data["records"]
    embeddings = np.stack([r["pca_embedding"] for r in records], axis=0)

    logger.info(f"[{suffix}] Training SAR on {len(records)} samples, dim={embeddings.shape[1]}")
    t0 = time.time()

    sar = StylisticsAwareRouter(
        n_prototypes=n_prototypes,
        epsilon=epsilon,
        epochs=epochs,
        lr=lr,
        neighborhood_k=neighborhood_k,
        seed=seed,
    )
    sar.fit(embeddings, records)

    logger.info(f"[{suffix}] SAR training done in {time.time()-t0:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    sar.save(sar_out)

    # Save training metadata
    meta = {
        "dataset": dataset,
        "detector": detector,
        "embedding_model": embedding_model,
        "n_prototypes": n_prototypes,
        "epsilon": epsilon,
        "epochs": epochs,
        "lr": lr,
        "neighborhood_k": neighborhood_k,
        "n_ref_samples": len(records),
        "embedding_dim": embeddings.shape[1],
        "training_time_s": time.time() - t0,
    }
    with open(output_dir / f"{suffix}_sar_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return sar


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SAR for MoSEs.")
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--detector", nargs="+", default=["roberta"])
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/sar_models")
    parser.add_argument("--embedding_model", type=str, default="bge-m3")
    parser.add_argument("--n_prototypes", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--neighborhood_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = args.dataset
    if "all" in datasets:
        datasets = ALL_DATASETS

    detectors = args.detector
    if "all" in detectors:
        detectors = DETECTORS

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    for dataset in datasets:
        for detector in detectors:
            try:
                train_sar_for_dataset(
                    dataset=dataset,
                    detector=detector,
                    processed_dir=processed_dir,
                    output_dir=output_dir,
                    embedding_model=args.embedding_model,
                    n_prototypes=args.n_prototypes,
                    epsilon=args.epsilon,
                    epochs=args.epochs,
                    lr=args.lr,
                    neighborhood_k=args.neighborhood_k,
                    seed=args.seed,
                )
            except FileNotFoundError as e:
                logger.warning(f"Skipping {dataset}/{detector}: {e}")
            except Exception as e:
                logger.error(f"Error {dataset}/{detector}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
