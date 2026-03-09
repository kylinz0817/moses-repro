"""
tests/test_pipeline.py
-----------------------
Smoke tests for the MoSEs pipeline. 
Tests all components with synthetic data to verify correctness before
running on real data (which requires GPU hours).

Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ref_records():
    """Create synthetic reference records for testing."""
    np.random.seed(42)
    N = 100
    records = []
    for i in range(N):
        label = i % 2  # alternating
        records.append({
            "text": f"Sample text number {i} " * 10,
            "label": label,
            "score": float(0.3 + 0.4 * label + np.random.randn() * 0.1),
            "cond_features": np.random.randn(6).astype(np.float32),
            "embedding": np.random.randn(32).astype(np.float32),
            "pca_embedding": np.random.randn(32).astype(np.float32),
        })
    return records


@pytest.fixture
def synthetic_test_records():
    """Create synthetic test records."""
    np.random.seed(123)
    N = 20
    records = []
    for i in range(N):
        label = i % 2
        records.append({
            "text": f"Test text number {i} " * 10,
            "label": label,
            "score": float(0.3 + 0.4 * label + np.random.randn() * 0.1),
            "cond_features": np.random.randn(6).astype(np.float32),
            "embedding": np.random.randn(32).astype(np.float32),
            "pca_embedding": np.random.randn(32).astype(np.float32),
        })
    return records


# ---------------------------------------------------------------------------
# Tests: Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_conditional_features_shape(self):
        from preprocess import extract_conditional_features
        text = "This is a test sentence with some words."
        feat = extract_conditional_features(text)
        assert feat.shape == (6,), f"Expected (6,), got {feat.shape}"

    def test_conditional_features_with_logprobs(self):
        from preprocess import extract_conditional_features
        text = "This is a test sentence."
        log_probs = [-1.2, -0.8, -2.1, -1.5, -0.9]
        feat = extract_conditional_features(text, log_probs)
        assert feat.shape == (6,)
        assert feat[1] == pytest.approx(np.mean(log_probs), abs=1e-5)

    def test_ngram_repetition(self):
        from preprocess import compute_ngram_repetition
        tokens_repeated = ["a", "b", "a", "b", "a", "b"]
        tokens_unique = ["a", "b", "c", "d", "e", "f"]
        assert compute_ngram_repetition(tokens_repeated, 2) > compute_ngram_repetition(tokens_unique, 2)

    def test_type_token_ratio(self):
        from preprocess import compute_type_token_ratio
        assert compute_type_token_ratio(["a", "a", "a"]) == pytest.approx(1/3)
        assert compute_type_token_ratio(["a", "b", "c"]) == pytest.approx(1.0)
        assert compute_type_token_ratio([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: SAR
# ---------------------------------------------------------------------------

class TestSAR:
    def test_sar_fit(self, synthetic_ref_records):
        from train_sar import StylisticsAwareRouter
        embeddings = np.stack([r["pca_embedding"] for r in synthetic_ref_records])
        sar = StylisticsAwareRouter(n_prototypes=5, epochs=10, seed=42)
        sar.fit(embeddings, synthetic_ref_records)
        assert sar.prototypes is not None
        assert sar.prototypes.shape == (5, embeddings.shape[1])

    def test_sar_route(self, synthetic_ref_records):
        from train_sar import StylisticsAwareRouter
        embeddings = np.stack([r["pca_embedding"] for r in synthetic_ref_records])
        sar = StylisticsAwareRouter(n_prototypes=5, epochs=10, neighborhood_k=10, seed=42)
        sar.fit(embeddings, synthetic_ref_records)
        
        query = np.random.randn(embeddings.shape[1]).astype(np.float32)
        proto_idx, nbr_indices, nbr_records = sar.route(query)
        
        assert isinstance(proto_idx, int)
        assert 0 <= proto_idx < 5
        assert len(nbr_records) <= 10
        assert len(nbr_records) > 0

    def test_sinkhorn_knopp(self):
        from train_sar import sinkhorn_knopp
        N, K = 20, 5
        cost = np.random.rand(N, K)
        T = sinkhorn_knopp(cost, epsilon=0.05)
        assert T.shape == (N, K)
        assert T.min() >= 0
        # Check that transport plan is non-negative
        assert np.all(T >= 0)


# ---------------------------------------------------------------------------
# Tests: CTE
# ---------------------------------------------------------------------------

class TestCTE:
    def test_build_cte_features(self, synthetic_ref_records, synthetic_test_records):
        from train_cte import build_cte_features
        query = synthetic_test_records[0]
        neighbors = synthetic_ref_records[:10]
        feat = build_cte_features(query, neighbors)
        assert feat.ndim == 1
        # Expected: 1 + 6 + 5 + pca_dims = 12 + 32 = 44
        expected_len = 1 + 6 + 5 + query["pca_embedding"].shape[0]
        assert feat.shape[0] == expected_len, f"Expected {expected_len}, got {feat.shape[0]}"

    def test_cte_lr_fit_predict(self, synthetic_ref_records, synthetic_test_records):
        from train_sar import StylisticsAwareRouter
        from train_cte import CTELogisticRegression, build_cte_features
        
        ref_embs = np.stack([r["pca_embedding"] for r in synthetic_ref_records])
        ref_labels = np.array([r["label"] for r in synthetic_ref_records])
        test_labels = np.array([r["label"] for r in synthetic_test_records])

        sar = StylisticsAwareRouter(n_prototypes=5, epochs=10, seed=42)
        sar.fit(ref_embs, synthetic_ref_records)

        ref_X = np.stack([
            build_cte_features(r, sar.route(r["pca_embedding"])[2])
            for r in synthetic_ref_records
        ])
        test_X = np.stack([
            build_cte_features(r, sar.route(r["pca_embedding"])[2])
            for r in synthetic_test_records
        ])

        cte = CTELogisticRegression(seed=42)
        cte.fit(ref_X, ref_labels)
        preds = cte.predict(test_X)
        
        assert len(preds) == len(synthetic_test_records)
        assert set(preds).issubset({0, 1})

    def test_static_threshold(self, synthetic_ref_records, synthetic_test_records):
        from train_cte import fit_static_threshold
        ref_scores = np.array([r["score"] for r in synthetic_ref_records])
        ref_labels = np.array([r["label"] for r in synthetic_ref_records])
        thresh = fit_static_threshold(ref_scores, ref_labels)
        assert isinstance(thresh, float)
        assert 0.0 <= thresh <= 1.0


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_smoke(self, synthetic_ref_records, synthetic_test_records):
        """Smoke test: run full pipeline end-to-end on synthetic data."""
        from train_sar import StylisticsAwareRouter
        from train_cte import (
            CTELogisticRegression, CTEXGBoost, build_cte_features,
            fit_static_threshold, nearest_voting
        )
        from sklearn.metrics import accuracy_score

        ref_embs = np.stack([r["pca_embedding"] for r in synthetic_ref_records])
        ref_labels = np.array([r["label"] for r in synthetic_ref_records])
        ref_scores = np.array([r["score"] for r in synthetic_ref_records])
        test_labels = np.array([r["label"] for r in synthetic_test_records])
        test_scores = np.array([r["score"] for r in synthetic_test_records])
        test_embs = np.stack([r["pca_embedding"] for r in synthetic_test_records])

        # Static threshold
        thresh = fit_static_threshold(ref_scores, ref_labels)
        static_preds = (test_scores >= thresh).astype(int)
        static_acc = accuracy_score(test_labels, static_preds)
        assert 0.0 <= static_acc <= 1.0

        # SAR
        sar = StylisticsAwareRouter(n_prototypes=5, epochs=10, neighborhood_k=10, seed=42)
        sar.fit(ref_embs, synthetic_ref_records)

        # CTE features
        ref_X = np.stack([
            build_cte_features(r, sar.route(r["pca_embedding"])[2])
            for r in synthetic_ref_records
        ])
        test_X = np.stack([
            build_cte_features(r, sar.route(r["pca_embedding"])[2])
            for r in synthetic_test_records
        ])

        # MoSEs-lr
        cte_lr = CTELogisticRegression(seed=42)
        cte_lr.fit(ref_X, ref_labels)
        lr_preds = cte_lr.predict(test_X)
        lr_acc = accuracy_score(test_labels, lr_preds)
        assert 0.0 <= lr_acc <= 1.0

        print(f"\nSmoke test results: static={static_acc:.2f}, moses_lr={lr_acc:.2f}")
