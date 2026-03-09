"""
preprocess.py
-------------
Step 1 of the MoSEs pipeline.

For each dataset + base detector combination, this script:
1. Computes the base detector discrimination score for every sample
2. Computes conditional features (text length, log-prob stats, repetition, TTR)
3. Computes BGE-M3 semantic embeddings (1024D)
4. Fits PCA on reference embeddings and projects both ref and test
5. Saves processed data as .pkl files for downstream steps

Usage:
    # Single dataset, single detector
    python scripts/preprocess.py --dataset CMV --detector roberta --device cuda

    # All datasets and detectors
    python scripts/preprocess.py --dataset all --detector all --device cuda --fp16
"""

import os
import json
import pickle
import argparse
import time
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS_MAIN = ["CMV", "SciXGen", "WP", "XSum"]
DATASETS_LOW = ["CNN", "DialogSum", "IMDB", "PubMedQA"]
DATASETS_ADDITIONAL = ["TruthfulQA", "HC3"]
ALL_DATASETS = DATASETS_MAIN + DATASETS_LOW + DATASETS_ADDITIONAL

DETECTORS = ["roberta", "fastdetectgpt", "lastde"]


# ---------------------------------------------------------------------------
# Conditional feature extraction
# ---------------------------------------------------------------------------

def compute_ngram_repetition(tokens: List[str], n: int) -> float:
    """Fraction of n-grams that are repeated (non-unique)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - unique / total if total > 0 else 0.0


def compute_type_token_ratio(tokens: List[str]) -> float:
    """Ratio of unique tokens to total tokens."""
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


def extract_conditional_features(
    text: str,
    log_probs: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Compute the conditional feature vector used by CTE.

    Features (6 scalar features before PCA):
        0: text length (# whitespace-split tokens)
        1: mean token log-probability (0 if log_probs not provided)
        2: variance of token log-probabilities (0 if not provided)
        3: 2-gram repetition rate
        4: 3-gram repetition rate
        5: type-token ratio
    """
    tokens = text.split()
    length = len(tokens)
    bigram_rep = compute_ngram_repetition(tokens, 2)
    trigram_rep = compute_ngram_repetition(tokens, 3)
    ttr = compute_type_token_ratio(tokens)

    if log_probs is not None and len(log_probs) > 0:
        lp = np.array(log_probs)
        lp_mean = float(np.mean(lp))
        lp_var = float(np.var(lp))
    else:
        lp_mean = 0.0
        lp_var = 0.0

    return np.array([length, lp_mean, lp_var, bigram_rep, trigram_rep, ttr], dtype=np.float32)


# ---------------------------------------------------------------------------
# Base detector: RoBERTa
# ---------------------------------------------------------------------------

def load_roberta_detector(device: str):
    """Load RoBERTa-based AI-text detector from HuggingFace."""
    from transformers import pipeline
    logger.info("Loading RoBERTa detector (roberta-base-openai-detector)...")
    detector = pipeline(
        "text-classification",
        model="roberta-base-openai-detector",
        device=0 if device == "cuda" else -1,
        truncation=True,
        max_length=512,
    )
    return detector


def roberta_score(detector, text: str) -> float:
    """Returns P(AI-generated) in [0, 1]."""
    result = detector(text, truncation=True, max_length=512)[0]
    # Model returns LABEL_0=human, LABEL_1=AI
    label = result["label"]
    score = result["score"]
    return score if label == "LABEL_1" else 1.0 - score


# ---------------------------------------------------------------------------
# Base detector: Fast-DetectGPT
# ---------------------------------------------------------------------------

def load_proxy_model(model_name: str = "EleutherAI/gpt-neo-2.7B", device: str = "cuda", fp16: bool = False):
    """Load GPT-Neo proxy model for scoring."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    logger.info(f"Loading proxy model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    if device == "cuda":
        model = model.cuda()
    return tokenizer, model


def compute_token_logprobs(text: str, tokenizer, model, device: str = "cuda") -> Tuple[float, List[float]]:
    """
    Compute per-token log-probabilities under the proxy model.
    Returns (mean_logprob, list_of_token_logprobs).
    """
    import torch
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"]
    if device == "cuda":
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is mean NLL; we want per-token log-probs
        logits = outputs.logits  # (1, seq_len, vocab_size)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Shift: predict token t from tokens 0..t-1
    shift_logprobs = log_probs[0, :-1, :]   # (seq_len-1, vocab)
    shift_labels = input_ids[0, 1:]          # (seq_len-1,)
    token_logprobs = shift_logprobs[
        range(len(shift_labels)), shift_labels
    ].cpu().numpy().tolist()

    mean_lp = float(np.mean(token_logprobs)) if token_logprobs else 0.0
    return mean_lp, token_logprobs


def fast_detectgpt_score(text: str, tokenizer, model, device: str = "cuda", n_perturbations: int = 100) -> Tuple[float, List[float]]:
    """
    Fast-DetectGPT conditional probability curvature score.
    Simplified version: uses log-prob difference as the discrimination score.
    Full version with perturbations requires mask filling (see original paper).
    Returns (score, token_log_probs).
    """
    mean_lp, token_logprobs = compute_token_logprobs(text, tokenizer, model, device)
    # Fast-DetectGPT score: negative of conditional log-prob (higher = more AI-like)
    return -mean_lp, token_logprobs


def lastde_score(text: str, tokenizer, model, device: str = "cuda") -> Tuple[float, List[float]]:
    """
    Lastde: temporal dynamics of token log-probabilities.
    Uses last-layer entropy differences as the discrimination score.
    Returns (score, token_log_probs).
    """
    import torch
    mean_lp, token_logprobs = compute_token_logprobs(text, tokenizer, model, device)
    # Lastde score: based on the variance/dynamics of log-probs
    if len(token_logprobs) > 1:
        # Temporal difference score
        diffs = np.diff(token_logprobs)
        score = float(np.mean(np.abs(diffs)))
    else:
        score = 0.0
    return score, token_logprobs


# ---------------------------------------------------------------------------
# Embedding: BGE-M3
# ---------------------------------------------------------------------------

def load_bge_m3():
    """Load BGE-M3 embedding model."""
    from FlagEmbedding import BGEM3FlagModel
    logger.info("Loading BGE-M3 embedding model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return model


def compute_bge_embeddings(texts: List[str], model, batch_size: int = 32) -> np.ndarray:
    """Compute BGE-M3 dense embeddings. Returns (N, 1024) float32 array."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BGE-M3 embeddings"):
        batch = texts[i : i + batch_size]
        out = model.encode(batch, batch_size=batch_size, max_length=512)
        all_embeddings.append(out["dense_vecs"])
    return np.vstack(all_embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Alternative embeddings for method exploration experiments
# ---------------------------------------------------------------------------

def load_sentence_transformer(model_name: str):
    """Load a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading SentenceTransformer: {model_name}")
    return SentenceTransformer(model_name)


def compute_st_embeddings(texts: List[str], model, batch_size: int = 64) -> np.ndarray:
    """Compute SentenceTransformer embeddings."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


EMBEDDING_MODEL_NAMES = {
    "bge-m3": None,           # Loaded separately with FlagEmbedding
    "e5-large": "intfloat/e5-large-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def load_dataset_split(data_dir: Path, dataset: str, split: str) -> List[Dict]:
    """Load ref.json or test.json for a dataset."""
    path = data_dir / dataset / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data


def preprocess_dataset(
    dataset: str,
    detector: str,
    data_dir: Path,
    output_dir: Path,
    pca_dims: int = 32,
    device: str = "cuda",
    fp16: bool = False,
    embedding_model: str = "bge-m3",
    batch_size: int = 32,
    seed: int = 42,
):
    """
    Full preprocessing pipeline for one dataset + detector combination.

    Saves:
        {output_dir}/{dataset}_{detector}_{embedding_model}_ref.pkl
        {output_dir}/{dataset}_{detector}_{embedding_model}_test.pkl
    """
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{dataset}_{detector}_{embedding_model}"
    ref_out = output_dir / f"{suffix}_ref.pkl"
    test_out = output_dir / f"{suffix}_test.pkl"

    if ref_out.exists() and test_out.exists():
        logger.info(f"[{suffix}] Already processed. Skipping.")
        return

    logger.info(f"[{suffix}] Loading data...")
    ref_data = load_dataset_split(data_dir, dataset, "ref")
    test_data = load_dataset_split(data_dir, dataset, "test")
    all_data = ref_data + test_data

    texts = [d["text"] for d in all_data]
    labels = np.array([d["label"] for d in all_data], dtype=np.int32)

    # ---- Step A: Base detector scores ----
    logger.info(f"[{suffix}] Computing {detector} scores...")
    t0 = time.time()

    if detector == "roberta":
        roberta = load_roberta_detector(device)
        scores = []
        log_probs_list = [None] * len(texts)
        for text in tqdm(texts, desc="RoBERTa scoring"):
            scores.append(roberta_score(roberta, text))
        scores = np.array(scores, dtype=np.float32)
        del roberta

    elif detector in ("fastdetectgpt", "lastde"):
        tokenizer, proxy_model = load_proxy_model(
            "EleutherAI/gpt-neo-2.7B", device=device, fp16=fp16
        )
        scores = []
        log_probs_list = []
        for text in tqdm(texts, desc=f"{detector} scoring"):
            if detector == "fastdetectgpt":
                s, lp = fast_detectgpt_score(text, tokenizer, proxy_model, device)
            else:
                s, lp = lastde_score(text, tokenizer, proxy_model, device)
            scores.append(s)
            log_probs_list.append(lp)
        scores = np.array(scores, dtype=np.float32)
        del proxy_model
    else:
        raise ValueError(f"Unknown detector: {detector}")

    logger.info(f"[{suffix}] Detector scoring done in {time.time()-t0:.1f}s")

    # ---- Step B: Conditional features ----
    logger.info(f"[{suffix}] Computing conditional features...")
    cond_features = []
    for i, text in enumerate(texts):
        lp = log_probs_list[i] if detector != "roberta" else None
        feat = extract_conditional_features(text, lp)
        cond_features.append(feat)
    cond_features = np.stack(cond_features, axis=0)  # (N, 6)

    # ---- Step C: Semantic embeddings ----
    logger.info(f"[{suffix}] Computing {embedding_model} embeddings...")
    t0 = time.time()

    if embedding_model == "bge-m3":
        emb_model = load_bge_m3()
        embeddings = compute_bge_embeddings(texts, emb_model, batch_size=batch_size)
        del emb_model
    else:
        st_model_name = EMBEDDING_MODEL_NAMES[embedding_model]
        st_model = load_sentence_transformer(st_model_name)
        embeddings = compute_st_embeddings(texts, st_model, batch_size=batch_size)
        del st_model

    logger.info(f"[{suffix}] Embedding done in {time.time()-t0:.1f}s. Shape: {embeddings.shape}")

    # ---- Step D: PCA on reference set, apply to both ----
    logger.info(f"[{suffix}] Fitting PCA (n_components={pca_dims}) on reference set...")
    n_ref = len(ref_data)
    ref_embeddings = embeddings[:n_ref]
    test_embeddings = embeddings[n_ref:]

    pca = PCA(n_components=pca_dims, random_state=seed)
    ref_pca = pca.fit_transform(ref_embeddings).astype(np.float32)
    test_pca = pca.transform(test_embeddings).astype(np.float32)

    # ---- Step E: Assemble and save ----
    def build_records(data, disc_scores, cond_feats, full_embs, pca_embs):
        records = []
        for i, d in enumerate(data):
            records.append({
                "text": d["text"],
                "label": int(d["label"]),
                "score": float(disc_scores[i]),
                "cond_features": cond_feats[i],       # (6,) float32
                "embedding": full_embs[i],              # (D,) float32
                "pca_embedding": pca_embs[i],           # (pca_dims,) float32
            })
        return records

    ref_records = build_records(
        ref_data,
        scores[:n_ref],
        cond_features[:n_ref],
        ref_embeddings,
        ref_pca,
    )
    test_records = build_records(
        test_data,
        scores[n_ref:],
        cond_features[n_ref:],
        test_embeddings,
        test_pca,
    )

    with open(ref_out, "wb") as f:
        pickle.dump({"records": ref_records, "pca": pca}, f)
    with open(test_out, "wb") as f:
        pickle.dump({"records": test_records}, f)

    logger.info(f"[{suffix}] Saved: {ref_out} ({len(ref_records)} samples)")
    logger.info(f"[{suffix}] Saved: {test_out} ({len(test_records)} samples)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for MoSEs.")
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument(
        "--detector",
        nargs="+",
        default=["roberta"],
        choices=DETECTORS + ["all"],
    )
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--pca_dims", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="bge-m3",
        choices=list(EMBEDDING_MODEL_NAMES.keys()),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = args.dataset
    if "all" in datasets:
        datasets = ALL_DATASETS

    detectors = args.detector
    if "all" in detectors:
        detectors = DETECTORS

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Datasets: {datasets}")
    logger.info(f"Detectors: {detectors}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"PCA dims: {args.pca_dims}")
    logger.info(f"Device: {args.device}, fp16={args.fp16}")

    total_start = time.time()

    for dataset in datasets:
        for detector in detectors:
            try:
                preprocess_dataset(
                    dataset=dataset,
                    detector=detector,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    pca_dims=args.pca_dims,
                    device=args.device,
                    fp16=args.fp16,
                    embedding_model=args.embedding_model,
                    batch_size=args.batch_size,
                    seed=args.seed,
                )
            except FileNotFoundError as e:
                logger.warning(f"Skipping {dataset}/{detector}: {e}")
            except Exception as e:
                logger.error(f"Error processing {dataset}/{detector}: {e}", exc_info=True)

    logger.info(f"All preprocessing done in {(time.time()-total_start)/60:.1f} minutes.")


if __name__ == "__main__":
    main()
