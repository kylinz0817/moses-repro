# MoSEs Reproducibility Study

> **Reproduction and Extension of**: *MoSEs: Mixture of Stylistics Experts for AI-Generated Text Detection*
> 
> This repository contains our full reproduction pipeline, additional experiments, and analysis code for the NLP course reproducibility project.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Original Paper & Repo](#original-paper--repo)
- [Our Contributions](#our-contributions)
- [Dependencies](#dependencies)
- [Data Download](#data-download)
- [Project Structure](#project-structure)
- [Pipeline Steps](#pipeline-steps)
  - [Step 1: Preprocessing](#step-1-preprocessing)
  - [Step 2: SAR Training](#step-2-sar-training)
  - [Step 3: CTE Training & Evaluation](#step-3-cte-training--evaluation)
- [Additional Experiments](#additional-experiments)
- [Reproducing All Results](#reproducing-all-results)
- [Computational Requirements](#computational-requirements)
- [Results Summary](#results-summary)

---

## Overview

MoSEs (Mixture of Stylistics Experts) is a calibration and routing framework for AI-generated text detection. It operates on top of any base detector and improves performance by:
1. Grouping texts by stylistic similarity using a **Stylistics-Aware Router (SAR)**
2. Fitting a **Conditional Threshold Estimator (CTE)** per style cluster

We reproduce the main results on 4 main datasets and 4 low-resource datasets using 3 base detectors (RoBERTa, Fast-DetectGPT, Lastde), and additionally run experiments on:
- **Additional datasets** (TruthfulQA, HC3)
- **Alternative embedding models** (replacing BGE-M3 with E5-large, all-MiniLM)
- **New ablations** (training set size, PCA dimensions, neighborhood size k)
- **Hyperparameter tuning** (Sinkhorn ε, SAR epochs, CTE regularization)

---

## Original Paper & Repo

- **Paper**: [MoSEs: Mixture of Stylistics Experts for AI-Generated Text Detection](https://arxiv.org/abs/XXXX.XXXXX)
- **Original Code**: https://github.com/creator-xi/MoSEs
- **Our Repo**: This repository

---

## Our Contributions

| Contribution | Description |
|---|---|
| `scripts/preprocess.py` | Unified preprocessing for all datasets + new datasets |
| `scripts/train_sar.py` | SAR training with configurable hyperparameters |
| `scripts/train_cte.py` | CTE training (lr + xgb) |
| `scripts/evaluate.py` | Evaluation with full metrics logging |
| `scripts/run_all.sh` | End-to-end reproduction script |
| `experiments/` | All additional experiment scripts |
| `results/` | Saved result JSONs and tables |

---

## Dependencies

### Requirements

```bash
# Python 3.10 recommended
conda create -n moses python=3.10
conda activate moses

# Install base dependencies
pip install -r requirements.txt

# For Fast-DetectGPT and Lastde (proxy model inference)
pip install transformers==4.38.0 accelerate

# For BGE-M3 embeddings
pip install FlagEmbedding

# For XGBoost CTE
pip install xgboost==2.0.0

# For additional experiments
pip install sentence-transformers  # E5-large, MiniLM alternatives
```

### `requirements.txt`

```
torch>=2.0.0
transformers>=4.38.0
accelerate>=0.27.0
FlagEmbedding>=1.2.0
xgboost>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
sentence-transformers>=2.6.0
datasets>=2.18.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```

> **GPU Note**: You need ~14GB VRAM for GPT-Neo-2.7B (Fast-DetectGPT/Lastde) + BGE-M3 simultaneously. Use `--fp16` flag if memory is tight.

---

## Data Download

### Original MoSEs Datasets

```bash
# Download from the original repo's provided data links
python scripts/download_data.py --datasets all
# This downloads: CMV, SciXGen, WP, XSum (main) + CNN, DialogSum, IMDB, PubMedQA (low-resource)
# Data is saved to: data/raw/
```

Or manually from the original repo:
```bash
git clone https://github.com/creator-xi/MoSEs
cp -r MoSEs/data ./data/raw
```

### Additional Datasets (Our Extensions)

```bash
# TruthfulQA and HC3 (additional experiment datasets)
python scripts/download_data.py --datasets additional
# Downloads TruthfulQA from HuggingFace and HC3 from HuggingFace
```

Expected directory layout after download:
```
data/
  raw/
    CMV/          # train.json, test.json
    SciXGen/
    WP/
    XSum/
    CNN/
    DialogSum/
    IMDB/
    PubMedQA/
    TruthfulQA/   # our addition
    HC3/          # our addition
```

---

## Project Structure

```
moses-repro/
├── README.md
├── requirements.txt
├── configs/
│   ├── main_datasets.yaml       # Dataset paths and split sizes
│   ├── base_detectors.yaml      # Base detector configurations
│   └── hyperparams.yaml         # Default hyperparameters
├── data/
│   ├── raw/                     # Downloaded raw data
│   └── processed/               # Preprocessed features + embeddings
├── scripts/
│   ├── download_data.py         # Data download utility
│   ├── preprocess.py            # Feature extraction + embedding computation
│   ├── train_sar.py             # SAR training
│   ├── train_cte.py             # CTE training (lr + xg)
│   ├── evaluate.py              # Evaluation script
│   ├── run_pipeline.py          # Full end-to-end pipeline
│   └── run_all.sh               # Shell script to reproduce all results
├── experiments/
│   ├── additional_datasets/
│   │   └── run_additional.py    # HC3 + TruthfulQA experiments
│   ├── method_exploration/
│   │   └── alt_embeddings.py    # E5-large, MiniLM as BGE-M3 alternatives
│   ├── ablations/
│   │   ├── training_size.py     # Vary reference set size
│   │   ├── pca_dims.py          # Vary PCA dimensions (8, 16, 32, 64)
│   │   └── neighborhood_k.py    # Vary SAR neighborhood k
│   └── hyperparameter_tuning/
│       └── tune_hyperparams.py  # Grid search over ε, epochs, CTE params
├── results/
│   ├── main_results.json
│   ├── lowresource_results.json
│   └── ablation_results/
├── notebooks/
│   └── analysis.ipynb           # Result visualization and analysis
└── tests/
    └── test_pipeline.py         # Smoke tests
```

---

## Pipeline Steps

### Step 1: Preprocessing

Computes (a) base detector discrimination scores, (b) conditional features, and (c) semantic embeddings for all dataset samples.

```bash
# Preprocess a single dataset with one base detector
python scripts/preprocess.py \
    --dataset CMV \
    --detector roberta \
    --data_dir data/raw \
    --output_dir data/processed \
    --pca_dims 32 \
    --device cuda

# Preprocess all datasets with all detectors
python scripts/preprocess.py \
    --dataset all \
    --detector all \
    --data_dir data/raw \
    --output_dir data/processed \
    --pca_dims 32 \
    --device cuda \
    --fp16  # Use if GPU memory < 16GB

# Preprocess only with RoBERTa (no GPT-Neo needed, faster)
python scripts/preprocess.py \
    --dataset all \
    --detector roberta \
    --data_dir data/raw \
    --output_dir data/processed
```

**What this computes per sample:**
- Discrimination score from base detector
- Text length, mean/variance of token log-probs
- 2-gram and 3-gram repetition rates
- Type-token ratio
- 32D PCA projection of BGE-M3 (1024D) embedding

**Output**: `data/processed/{dataset}_{detector}_ref.pkl` and `..._test.pkl`

---

### Step 2: SAR Training

Trains the Stylistics-Aware Router on reference set embeddings.

```bash
# Train SAR for a single dataset
python scripts/train_sar.py \
    --dataset CMV \
    --detector roberta \
    --processed_dir data/processed \
    --output_dir results/sar_models \
    --epochs 100 \
    --epsilon 0.05 \
    --device cuda

# Train SAR for all datasets and detectors
python scripts/train_sar.py \
    --dataset all \
    --detector all \
    --processed_dir data/processed \
    --output_dir results/sar_models \
    --epochs 100 \
    --epsilon 0.05
```

---

### Step 3: CTE Training & Evaluation

Trains CTE (logistic regression or XGBoost) and evaluates on the test set.

```bash
# Train and evaluate MoSEs-lr
python scripts/train_cte.py \
    --dataset CMV \
    --detector roberta \
    --cte_type lr \
    --processed_dir data/processed \
    --sar_dir results/sar_models \
    --output_dir results/

# Train and evaluate MoSEs-xg
python scripts/train_cte.py \
    --dataset CMV \
    --detector roberta \
    --cte_type xgb \
    --processed_dir data/processed \
    --sar_dir results/sar_models \
    --output_dir results/

# Evaluate all combinations
python scripts/train_cte.py \
    --dataset all \
    --detector all \
    --cte_type both \
    --processed_dir data/processed \
    --sar_dir results/sar_models \
    --output_dir results/
```

---

## Additional Experiments

### Additional Datasets

```bash
python experiments/additional_datasets/run_additional.py \
    --datasets TruthfulQA HC3 \
    --detector roberta \
    --output_dir results/additional/
```

### Method Exploration: Alternative Embeddings

```bash
python experiments/method_exploration/alt_embeddings.py \
    --dataset all \
    --detector roberta \
    --embedding_models bge-m3 e5-large minilm \
    --output_dir results/alt_embeddings/
```

### Ablations

```bash
# Training size ablation (10%, 25%, 50%, 75%, 100% of reference set)
python experiments/ablations/training_size.py \
    --dataset CMV SciXGen WP XSum \
    --detector roberta \
    --sizes 0.1 0.25 0.5 0.75 1.0 \
    --output_dir results/ablations/training_size/

# PCA dimension ablation (8, 16, 32, 64, 128, no-PCA)
python experiments/ablations/pca_dims.py \
    --dataset all \
    --detector roberta \
    --pca_dims 8 16 32 64 128 \
    --output_dir results/ablations/pca_dims/

# Neighborhood size k ablation
python experiments/ablations/neighborhood_k.py \
    --dataset all \
    --detector roberta \
    --k_values 5 10 20 50 100 \
    --output_dir results/ablations/neighborhood_k/
```

### Hyperparameter Tuning

```bash
python experiments/hyperparameter_tuning/tune_hyperparams.py \
    --dataset CMV \
    --detector roberta \
    --param epsilon \
    --values 0.01 0.05 0.1 0.2 0.5 \
    --output_dir results/hparam_tuning/
```

---

## Reproducing All Results

To run the full reproduction in one command:

```bash
# Make sure data is downloaded first
python scripts/download_data.py --datasets all

# Full pipeline (all datasets, all detectors, both CTE variants)
bash scripts/run_all.sh

# Results will be in results/
python scripts/evaluate.py --summarize --results_dir results/
```

Estimated time: ~5-8 GPU hours on a T4. See [Computational Requirements](#computational-requirements).

---

## Computational Requirements

| Step | Estimated Time | Actual Time | Hardware |
|---|---|---|---|
| Data preprocessing (RoBERTa) | ~45 min | TBD | T4 GPU |
| Data preprocessing (GPT-Neo-2.7B) | ~2-3 hours | TBD | T4 GPU |
| BGE-M3 embeddings (all datasets) | ~1 hour | TBD | T4 GPU |
| SAR training (all) | ~20 min | TBD | T4 GPU |
| CTE training + eval (all) | ~5 min | TBD | CPU |
| **Total** | **~4-5 hours** | **TBD** | T4 GPU |

- **GPU VRAM**: ~14GB peak (GPT-Neo-2.7B in fp16)
- **Disk**: ~2-3GB for all processed data
- **RAM**: ~16GB recommended

---

## Results Summary

See `results/` directory and `notebooks/analysis.ipynb` for full tables.

| Dataset | Base Detector | Original Acc. | Our Acc. | Δ |
|---|---|---|---|---|
| CMV | RoBERTa | TBD | TBD | - |
| SciXGen | RoBERTa | TBD | TBD | - |
| ... | ... | ... | ... | - |

---

## Citation

```bibtex
@article{moses2024,
  title={MoSEs: Mixture of Stylistics Experts for AI-Generated Text Detection},
  author={...},
  year={2024}
}
```
