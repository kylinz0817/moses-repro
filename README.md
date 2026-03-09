# MoSEs Reproduction Project

This repository contains our reproduction workflow for:

**MoSEs: Mixture of Stylistics Experts for AI-Generated Text Detection**  
Original paper repo: https://github.com/creator-xi/MoSEs

## Repository overview

This repository is organized into two separate workflows:

1. `notebooks/original_reproduction.ipynb`  
   Reproduces the official released MoSEs fast-detect pipeline using the original authors’ repository.

2. `notebooks/additional_experiments.ipynb`  
   Runs our additional ablations, new-dataset experiments, and figures using cached outputs from the original pipeline.

## What is original vs. what is ours

- **Original code**: https://github.com/creator-xi/MoSEs
- **Our code**: this repository
- We use the original authors’ released code for the official reproduction.
- We add notebooks, helper scripts, logging utilities, and additional experiment code.

## Environment

We recommend Google Colab with GPU for the original reproduction.

Local / Conda setup:

```bash
conda create -n moses-repro python=3.10
conda activate moses-repro
pip install -r requirements.txt
