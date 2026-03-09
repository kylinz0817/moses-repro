#!/bin/bash
set -e
python experiments/ablations/training_size.py --dataset CMV SciXGen WP XSum --detector roberta
python experiments/ablations/pca_dims.py --dataset CMV SciXGen WP XSum --detector roberta
python experiments/ablations/neighborhood_k.py --dataset CMV SciXGen WP XSum --detector roberta
python experiments/hyperparameter_tuning/tune_hyperparams.py --dataset CMV SciXGen --detector roberta --param all
