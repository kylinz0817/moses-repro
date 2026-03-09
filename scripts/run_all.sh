#!/bin/bash
set -e
DEVICE=${DEVICE:-cuda}
python scripts/preprocess.py --dataset all --detector roberta --device $DEVICE
python scripts/train_sar.py --dataset all --detector roberta
python scripts/train_cte.py --dataset all --detector roberta --cte_type lr xgb
