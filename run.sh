#!/usr/bin/env bash
# Runs all HW1 experiments and writes the tables and figures to results/.
# Run from the course repo:  cd DeepDataMiningLearning/DeepDataMiningLearning/llm && bash ../../../hw1/run.sh
set -e
H=../../../hw1

python -m nanolm.sweep --sweep $H/configs/part1_untrained.yaml
python -m nanolm.sweep --sweep $H/configs/part2_models.yaml
python -m nanolm.sweep --sweep $H/configs/part3_context.yaml

python $H/part1_metrics.py
python $H/part2_models.py
python $H/part3_context.py
python $H/part4_attention.py
python $H/extra_credit_gru.py
