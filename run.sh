#!/usr/bin/env bash
# Runs every HW1 experiment and makes all tables and figures in results/.
# Run it from the course repo:  cd DeepDataMiningLearning/DeepDataMiningLearning/llm && bash ../../../hw1/run.sh
set -e
H=../../../hw1
S=outputs/nanolm/sweeps

# training (finished runs are skipped)
python -m nanolm.sweep --sweep $H/sweeps/hw1_char_models.yaml
python -m nanolm.sweep --sweep $H/sweeps/hw1_context.yaml
python -m nanolm.sweep --sweep $H/sweeps/hw1_untrained.yaml

# part 1
python $H/scripts/metrics_by_hand.py $S/hw1_char_models/gpt_modern_s1337 --out $H/results/part1/metrics_by_hand.md
python $H/scripts/untrained_vs_lnV.py $S/hw1_untrained --out $H/results/part1/untrained_vs_lnV.md
# part 2
python -m nanolm.analyze $S/hw1_char_models --baseline gpt_textbook --metric val_bpb --out $H/results/part2
python $H/scripts/plot_train_val.py $S/hw1_char_models --out-dir $H/results/part2
# part 3
python -m nanolm.analyze $S/hw1_context $S/hw1_char_models --metric val_bpb --out $H/results/part3
python $H/scripts/plot_context.py --char-models $S/hw1_char_models --context $S/hw1_context --out-dir $H/results/part3
# part 4
python $H/scripts/attention_heads.py --run $S/hw1_char_models/gpt_modern_s1337 --out-dir $H/results/part4
# extra credit
python $H/scripts/gru_scratch.py --out $H/results/extra_credit/gru_check.txt
