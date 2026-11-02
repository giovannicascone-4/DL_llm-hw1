#!/usr/bin/env bash
# HW1 runner. Every step is resumable: finished runs (metrics.json present) are skipped.
#
#   bash hw1/run_all.sh all          # everything, in order
#   bash hw1/run_all.sh part2        # one part: train | part1 | part2 | part3 | part4 | ec | info
#   STEPS=1000 bash hw1/run_all.sh all        # laptop version: half the token budget for EVERY model
#   SEEDS="1337" bash hw1/run_all.sh part3    # one seed only (faster; weaker statistics)
#   PAR_GPU=8 PAR_CPU=4 RNN_DEVICE=cpu bash hw1/run_all.sh all   # parallelism knobs (defaults: 6, 4, auto)
#
# Order matters: part1a and part4 use the trained gpt_modern run from part2; part3 reuses part2's seq_len=256 runs.
set -euo pipefail
HW1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HW1")"
VENV="${VENV:-$HOME/venvs/nanolm}"
LLM="$ROOT/DeepDataMiningLearning/DeepDataMiningLearning/llm"
RES="$HW1/results"
STEPS="${STEPS:-2000}"
SEEDS="${SEEDS:-}"
# Use the whole Mac. These models are tiny (0.3-4M params), so one run cannot keep the GPU or the CPU busy.
# Training runs in TWO pools at the same time:
#   GPU pool : Transformer runs on MPS, PAR_GPU of them at once (sweep.py "--gpus 0,0,..": N slots, same device)
#   RNN pool : RNN/GRU/LSTM runs, PAR_CPU at once, on RNN_DEVICE = cpu | mps | auto
#              auto = a 5-step benchmark of the LSTM-512 (the slowest run) decides whether the CPU cores are
#              worth using for recurrent models or whether they are better off on the GPU too.
# CPU threads are split so nothing is oversubscribed. 48 GB is plenty: each run needs < ~1.5 GB.
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
PAR_GPU="${PAR_GPU:-6}"
PAR_CPU="${PAR_CPU:-4}"
RNN_DEVICE="${RNN_DEVICE:-auto}"
CPU_THREADS=$(( (NCPU - PAR_GPU) / PAR_CPU )); [ "$CPU_THREADS" -ge 1 ] || CPU_THREADS=1
slots() { local s; s="$(printf '0,%.0s' $(seq 1 "$1"))"; echo "${s%,}"; }
source "$VENV/bin/activate"
export PYTORCH_ENABLE_MPS_FALLBACK=1
# python.org Python on macOS ships without trusted root certificates -> every HTTPS download fails
# (CERTIFICATE_VERIFY_FAILED). Point Python at certifi's CA bundle instead.
pip install -q certifi
export SSL_CERT_FILE="$(python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PYTHONPATH="$LLM${PYTHONPATH:+:$PYTHONPATH}"
cd "$LLM"                           # nanolm resolves data/ and outputs/ relative to here
mkdir -p "$RES"/{part1,part2,part3,part4,extra_credit}
SW=outputs/nanolm/sweeps

steps_set=()
[ "$STEPS" != "2000" ] && steps_set=("train.max_steps=$STEPS")
seeds=()
[ -n "$SEEDS" ] && seeds=(--seeds $SEEDS)
ts() { date "+%H:%M:%S"; }
tag() { local l; while IFS= read -r l; do printf '%s %s\n' "$1" "$l"; done; }   # prefix lines, unbuffered

pick_rnn_device() {   # RNN_DEVICE=auto: is the LSTM-512 faster per run on a CPU slice or on the (shared) GPU?
  [ "$RNN_DEVICE" != "auto" ] && return
  local cfg=(-c nanolm/configs/01_char_rnn.yaml model.dim=512 train.seq_len=256 train.batch_size=32)
  local c g
  c=$(OMP_NUM_THREADS=$CPU_THREADS python -m nanolm.bench "${cfg[@]}" device=cpu --steps 5 2>/dev/null | sed -n 's/.*| \([0-9.]*\)k tokens\/s.*/\1/p')
  g=$(python -m nanolm.bench "${cfg[@]}" --steps 5 2>/dev/null | sed -n 's/.*| \([0-9.]*\)k tokens\/s.*/\1/p')
  echo "[hw1] LSTM-512 benchmark: cpu ($CPU_THREADS threads) ${c:-?}k tok/s vs gpu alone ${g:-?}k tok/s"
  # the GPU will be shared by PAR_GPU transformer runs, so a CPU slot at >= 1/2 of a lone GPU run is a net win
  if [ -n "$c" ] && [ -n "$g" ] && awk -v c="$c" -v g="$g" 'BEGIN{exit !(c >= g/2)}'; then RNN_DEVICE=cpu; else RNN_DEVICE=mps; fi
}

sweep() {   # sweep <yaml> <pool: gpu|rnn> [--only names...]
  local yaml="$1" pool="$2"; shift 2
  local n=$PAR_GPU set=(${steps_set[@]+"${steps_set[@]}"}) thr=1
  if [ "$pool" = rnn ]; then
    n=$PAR_CPU
    if [ "$RNN_DEVICE" = cpu ]; then set+=("device=cpu"); thr=$CPU_THREADS; fi
  fi
  local extra=(); [ ${#set[@]} -gt 0 ] && extra=(--set "${set[@]}")
  OMP_NUM_THREADS=$thr python -m nanolm.sweep --gpus "$(slots "$n")" --sweep "$HW1/sweeps/$yaml" "$@" \
    ${extra[@]+"${extra[@]}"} ${seeds[@]+"${seeds[@]}"}
}

train_all() {   # every training run of HW1, both pools at once
  pick_rnn_device
  echo "[$(ts)] TRAINING: GPU pool $PAR_GPU x transformer | RNN pool $PAR_CPU x recurrent on $RNN_DEVICE" \
       "($CPU_THREADS threads each if cpu) | $NCPU cores | $STEPS steps per run"
  ( sweep hw1_char_models.yaml rnn --only rnn gru lstm lstm_512
    sweep hw1_context.yaml rnn --only lstm_ctx64 lstm_ctx128 lstm_ctx512 ) 2>&1 | tag "[rnn]" &
  local p1=$!
  ( sweep hw1_char_models.yaml gpu --only gpt_textbook gpt_modern
    sweep hw1_context.yaml gpu --only gpt_ctx64 gpt_ctx128 gpt_ctx512
    sweep hw1_untrained.yaml gpu ) 2>&1 | tag "[gpu]" &
  local p2=$!
  wait $p1; wait $p2
  local n; n=$(ls $SW/hw1_char_models/*/metrics.json $SW/hw1_context/*/metrics.json $SW/hw1_untrained/*/metrics.json 2>/dev/null | wc -l)
  echo "[$(ts)] TRAINING done: $n finished runs (expected 32 with 2 seeds). Rerun the same command to retry any failures."
}

prep() {   # build data + tokenizer caches ONCE, so parallel runs never write the same cache file together
  python - <<'EOF'
from nanolm.config import load_config
from nanolm.data import prepare_dataset
for tok in ["char", "byte", "sp_train:bpe:512", "word:2048"]:
    m = prepare_dataset(load_config("nanolm/configs/base.yaml", [f"data.tokenizer={tok}"]).data)
    print(f"[prep] {tok}: vocab {m['vocab_size']}")
EOF
}

part2() {
  echo "[$(ts)] PART 2: RNN vs GRU vs LSTM vs Transformer (6 models x 2 seeds, $STEPS steps each)"
  sweep hw1_char_models.yaml rnn --only rnn gru lstm lstm_512 2>&1 | tag "[rnn]" &
  sweep hw1_char_models.yaml gpu --only gpt_textbook gpt_modern 2>&1 | tag "[gpu]"
  wait
  python -m nanolm.analyze $SW/hw1_char_models --baseline gpt_textbook --metric val_bpb --out "$RES/part2" \
    | tee "$RES/part2/analyze.txt"
  python "$HW1/scripts/plot_train_val.py" $SW/hw1_char_models --out-dir "$RES/part2"
}

part1() {
  echo "[$(ts)] PART 1a: metrics by hand (trained char Transformer, gpt_modern seed 1337)"
  local run=$SW/hw1_char_models/gpt_modern_s1337
  [ -f "$run/metrics.json" ] || { echo "  needs part2 first ($run)"; return 1; }
  python "$HW1/scripts/metrics_by_hand.py" "$run" --out "$RES/part1/metrics_by_hand_gpt_modern.md"
  python "$HW1/scripts/metrics_by_hand.py" $SW/hw1_char_models/lstm_s1337 --out "$RES/part1/metrics_by_hand_lstm.md" || true
  echo "[$(ts)] PART 1b: untrained models (train.max_steps=1) for 4 tokenizers vs ln V"
  sweep hw1_untrained.yaml gpu
  python "$HW1/scripts/untrained_vs_lnV.py" $SW/hw1_untrained --out "$RES/part1/untrained_vs_lnV.md"
}

part3() {
  echo "[$(ts)] PART 3: context length 64/128/512 (256 comes from part2)"
  sweep hw1_context.yaml rnn --only lstm_ctx64 lstm_ctx128 lstm_ctx512 2>&1 | tag "[rnn]" &
  sweep hw1_context.yaml gpu --only gpt_ctx64 gpt_ctx128 gpt_ctx512 2>&1 | tag "[gpu]"
  wait
  python "$HW1/scripts/plot_context.py" --char-models $SW/hw1_char_models --context $SW/hw1_context --out-dir "$RES/part3"
  python -m nanolm.analyze $SW/hw1_context $SW/hw1_char_models --metric val_bpb --out "$RES/part3/analyze_all" \
    > "$RES/part3/analyze_all.txt"
}

part4() {
  echo "[$(ts)] PART 4: attention analysis on 3 kinds of text"
  python "$HW1/scripts/attention_heads.py" --run $SW/hw1_char_models/gpt_modern_s1337 --out-dir "$RES/part4"
}

ec() {
  echo "[$(ts)] EXTRA CREDIT: GRU gate by gate vs nn.GRU"
  python "$HW1/scripts/gru_scratch.py" --out "$RES/extra_credit/gru_check.txt"
}

info() {
  python "$HW1/scripts/run_info.py" $SW/hw1_char_models $SW/hw1_context $SW/hw1_untrained --out "$RES/run_info.md"
  # submission copies of the configs (the report must ship them)
  mkdir -p "$RES/configs"
  cp "$HW1"/sweeps/*.yaml "$RES/configs/"
  cp nanolm/configs/base.yaml nanolm/configs/02_char_transformer.yaml nanolm/configs/02b_char_transformer_modern.yaml \
     nanolm/configs/laptop/L3_tokenizer.yaml "$RES/configs/"
}

prep
case "${1:-all}" in
  train) train_all ;; part1) part1 ;; part2) pick_rnn_device; part2 ;; part3) pick_rnn_device; part3 ;; part4) part4 ;; ec) ec ;; info) info ;;
  all) train_all; part2; part1; part3; part4; ec; info ;;   # parts only analyze here: their runs are done
  *) echo "usage: $0 [all|train|part1|part2|part3|part4|ec|info]"; exit 2 ;;
esac
echo "[$(ts)] done -> $RES"
