#!/usr/bin/env bash
# HW1 one-time setup on your Mac: venv + packages + nanolm repo + tests + machine benchmark.
#   bash hw1/setup.sh            (run from the "LLM Homework1" folder, or from anywhere)
set -euo pipefail
HW1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HW1")"                                   # .../LLM Homework1
VENV="${VENV:-$HOME/venvs/nanolm}"
REPO="$ROOT/DeepDataMiningLearning"
LLM="$REPO/DeepDataMiningLearning/llm"

# 1. Python environment (arm64 Python >= 3.10; NOT Rosetta, or MPS will be missing)
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -c 'import platform,sys; print("python", sys.version.split()[0], platform.machine())'
pip install -q --upgrade pip
# python.org Python on macOS ships without trusted root certificates -> every HTTPS download fails
# (CERTIFICATE_VERIFY_FAILED). Point Python at certifi's CA bundle instead.
pip install -q certifi
export SSL_CERT_FILE="$(python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
pip install -q torch numpy pyyaml sentencepiece tiktoken pandas matplotlib scipy tabulate pytest

# 2. The course code
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/lkk688/DeepDataMiningLearning.git "$REPO"
fi
cd "$LLM"
git -C "$REPO" log -1 --format='repo commit: %H (%cd)'

# 3. Sanity checks (tests ~15 s; smoke run ~30 s, must print "device: mps")
python -m pytest nanolm/tests -q
python -m nanolm.train -c nanolm/configs/00_smoke.yaml 2>&1 | tee "$HW1/smoke.log" | grep -E "\[run\] device|\[final\]"

# 4. How fast is THIS Mac? (projected time for 2000 steps of each model family)
mkdir -p "$HW1/results"
{
  echo "== transformer (modern) seq 256 =="; python -m nanolm.bench -c nanolm/configs/laptop/L2_char_gpt.yaml
  echo "== LSTM 2x256 seq 256 ==";           python -m nanolm.bench -c nanolm/configs/01_char_rnn.yaml train.seq_len=256 train.batch_size=32
  echo "== LSTM 2x512 seq 256 ==";           python -m nanolm.bench -c nanolm/configs/01_char_rnn.yaml model.dim=512 train.seq_len=256 train.batch_size=32
} 2>&1 | tee "$HW1/results/bench.txt"
echo
echo "Setup done. Look at the 'projected' times above, then:  bash \"$HW1/run_all.sh\" all"
