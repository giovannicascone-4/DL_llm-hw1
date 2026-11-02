# HW1: Language Models from Scratch (Part 1)

Giovanni Cascone, San José State University

Report: `HW1_Report_Giovanni_Cascone.pdf`

All experiments use the course code [nanolm](https://github.com/lkk688/DeepDataMiningLearning) (commit a4d756e) without changes.
Trained on a MacBook Pro M5 Max (MPS).

## How to run

```bash
pip install torch numpy pyyaml sentencepiece tiktoken pandas matplotlib scipy tabulate
git clone https://github.com/lkk688/DeepDataMiningLearning.git   # next to this hw1 folder
cd DeepDataMiningLearning/DeepDataMiningLearning/llm
bash ../../../hw1/run.sh
```

## Files

| folder | contents |
|---|---|
| `sweeps/` | experiment configs: `hw1_untrained.yaml` (part 1), `hw1_char_models.yaml` (part 2), `hw1_context.yaml` (part 3) |
| `scripts/` | analysis and plotting scripts |
| `results/part1/` | metrics by hand, untrained loss vs. ln V |
| `results/part2/` | `summary.csv` / `summary.md`, learning curves, train vs. validation loss |
| `results/part3/` | `summary.csv` / `summary.md`, bpb vs. context length |
| `results/part4/` | attention maps for three texts, per-head attention weights |
| `results/extra_credit/` | GRU gate-by-gate check against `nn.GRU` |
