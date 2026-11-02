# HW1: Language Models from Scratch (Part 1)

Giovanni Cascone, San José State University. Report: `HW1_Report_Giovanni_Cascone.pdf`

Uses the course code [nanolm](https://github.com/lkk688/DeepDataMiningLearning) (commit a4d756e) without changes.

| file | what it does |
|---|---|
| `part1_metrics.py` | Part 1: metrics by hand, untrained loss vs. ln V |
| `part2_models.py` | Part 2: results table, learning curves, train vs. validation loss |
| `part3_context.py` | Part 3: bpb vs. context length |
| `part4_attention.py` | Part 4: attention maps and per-head attention weights |
| `extra_credit_gru.py` | Extra credit: GRU written gate by gate, checked against `nn.GRU` |
| `configs/` | the experiment configs for parts 1, 2 and 3 |
| `results/` | all tables (`.md`, `.csv`) and figures (`.png`), named by part |
| `run.sh` | runs everything |

To reproduce: clone the course repo next to this folder, then

```bash
cd DeepDataMiningLearning/DeepDataMiningLearning/llm
bash ../../../hw1/run.sh
```
