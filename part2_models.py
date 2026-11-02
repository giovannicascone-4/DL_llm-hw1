# Part 2: RNN vs GRU vs LSTM vs Transformer. Results table, learning curves, train vs. validation loss.
# Run from DeepDataMiningLearning/DeepDataMiningLearning/llm
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("outputs/nanolm/sweeps/hw1_char_models")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
MODELS = ["rnn", "gru", "lstm", "lstm_512", "gpt_textbook", "gpt_modern"]


def read_log(path):
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines()])


final, val, train = [], [], []
for run in sorted(RUNS.glob("*_s*")):
    m = json.loads((run / "metrics.json").read_text())
    final.append(m)
    v = read_log(run / "eval_log.jsonl")
    v["name"] = m["name"]
    val.append(v)
    t = read_log(run / "train_log.jsonl")
    t["name"] = m["name"]
    t["train_tokens"] = t["step"] * m["tokens_seen"] / m["steps"]
    train.append(t)
final, val, train = pd.DataFrame(final), pd.concat(val), pd.concat(train)

# Results table: mean and std of the final validation bpb over the 2 seeds
table = final.groupby("name").agg(params=("params", "first"), bpb_mean=("val_bpb", "mean"),
                                  bpb_std=("val_bpb", "std")).sort_values("bpb_mean")
table.to_csv(OUT / "part2_summary.csv")
lines = ["## Results (final validation bpb, 2 seeds)", "", "| model | parameters | val bpb |", "|---|---:|---:|"]
for name, r in table.iterrows():
    lines.append(f"| {name} | {r.params / 1e6:.2f}M | {r.bpb_mean:.3f} ± {r.bpb_std:.3f} |")
(OUT / "part2.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# Figure: learning curves (validation bpb vs. training tokens, mean of the 2 seeds)
plt.figure(figsize=(7, 4.5))
for name in MODELS:
    curve = val[val.name == name].groupby("train_tokens")["bpb"].mean()
    plt.plot(curve.index / 1e6, curve.values, label=name)
plt.xlabel("training tokens (millions)")
plt.ylabel("validation bits per byte")
plt.ylim(2.05, 3.3)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "part2_learning_curves.png", dpi=150)

# Figure: training vs. validation loss for each model
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
for ax, name in zip(axes.flat, MODELS):
    t = train[train.name == name].groupby("train_tokens")["loss"].mean()
    v = val[val.name == name].groupby("train_tokens")["loss"].mean()
    ax.plot(t.index / 1e6, t.values, label="train loss")
    ax.plot(v.index / 1e6, v.values, label="validation loss")
    ax.set_title(name)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylim(0.95, 2.2)
    ax.grid(alpha=0.3)
axes[0, 0].set_ylabel("loss (nats per character)")
axes[1, 0].set_ylabel("loss (nats per character)")
axes[0, 0].legend()
plt.tight_layout()
plt.savefig(OUT / "part2_train_vs_val.png", dpi=150)
