# Part 3: validation bpb vs. context length (train.seq_len) for the LSTM and the modern Transformer.
# The seq_len=256 runs are the 'lstm' and 'gpt_modern' runs of Part 2 (same settings).
# Run from DeepDataMiningLearning/DeepDataMiningLearning/llm
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("outputs/nanolm/sweeps")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

rows = []
for f in list((RUNS / "hw1_context").glob("*/metrics.json")) + list((RUNS / "hw1_char_models").glob("*/metrics.json")):
    m = json.loads(f.read_text())
    name = m["name"]
    if name in ("lstm", "gpt_modern"):
        model, seq_len = ("LSTM" if name == "lstm" else "Transformer"), 256
    elif "_ctx" in name:
        model = "LSTM" if name.startswith("lstm") else "Transformer"
        seq_len = int(name.split("_ctx")[1])
    else:
        continue
    rows.append({"model": model, "seq_len": seq_len, "bpb": m["val_bpb"]})

table = pd.DataFrame(rows).groupby(["model", "seq_len"])["bpb"].agg(["mean", "std"]).reset_index()
lines = ["## Validation bpb vs. context length (2 seeds)", "", "| model | seq_len | val bpb |", "|---|---:|---:|"]
for _, r in table.iterrows():
    lines.append(f"| {r.model} | {r.seq_len} | {r['mean']:.3f} ± {r['std']:.3f} |")
(OUT / "part3.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

plt.figure(figsize=(6.5, 4.5))
for model, t in table.groupby("model"):
    plt.errorbar(t.seq_len, t["mean"], yerr=t["std"], marker="o", capsize=4, label=model)
plt.xscale("log", base=2)
plt.xticks([64, 128, 256, 512], ["64", "128", "256", "512"])
plt.xlabel("context length, train.seq_len (characters)")
plt.ylabel("validation bits per byte")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "part3_context_length.png", dpi=150)
