# Part 1: recompute the metrics of a trained model by hand, and compare untrained models with ln V.
# Run from DeepDataMiningLearning/DeepDataMiningLearning/llm
import json
import math
import re
from pathlib import Path

RUNS = Path("outputs/nanolm/sweeps")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# 1a. Trained character model: recompute everything from val_loss, val_tokens and val_bytes
m = json.loads((RUNS / "hw1_char_models/gpt_modern_s1337/metrics.json").read_text())
loss = m["val_loss"]
perplexity = math.exp(loss)
bits_per_token = loss / math.log(2)
bpb = bits_per_token * m["val_tokens"] / m["val_bytes"]

lines = ["## Metrics by hand (modern Transformer, seed 1337)", "",
         "| metric | by hand | logged |", "|---|---:|---:|",
         f"| loss (nats/token) | {loss:.6f} | {m['val_loss']:.6f} |",
         f"| perplexity = e^loss | {perplexity:.6f} | {m['val_ppl']:.6f} |",
         f"| bits/token = loss / ln 2 | {bits_per_token:.6f} | - |",
         f"| bits per byte = bits/token x tokens / bytes | {bpb:.6f} | {m['val_bpb']:.6f} |", ""]

# 1b. Untrained models (train.max_steps=1): the loss should be close to ln V
lines += ["## Untrained models vs ln V", "",
          "| tokenizer | seed | V | ln V | untrained loss | loss / ln V |", "|---|---:|---:|---:|---:|---:|"]
for f in sorted((RUNS / "hw1_untrained").glob("*/metrics.json")):
    m = json.loads(f.read_text())
    V = int(re.search(r"vocab=(\d+)", (f.parent / "stdout.log").read_text()).group(1))
    lines.append(f"| {m['name']} | {m['seed']} | {V} | {math.log(V):.4f} | {m['val_loss']:.4f} | "
                 f"{m['val_loss'] / math.log(V):.3f} |")

(OUT / "part1.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
