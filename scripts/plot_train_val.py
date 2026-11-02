"""Part 2: training vs. validation loss for each model (train_vs_val.png)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["rnn", "gru", "lstm", "lstm_512", "gpt_textbook", "gpt_modern"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out-dir", default="results/part2")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr, ev, meta = [], [], {}
    for mf in sorted(Path(a.sweep_dir).glob("*/metrics.json")):
        m = json.loads(mf.read_text())
        d = mf.parent
        tps = m["tokens_seen"] / max(m["steps"], 1)
        meta[(m["name"], m["seed"])] = m
        for line in (d / "train_log.jsonl").read_text().splitlines():
            r = json.loads(line)
            tr.append({"name": m["name"], "seed": m["seed"], "tokens": r["step"] * tps, "loss": r["loss"]})
        for line in (d / "eval_log.jsonl").read_text().splitlines():
            r = json.loads(line)
            ev.append({"name": m["name"], "seed": m["seed"], "tokens": r["train_tokens"], "loss": r["loss"],
                       "bpb": r["bpb"], "step": r["step"]})
    tr = pd.DataFrame(tr, columns=["name", "seed", "tokens", "loss"])
    ev = pd.DataFrame(ev, columns=["name", "seed", "tokens", "loss", "bpb", "step"])
    names = [n for n in ORDER if n in set(ev.name)] + sorted(set(ev.name) - set(ORDER))
    cols = 3
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), sharey=True, squeeze=False)
    for ax, n in zip(axes.flat, names):
        t = tr[tr.name == n].groupby("tokens")["loss"].mean()
        v = ev[ev.name == n].groupby("tokens")["loss"].mean()
        ax.plot(t.index / 1e6, t.values, label="train loss (dropout on)", alpha=0.8)
        ax.plot(v.index / 1e6, v.values, label="val loss", marker=".")
        ax.set_title(n)
        ax.set_xlabel("training tokens seen (millions)")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for r in range(rows):
        axes[r][0].set_ylabel("loss (nats / char)")
    axes[0][0].legend(fontsize=8)
    lo = min(tr.loss.min() if len(tr) else ev.loss.min(), ev.loss.min())
    axes[0][0].set_ylim(lo - 0.05, lo + 1.2)
    fig.suptitle("Tiny Shakespeare: train vs validation loss (mean over seeds)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "train_vs_val.png", dpi=150)
    plt.close(fig)

    print(f"wrote {out}/train_vs_val.png")


if __name__ == "__main__":
    main()
