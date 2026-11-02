"""HW1 part 2 (question: is the larger LSTM better or worse, and what in your curves shows why?)

Train loss vs validation loss, both in nats/char, against training tokens, one panel per model.
A widening gap with val loss turning UP = overfitting (memorizing the 1 MB corpus).

    python ../../../hw1/scripts/plot_train_val.py outputs/nanolm/sweeps/hw1_char_models --out-dir results/part2

Note: train loss is measured with dropout on, so a small train > val gap early on is normal.
Writes train_vs_val.png and best_vs_final.md (step of the best periodic val bpb vs the final one).
"""
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

    lines = ["| model | seed | best periodic val bpb | at tokens (M) | final periodic val bpb | final - best | "
             "last train loss | last val loss | gap (val - train) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        for s in sorted(ev[ev.name == n].seed.unique()):
            e = ev[(ev.name == n) & (ev.seed == s)].sort_values("tokens")
            t = tr[(tr.name == n) & (tr.seed == s)].sort_values("tokens")
            b = e.loc[e.bpb.idxmin()]
            f = e.iloc[-1]
            lt = t.loss.iloc[-3:].mean() if len(t) else float("nan")      # average of the last 3 logged train losses
            lines.append(f"| {n} | {s} | {b.bpb:.4f} | {b.tokens / 1e6:.1f} | {f.bpb:.4f} | {f.bpb - b.bpb:+.4f} | "
                         f"{lt:.3f} | {f.loss:.3f} | {f.loss - lt:+.3f} |")
    (out / "best_vs_final.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/train_vs_val.png, best_vs_final.md")


if __name__ == "__main__":
    main()
