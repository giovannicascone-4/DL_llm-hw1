"""Part 3: bpb vs. context length for the LSTM and the Transformer (context_bpb.png, context_table.md)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FAMILY = {"lstm": ("lstm", 256), "gpt_modern": ("transformer (modern)", 256)}


def classify(name: str):
    if name in FAMILY:
        return FAMILY[name]
    if name.startswith("lstm_ctx"):
        return "lstm", int(name[len("lstm_ctx"):])
    if name.startswith("gpt_ctx"):
        return "transformer (modern)", int(name[len("gpt_ctx"):])
    return None


def load(dirs):
    rows, curves = [], []
    for d in dirs:
        for mf in sorted(Path(d).glob("*/metrics.json")):
            m = json.loads(mf.read_text())
            c = classify(m["name"])
            if c is None:
                continue
            m["family"], m["ctx"] = c
            rows.append(m)
            el = mf.parent / "eval_log.jsonl"
            if el.exists():
                for line in el.read_text().splitlines():
                    e = json.loads(line)
                    curves.append({"family": c[0], "ctx": c[1], "seed": m["seed"],
                                   "train_tokens": e["train_tokens"], "bpb": e["bpb"]})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char-models", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--out-dir", default="results/part3")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, cv = load([a.char_models, a.context])
    if df.empty:
        raise SystemExit("no runs found")

    g = df.groupby(["family", "ctx"])["val_bpb"].agg(["mean", "std", "count"]).reset_index()
    noise = g["std"].dropna().median() if g["std"].notna().any() else float("nan")

    # ---- table with the step-to-step gain and whether it is resolved (> 2x seed noise floor)
    lines = [f"metric: final val_bpb on the full validation split (windows of seq_len). "
             f"Seed noise floor (median std) = {noise:.4f} bpb; a gain is 'resolved' if > 2x that.", "",
             "| model | seq_len | n seeds | val bpb (mean ± std) | gain vs previous seq_len | resolved? |",
             "|---|---:|---:|---:|---:|:---:|"]
    for fam, sub in g.groupby("family"):
        sub = sub.sort_values("ctx")
        prev = None
        for _, r in sub.iterrows():
            std = "" if math.isnan(r["std"]) else f" ± {r['std']:.4f}"
            if prev is None:
                gain, res = "—", "—"
            else:
                d = prev - r["mean"]
                gain = f"{d:+.4f}"
                res = "n/a" if math.isnan(noise) else ("yes" if abs(d) > 2 * noise else "not resolved")
            lines.append(f"| {fam} | {int(r['ctx'])} | {int(r['count'])} | {r['mean']:.4f}{std} | {gain} | {res} |")
            prev = r["mean"]
    (out / "context_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- summary figure: bpb vs context
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for fam, sub in g.groupby("family"):
        sub = sub.sort_values("ctx")
        ax.errorbar(sub["ctx"], sub["mean"], yerr=sub["std"].fillna(0), marker="o", capsize=4, label=fam)
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512])
    ax.set_xticklabels(["64", "128", "256", "512"])
    ax.set_xlabel("training / evaluation context length, train.seq_len (characters)")
    ax.set_ylabel("final validation bits-per-byte (lower is better)")
    ax.set_title("Tiny Shakespeare, same token budget per run (mean ± std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "context_bpb.png", dpi=150)
    plt.close(fig)

    print(f"\nwrote {out}/context_bpb.png, context_table.md")


if __name__ == "__main__":
    main()
