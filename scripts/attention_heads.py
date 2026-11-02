"""HW1 part 4: attention maps on three kinds of text + one number per head and pattern (as in §1.6.2).

    python ../../../hw1/scripts/attention_heads.py --run outputs/nanolm/sweeps/hw1_char_models/gpt_modern_s1337 \
        --out-dir results/part4

For each text it
  1. draws the maps with nanolm's own `viz attention` code (same as
     `python -m nanolm.viz attention --run <run> --text "..."`, but saved into --out-dir instead of
     overwriting docs/nanolm_figs/p1_attention_maps.png), and
  2. measures, for every layer x head, the average attention weight that a query position t puts on:
       prev        position t-1                          (previous-token head)
       self        position t
       word_start  first letter of the current word      (only t with t - start >= 2, so it is not prev/self)
       line_mark   the last ':' or newline before t      (only when t - pos >= 2)
       first       position 0                            (attention sink; t >= 2)
       induction   t - P + 1, the token AFTER the previous copy of x_t  (only for the 'repeat' text, period P)
     plus the mean look-back distance sum_s A[t,s] (t - s).
     Each pattern is also given its chance level: the mean of 1/(t+1) over the same query positions,
     i.e. what a head that spreads its weight uniformly over the visible prefix would score.

Outputs: attn_<kind>.png, head_stats_<kind>.csv, head_stats.md (all texts, + a 'best head per pattern' table).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())          # run from DeepDataMiningLearning/llm so `nanolm` imports
import nanolm.viz as viz  # noqa: E402
from nanolm.generate import load_run  # noqa: E402

# chars used here all occur in Tiny Shakespeare (unknown chars would be mapped to '?')
REPEAT_UNIT = "xqzvbkmwpt jrdlhs "
TEXTS = {
    "shakespeare": "ROMEO:\nBut soft! what light through yonder window breaks?\nJULIET:\nAy me!",
    "modern_prose": "The engineer trained a small language model on her laptop, and it wrote poems.",
    "repeat": REPEAT_UNIT * 3,
}


def positions(text: str):
    """Per query position t: start of current word, last ':'/newline before t."""
    T = len(text)
    word_start = np.full(T, -1)
    line_mark = np.full(T, -1)
    s, last = -1, -1
    for t, ch in enumerate(text):
        if ch.isalpha() or ch == "'":
            if s < 0:
                s = t
            word_start[t] = s
        else:
            s = -1
        line_mark[t] = last            # strictly before t
        if ch in ":\n":
            last = t
    return word_start, line_mark


def head_stats(maps, text: str, period: int | None):
    T = len(text)
    ws, lm = positions(text)
    t_all = np.arange(T)

    def score(A, pairs):
        if not pairs:
            return np.nan, np.nan
        w = np.mean([A[t, s] for t, s in pairs])
        chance = np.mean([1.0 / (t + 1) for t, _ in pairs])
        return w, chance

    patterns = {
        "prev": [(t, t - 1) for t in t_all if t >= 1],
        "self": [(t, t) for t in t_all if t >= 1],
        "word_start": [(t, ws[t]) for t in t_all if ws[t] >= 0 and t - ws[t] >= 2],
        "line_mark": [(t, lm[t]) for t in t_all if lm[t] >= 0 and t - lm[t] >= 2],
        "first": [(t, 0) for t in t_all if t >= 2],
    }
    if period:
        patterns["induction"] = [(t, t - period + 1) for t in t_all if t >= period]
    rows = []
    for l, layer in enumerate(maps):
        for h in range(layer.shape[0]):
            A = layer[h]
            r = {"layer": l, "head": h}
            for k, pairs in patterns.items():
                r[k], r[f"{k}_chance"] = score(A, pairs)
                r[f"{k}_n"] = len(pairs)
            r["lookback"] = float(np.mean([(A[t, : t + 1] * (t - np.arange(t + 1))).sum() for t in t_all[1:]]))
            rows.append(r)
    return pd.DataFrame(rows), list(patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir of a trained GPT (needs model.pt)")
    ap.add_argument("--out-dir", default="results/part4")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    viz.style()
    viz.FIG_DIR = out                         # keep the repo's tutorial figure untouched
    cfg, model, tok, dinfo, _ = load_run(a.run, device="cpu")
    if cfg.model.arch != "gpt":
        raise SystemExit("attention analysis needs a Transformer run (model.arch=gpt)")

    md = [f"Run: `{a.run}` ({cfg.model.n_layers} layers x {cfg.model.n_heads} heads, tokenizer {tok.spec}). "
          "Each cell: mean attention weight on the target position over the query positions where it is "
          "defined; `chance` = mean 1/(t+1) over the same positions (uniform over the visible prefix).", ""]
    best = []
    for kind, text in TEXTS.items():
        ids = tok.encode(text)
        if tok.spec == "char" and len(ids) != len(text):
            raise SystemExit("char tokenizer should give one id per character")
        unknown = sorted({c for c in text if tok.decode(tok.encode(c)) != c})
        if unknown:
            print(f"[warn] {kind}: characters not in the vocabulary (mapped to '?'): {unknown}")
        torch.manual_seed(0)
        viz.fig_attention(run_dir=a.run, text=text)
        shutil.move(str(out / "p1_attention_maps.png"), str(out / f"attn_{kind}.png"))

        maps = viz.attention_maps(model, torch.tensor([ids]))
        period = len(REPEAT_UNIT) if kind == "repeat" else None
        df, pats = head_stats(maps, text, period)
        df.to_csv(out / f"head_stats_{kind}.csv", index=False)

        md += [f"### {kind}  ({len(text)} chars)", "", "```", text, "```", ""]
        hdr = "| head | " + " | ".join(pats) + " | look-back (chars) |"
        md += [hdr, "|---|" + "---:|" * (len(pats) + 1)]
        for _, r in df.iterrows():
            cells = []
            for p in pats:
                v = r[p]
                cells.append("—" if np.isnan(v) else (f"**{v:.2f}**" if v >= 0.3 else f"{v:.2f}"))
            md.append(f"| L{int(r['layer'])}·H{int(r['head'])} | " + " | ".join(cells) + f" | {r['lookback']:.1f} |")
        ch = " | ".join(f"{df[p + '_chance'].iloc[0]:.2f}" if not np.isnan(df[p + '_chance'].iloc[0]) else "—"
                        for p in pats)
        md += [f"| *chance* | {ch} | |",
               f"| *# query positions* | " + " | ".join(str(int(df[p + '_n'].iloc[0])) for p in pats) + " | |", ""]
        for p in pats:
            if df[p].notna().any():
                i = df[p].idxmax()
                best.append({"text": kind, "pattern": p, "best head": f"L{int(df['layer'][i])}·H{int(df['head'][i])}",
                             "weight": df[p][i], "chance": df[p + "_chance"][i],
                             "x chance": df[p][i] / df[p + "_chance"][i]})
        print(f"[attn] {kind}: done")

    b = pd.DataFrame(best)
    md += ["### Best head per pattern and text", "", "| text | pattern | best head | weight | chance | × chance |",
           "|---|---|---|---:|---:|---:|"]
    for _, r in b.iterrows():
        md.append(f"| {r['text']} | {r['pattern']} | {r['best head']} | {r['weight']:.3f} | {r['chance']:.3f} | "
                  f"{r['x chance']:.1f} |")
    (out / "head_stats.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out}/attn_*.png, head_stats_*.csv, head_stats.md")


if __name__ == "__main__":
    main()
