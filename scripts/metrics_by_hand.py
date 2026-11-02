"""HW1 part 1a: recompute loss, perplexity, bits/token and bits-per-byte from metrics.json by hand.

    python ../../../hw1/scripts/metrics_by_hand.py <run_dir> [--out results/part1_metrics_by_hand.md]

Inputs used (all from the run's metrics.json):
    val_loss    mean -ln p(target) over the scored tokens        [nats / token]
    val_tokens  number of scored target tokens N_tok
    val_bytes   number of UTF-8 bytes those targets cover N_bytes
Checked against the values the trainer logged: val_ppl, val_bpb (and val_bytes_per_token).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="write a markdown table here")
    a = ap.parse_args()
    m = json.loads((Path(a.run_dir) / "metrics.json").read_text())

    L, N, NB = m["val_loss"], m["val_tokens"], m["val_bytes"]
    total_nats = L * N                      # sum of -ln p over the validation set
    total_bits = total_nats / math.log(2)
    ppl = math.exp(L)
    bits_per_token = L / math.log(2)
    bytes_per_token = NB / N
    bpb = bits_per_token / bytes_per_token  # = total_bits / NB

    rows = [
        ("loss (nats/token)", "val_loss (given)", L, m["val_loss"]),
        ("total nats", "loss x N_tok", total_nats, None),
        ("total bits", "total nats / ln 2", total_bits, None),
        ("perplexity", "exp(loss)", ppl, m.get("val_ppl")),
        ("bits / token", "loss / ln 2", bits_per_token, None),
        ("bytes / token", "N_bytes / N_tok", bytes_per_token, m.get("val_bytes_per_token")),
        ("bits per byte", "(loss / ln 2) x (N_tok / N_bytes)", bpb, m.get("val_bpb")),
    ]
    lines = [f"Run: `{a.run_dir}`  (arch={m.get('arch')}, tokenizer={m.get('tokenizer')}, "
             f"N_tok={N:,}, N_bytes={NB:,})", "",
             "| quantity | formula | by hand | logged | abs. diff |", "|---|---|---:|---:|---:|"]
    for name, formula, mine, logged in rows:
        lg = "" if logged is None else f"{logged:.6f}"
        df = "" if logged is None else f"{abs(mine - logged):.2e}"
        lines.append(f"| {name} | {formula} | {mine:.6f} | {lg} | {df} |")
    lines += ["", f"Check: nats -> bits factor 1/ln2 = {1 / math.log(2):.4f}; "
              f"for characters bytes/token = {bytes_per_token:.4f}, so bpb = bits/token "
              f"{'(ASCII text: 1 char = 1 byte)' if abs(bytes_per_token - 1) < 1e-9 else ''}."]
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
