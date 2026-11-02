"""Part 1: loss of the untrained models (train.max_steps=1) compared with ln V."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def vocab_size(run_dir: Path) -> int:
    ck = run_dir / "model.pt"
    if ck.exists():
        import torch
        meta = torch.load(ck, map_location="cpu", weights_only=False)["data_meta"]
        return int(meta["vocab_size"])
    log = (run_dir / "stdout.log").read_text()
    return int(re.search(r"vocab=(\d+)", log).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for mf in sorted(Path(a.sweep_dir).glob("*/metrics.json")):
        m = json.loads(mf.read_text())
        V = vocab_size(mf.parent)
        lnV = math.log(V)
        tpb = m["val_tokens"] / m["val_bytes"]
        rows.append(dict(name=m["name"], seed=m["seed"], tok=m["tokenizer"], V=V, loss=m["val_loss"], lnV=lnV,
                         gap=m["val_loss"] - lnV, ratio=m["val_loss"] / lnV, bpb=m["val_bpb"],
                         uni_bpb=math.log2(V) * tpb, bpt=1 / tpb, steps=m["steps"]))
    if not rows:
        raise SystemExit(f"no runs under {a.sweep_dir}")
    out = ["| run | seed | tokenizer | V | steps | untrained loss (nats) | ln V | loss - ln V | loss / ln V "
           "| bytes/token | val bpb | uniform bpb = log2 V x tok/byte |",
           "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["V"], r["seed"])):
        out.append(f"| {r['name']} | {r['seed']} | {r['tok']} | {r['V']} | {r['steps']} | {r['loss']:.4f} | "
                   f"{r['lnV']:.4f} | {r['gap']:+.4f} | {r['ratio']:.3f} | {r['bpt']:.3f} | {r['bpb']:.3f} | "
                   f"{r['uni_bpb']:.3f} |")
    txt = "\n".join(out)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
