"""HW1 bookkeeping for the report's Setup section: git commit, device, wall-clock, params, tokens, FLOPs.

    python ../../../hw1/scripts/run_info.py outputs/nanolm/sweeps/hw1_* --out results/run_info.md

FLOPs ~ 6 x N x D  (N = parameters, D = training tokens seen), as in §1.6.7.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, commits, devices, platforms = [], set(), set(), set()
    for root in a.roots:
        for mf in sorted(Path(root).glob("*/metrics.json")):
            m = json.loads(mf.read_text())
            meta_f = mf.parent / "meta.json"
            meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
            git = meta.get("git") or {}
            commits.add(f"{git.get('commit')}{' (dirty)' if git.get('dirty') else ''}")
            devices.add(m.get("device", "?"))
            platforms.add(f"{meta.get('platform', '?')} | torch {meta.get('torch', '?')}")
            rows.append((Path(root).name, m["name"], m["seed"], m["params"], m["tokens_seen"],
                         6 * m["params"] * m["tokens_seen"], m["train_time_sec"], m.get("tok_per_sec", 0),
                         m.get("peak_mem_mb", 0), m["val_bpb"]))
    total = sum(r[6] for r in rows)
    out = [f"- git commit: {', '.join(sorted(commits))}",
           f"- device: {', '.join(sorted(devices))}",
           f"- platform: {', '.join(sorted(platforms))}",
           f"- runs: {len(rows)}, total training wall-clock: {total / 60:.1f} min ({total / 3600:.2f} h)", "",
           "| sweep | run | seed | params | train tokens | FLOPs (6ND) | train time (s) | tok/s | peak mem (MB) | val bpb |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:,} | {r[4]:,} | {r[5]:.2e} | {r[6]:.0f} | {r[7]:,} | "
                   f"{r[8]:.0f} | {r[9]:.4f} |")
    txt = "\n".join(out)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
