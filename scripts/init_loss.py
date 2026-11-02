"""HW1 part 1b (supplement): loss at TRUE initialization (step 0, no optimizer step) vs ln V.

    cd DeepDataMiningLearning/DeepDataMiningLearning/llm
    python ../../../hw1/scripts/init_loss.py --out ../../../hw1/results/part1/init_loss_step0.md

`train.max_steps=1` scores the model AFTER one AdamW step. Adam's first step moves every weight by
about lr in the direction of -sign(grad), so even at warm-up LR it already lowers the loss a little,
mostly by down-weighting vocabulary entries that never occur. This script builds the same model
(same config as the hw1_untrained sweep), evaluates it on the full validation split with NO training,
and also reports the size of the vocabulary actually used in the validation text and the unigram entropy.
Runs on CPU, a few seconds per tokenizer.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from nanolm.config import load_config  # noqa: E402
from nanolm.data import TokenData, byte_lut_tensors, prepare_dataset  # noqa: E402
from nanolm.device import get_device, seed_everything  # noqa: E402
from nanolm.evaluate import evaluate  # noqa: E402
from nanolm.models import build_model  # noqa: E402

TOKENIZERS = [("char", "char"), ("byte", "byte"), ("bpe_512", "sp_train:bpe:512"), ("word_2048", "word:2048")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1337, 42])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = get_device("cpu", "fp32")
    lines = ["| tokenizer | seed | V | step-0 loss (nats) | ln V | loss - ln V | vocab used in val (U) | ln U "
             "| unigram entropy of val | step-0 bpb | log2 V x tok/byte |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, spec in TOKENIZERS:
        for seed in a.seeds:
            cfg = load_config("nanolm/configs/laptop/L3_tokenizer.yaml", [f"data.tokenizer={spec}", "device=cpu",
                                                                          f"seed={seed}"])
            seed_everything(seed)
            meta = prepare_dataset(cfg.data)
            data = TokenData(meta, seed=seed)
            V = meta["vocab_size"]
            cfg.model.vocab_size = V
            cfg.model.max_seq_len = max(cfg.model.max_seq_len, cfg.train.seq_len)
            model = build_model(cfg.model).to(d.device)
            r = evaluate(model, data.val_windows(cfg.train.seq_len, 0), cfg.train.seq_len, 32,
                         byte_lut_tensors(meta, d.device), d)
            v = np.asarray(data.val).astype(np.int64)
            cnt = np.bincount(v, minlength=V)
            U = int((cnt > 0).sum())
            p = cnt[cnt > 0] / cnt.sum()
            H = float(-(p * np.log(p)).sum())
            tpb = r["tokens"] / r["bytes"]
            lines.append(f"| {name} | {seed} | {V} | {r['loss']:.4f} | {math.log(V):.4f} | {r['loss'] - math.log(V):+.4f} "
                         f"| {U} | {math.log(U):.4f} | {H:.4f} | {r['bpb']:.3f} | {math.log2(V) * tpb:.3f} |")
            print(lines[-1], flush=True)
    txt = "\n".join(lines)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    main()
