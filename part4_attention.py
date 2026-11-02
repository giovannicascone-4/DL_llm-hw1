# Part 4: attention maps (nanolm.viz attention) on three kinds of text, and for every head the average
# attention weight on: the previous character, the first letter of the current word, the last ':' or newline.
# Run from DeepDataMiningLearning/DeepDataMiningLearning/llm
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import nanolm.viz as viz  # noqa: E402
from nanolm.generate import load_run  # noqa: E402

RUN = "outputs/nanolm/sweeps/hw1_char_models/gpt_modern_s1337"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
TEXTS = {
    "shakespeare": "ROMEO:\nBut soft! what light through yonder window breaks?\nJULIET:\nAy me!",
    "modern": "The engineer trained a small language model on her laptop, and it wrote poems.",
    "repeated": "xqzvbkmwpt jrdlhs " * 3,
}


def targets(text):
    """For each position t, the position each kind of head should look at."""
    prev, word, line = {}, {}, {}
    word_start, last_mark = None, None
    for t, ch in enumerate(text):
        word_start = (t if word_start is None else word_start) if (ch.isalpha() or ch == "'") else None
        if t >= 1:
            prev[t] = t - 1
        if word_start is not None and t - word_start >= 2:   # skip t-1 and t itself
            word[t] = word_start
        if last_mark is not None and t - last_mark >= 2:
            line[t] = last_mark
        if ch in ":\n":
            last_mark = t
    return {"previous char": prev, "word start": word, "last ':' or newline": line}


viz.style()
viz.FIG_DIR = OUT
cfg, model, tok, _, _ = load_run(RUN, device="cpu")
lines = []
for kind, text in TEXTS.items():
    viz.fig_attention(run_dir=RUN, text=text)                       # the maps, as in `nanolm.viz attention`
    (OUT / "p1_attention_maps.png").rename(OUT / f"part4_attention_{kind}.png")
    maps = viz.attention_maps(model, torch.tensor([tok.encode(text)]))   # one [heads, T, T] array per layer
    pats = targets(text)
    lines += [f"## {kind}: {text!r}", "", "| head | " + " | ".join(pats) + " |", "|---|" + "---:|" * len(pats)]
    for layer, A in enumerate(maps):
        for head in range(A.shape[0]):
            cells = [f"{np.mean([A[head, t, s] for t, s in p.items()]):.2f}" if p else "-" for p in pats.values()]
            lines.append(f"| L{layer}·H{head} | " + " | ".join(cells) + " |")
    chance = [f"{np.mean([1 / (t + 1) for t in p]):.2f}" if p else "-" for p in pats.values()]
    lines += ["| uniform attention (chance) | " + " | ".join(chance) + " |", ""]

(OUT / "part4.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
