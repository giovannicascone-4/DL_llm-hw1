# HW1 — Language models from scratch (Part 1): run kit

Everything here drives the course's `nanolm` code. No library code is edited: every experiment is a sweep YAML,
and every number comes from a nanolm run directory, as the general requirements ask.

## Layout

```
LLM Homework1/
  hw1/
    setup.sh                 one-time: venv, packages, clone repo, tests, smoke run, machine benchmark
    run_all.sh               all experiments + analysis (resumable)
    sweeps/
      hw1_untrained.yaml     part 1b: max_steps=1, tokenizers char / byte / bpe_512 / word_2048, 2 seeds
      hw1_char_models.yaml   part 2: rnn, gru, lstm, lstm_512, gpt_textbook, gpt_modern, 2 seeds
      hw1_context.yaml       part 3: lstm + gpt_modern at seq_len 64 / 128 / 512 (256 reused from part 2)
    scripts/
      metrics_by_hand.py     part 1a
      untrained_vs_lnV.py    part 1b
      plot_train_val.py      part 2 (train vs val loss: the "larger LSTM" question)
      plot_context.py        part 3
      attention_heads.py     part 4 (maps + per-head numbers)
      gru_scratch.py         extra credit: GRU gate by gate, checked against nn.GRU
      run_info.py            git commit, device, wall-clock, params, FLOPs for the Setup section
    results/                 everything for the report lands here
  DeepDataMiningLearning/    the course repo (cloned by setup.sh)
```

## Run it (in Terminal on your Mac)

```bash
cd ~/Desktop/"Deep Learning"/"LLM Homework1"
bash hw1/setup.sh                 # ~10 min, mostly the PyTorch download
```

Check two things in the output: the smoke run says `device: mps`, and the `projected` times in
`hw1/results/bench.txt`. Total compute is 24 training runs at 2000 steps (16.4M chars each) plus 8 tiny ones.
- If the LSTM-512 and Transformer projections are about 3 min or less, use the full budget:
  `bash hw1/run_all.sh all`
- If they are slower, use the laptop version, which halves the token budget for every model so the
  comparison stays fair: `STEPS=1000 bash hw1/run_all.sh all`
- Pick one `STEPS` value and keep it. Runs that have already finished are skipped, so switching `STEPS`
  halfway would mix budgets. If you change it, delete
  `DeepDataMiningLearning/DeepDataMiningLearning/llm/outputs/nanolm/sweeps/hw1_*` first.

**Using the whole Mac.** All 32 training runs are launched at once in two pools that run side by side:
- **GPU pool:** `PAR_GPU=6` Transformer runs at a time on MPS.
- **RNN pool:** `PAR_CPU=4` RNN/GRU/LSTM runs at a time. With `RNN_DEVICE=auto` (the default), a 5-step benchmark
  of the LSTM-512 decides whether these run on the CPU cores, which then share all cores, or on the GPU.

On a 48 GB M-Max you can push it: `PAR_GPU=8 PAR_CPU=6 caffeinate -i bash hw1/run_all.sh all`. Results are the same
however many runs go at once, because each run is its own seeded process. Only the per-run `train_time_sec` grows
when runs share the machine, so quote the total time in the report. The `device` field of each run's `metrics.json`
records where it ran; mention it in Setup.

The script is resumable: after Ctrl-C or a crash, run the same command again. To keep the Mac awake:
`caffeinate -i bash hw1/run_all.sh all`. You can also run one part at a time:
`bash hw1/run_all.sh part2` (then `part1`, `part3`, `part4`, `ec`, `info`). Part 2 has to come first,
because part 1a and part 4 use its trained `gpt_modern` model and part 3 reuses its seq_len=256 runs.

## What lands in `hw1/results/` → which report item

| HW item | files |
|---|---|
| 1a metrics by hand | `part1/metrics_by_hand_gpt_modern.md` (also `_lstm.md`) |
| 1b untrained loss vs ln V | `part1/untrained_vs_lnV.md` |
| 2 learning curves + table | `part2/curves.png`, `part2/bars.png`, `part2/summary.md` / `summary.csv` (params, mean ± std, paired p vs gpt_textbook), `part2/train_vs_val.png`, `part2/best_vs_final.md` |
| 3 context length | `part3/context_bpb.png`, `part3/context_curves.png`, `part3/context_table.md` (gain per doubling, "resolved" if > 2× seed noise) |
| 4 attention | `part4/attn_{shakespeare,modern_prose,repeat}.png`, `part4/head_stats.md` (per-head weights + chance levels) |
| extra credit | `extra_credit/gru_check.txt` |
| Setup section | `run_info.md` (commit, device, wall-clock, params, 6ND FLOPs), `configs/` (submit these) |

## Notes for the analysis (where to look)

- **1b:** an untrained model should give loss ≈ ln V and bpb ≈ log2(V) × tokens/byte. In the dry run the
  values were 0.98 to 1.00 × ln V, which is what you expect for a well-initialized model.
- **Vanilla RNN:** no gates, so the gradient through time is a product of Jacobians and vanishes or explodes
  (grad_clip = 1 only treats the exploding side). Compare its curve with GRU/LSTM, and note it has 0.28M parameters.
- **Larger LSTM:** in `train_vs_val.png`, watch whether val loss turns upward while train loss keeps falling,
  and check the step of the best bpb in `best_vs_final.md` (the tutorial saw a peak at about 10M tokens).
  With `STEPS=1000` the overfitting may not show up yet. If so, say so: that is a limitation, not a contradiction.
- **Context:** evaluation uses windows of seq_len, so a token gets on average about seq_len/2 characters of context.
  Every run uses the same tokens/step (batch × seq_len = 8192). An LSTM is also reset to a zero state at
  each window (truncated BPTT), but its effective memory is much shorter than the window.
- **Attention:** a head is "previous-token" if its `prev` weight is far above chance (×5 or more), and
  "word-start" if `word_start` is. The `repeat` text tests copying/induction (`induction` column).
  Quote the numbers from `head_stats.md`.
