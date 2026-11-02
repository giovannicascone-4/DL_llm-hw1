Run: `outputs/nanolm/sweeps/hw1_char_models/lstm_s1337`  (arch=lstm, tokenizer=char, N_tok=111,360, N_bytes=111,360)

| quantity | formula | by hand | logged | abs. diff |
|---|---|---:|---:|---:|
| loss (nats/token) | val_loss (given) | 1.478962 | 1.478962 | 0.00e+00 |
| total nats | loss x N_tok | 164697.243277 |  |  |
| total bits | total nats / ln 2 | 237607.896124 |  |  |
| perplexity | exp(loss) | 4.388390 | 4.388390 | 0.00e+00 |
| bits / token | loss / ln 2 | 2.133692 |  |  |
| bytes / token | N_bytes / N_tok | 1.000000 | 1.000000 | 0.00e+00 |
| bits per byte | (loss / ln 2) x (N_tok / N_bytes) | 2.133692 | 2.133692 | 4.44e-16 |

Check: nats -> bits factor 1/ln2 = 1.4427; for characters bytes/token = 1.0000, so bpb = bits/token (ASCII text: 1 char = 1 byte).
