## Metrics by hand (modern Transformer, seed 1337)

| metric | by hand | logged |
|---|---:|---:|
| loss (nats/token) | 1.467075 | 1.467075 |
| perplexity = e^loss | 4.336532 | 4.336532 |
| bits/token = loss / ln 2 | 2.116542 | - |
| bits per byte = bits/token x tokens / bytes | 2.116542 | 2.116542 |

## Untrained models vs ln V

| tokenizer | seed | V | ln V | untrained loss | loss / ln V |
|---|---:|---:|---:|---:|---:|
| bpe_512 | 1337 | 512 | 6.2383 | 6.2347 | 0.999 |
| bpe_512 | 42 | 512 | 6.2383 | 6.2190 | 0.997 |
| byte | 1337 | 256 | 5.5452 | 5.4436 | 0.982 |
| byte | 42 | 256 | 5.5452 | 5.4404 | 0.981 |
| char | 1337 | 65 | 4.1744 | 4.1601 | 0.997 |
| char | 42 | 65 | 4.1744 | 4.1758 | 1.000 |
| word_2048 | 1337 | 2048 | 7.6246 | 7.6017 | 0.997 |
| word_2048 | 42 | 2048 | 7.6246 | 7.5338 | 0.988 |
