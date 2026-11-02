metric: final val_bpb on the full validation split (windows of seq_len). Seed noise floor (median std) = 0.0038 bpb; a gain is 'resolved' if > 2x that.

| model | seq_len | n seeds | val bpb (mean ± std) | gain vs previous seq_len | resolved? |
|---|---:|---:|---:|---:|:---:|
| lstm | 64 | 2 | 2.1806 ± 0.0051 | — | — |
| lstm | 128 | 2 | 2.1487 ± 0.0029 | +0.0319 | yes |
| lstm | 256 | 2 | 2.1370 ± 0.0047 | +0.0116 | yes |
| lstm | 512 | 2 | 2.1299 ± 0.0064 | +0.0071 | not resolved |
| transformer (modern) | 64 | 2 | 2.1562 ± 0.0006 | — | — |
| transformer (modern) | 128 | 2 | 2.1135 ± 0.0090 | +0.0427 | yes |
| transformer (modern) | 256 | 2 | 2.1159 ± 0.0010 | -0.0023 | not resolved |
| transformer (modern) | 512 | 2 | 2.1030 ± 0.0007 | +0.0129 | yes |
