Device: NVIDIA L4

| probe | leo-4b-v5 bf16 | leo-4b-v5 bf16 2 views |
|---|---|---|
| option-order flip rate, emotion (300 rows x 5 shuffles) | 0.082 | 0.039 |
| mean total-variation shift, emotion | 0.049 | 0.021 |
| option-order flip rate, fin_topic (300 rows x 5 shuffles) | 0.115 | 0.061 |
| mean total-variation shift, fin_topic | 0.088 | 0.042 |
| 4 questions together vs one at a time: max prob. difference (100 states) | 0.0264 | 0.0318 |
| ... top answers that changed | 3 | 5 |
| model p50 / p95 ms, short state, 1 question(s) | 63 / 64 | 64 / 64 |
| model p50 / p95 ms, short state, 10 question(s) | 77 / 78 | 131 / 133 |
| model p50 / p95 ms, short state, 50 question(s) | 340 / 344 | 1021 / 1024 |
| model p50 / p95 ms, long state, 1 question(s) | 74 / 75 | 75 / 77 |
| model p50 / p95 ms, long state, 10 question(s) | 132 / 134 | 160 / 164 |
| model p50 / p95 ms, long state, 50 question(s) | 391 / 393 | 1023 / 1026 |
