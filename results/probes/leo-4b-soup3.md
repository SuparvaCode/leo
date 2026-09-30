Device: NVIDIA L4

| probe | leo-4b-soup3 bf16 | leo-4b-soup3 bf16 2 views |
|---|---|---|
| option-order flip rate, emotion (300 rows x 5 shuffles) | 0.074 | 0.024 |
| mean total-variation shift, emotion | 0.047 | 0.021 |
| option-order flip rate, fin_topic (300 rows x 5 shuffles) | 0.101 | 0.058 |
| mean total-variation shift, fin_topic | 0.084 | 0.041 |
| 4 questions together vs one at a time: max prob. difference (100 states) | 0.0172 | 0.0191 |
| ... top answers that changed | 3 | 2 |
| model p50 / p95 ms, short state, 1 question(s) | 67 / 69 | 71 / 72 |
| model p50 / p95 ms, short state, 10 question(s) | 75 / 79 | 129 / 131 |
| model p50 / p95 ms, short state, 50 question(s) | 334 / 337 | 1001 / 1004 |
| model p50 / p95 ms, long state, 1 question(s) | 73 / 83 | 74 / 75 |
| model p50 / p95 ms, long state, 10 question(s) | 131 / 132 | 159 / 161 |
| model p50 / p95 ms, long state, 50 question(s) | 382 / 384 | 1002 / 1005 |
