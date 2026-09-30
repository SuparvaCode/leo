Device: NVIDIA L4

| probe | leo-1.7b-v5 bf16 | leo-1.7b-v5 bf16 2 views |
|---|---|---|
| option-order flip rate, emotion (300 rows x 5 shuffles) | 0.077 | 0.041 |
| mean total-variation shift, emotion | 0.061 | 0.027 |
| option-order flip rate, fin_topic (300 rows x 5 shuffles) | 0.177 | 0.090 |
| mean total-variation shift, fin_topic | 0.101 | 0.052 |
| 4 questions together vs one at a time: max prob. difference (100 states) | 0.0164 | 0.0153 |
| ... top answers that changed | 3 | 3 |
| model p50 / p95 ms, short state, 1 question(s) | 53 / 55 | 52 / 53 |
| model p50 / p95 ms, short state, 10 question(s) | 55 / 56 | 55 / 58 |
| model p50 / p95 ms, short state, 50 question(s) | 140 / 142 | 430 / 433 |
| model p50 / p95 ms, long state, 1 question(s) | 53 / 54 | 53 / 54 |
| model p50 / p95 ms, long state, 10 question(s) | 56 / 57 | 67 / 69 |
| model p50 / p95 ms, long state, 50 question(s) | 165 / 168 | 433 / 435 |
