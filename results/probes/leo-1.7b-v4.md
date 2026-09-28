Device: NVIDIA GeForce RTX 3050

| probe | leo-1.7b-v4 bf16 | leo-1.7b-v4 bf16 2 views | Jev (live) |
|---|---|---|---|
| option-order flip rate, emotion (300 rows x 5 shuffles) | 0.111 | 0.044 | 0.018 |
| mean total-variation shift, emotion | 0.065 | 0.027 | 0.036 |
| option-order flip rate, fin_topic (300 rows x 5 shuffles) | 0.188 | 0.105 | 0.069 |
| mean total-variation shift, fin_topic | 0.140 | 0.057 | 0.063 |
| 4 questions together vs one at a time: max prob. difference (100 states) | 0.0427 | 0.0219 | 0.1300 |
| ... top answers that changed | 4 | 4 | 1 |
| model p50 / p95 ms, short state, 1 question(s) | 40 / 88 | 36 / 42 | 52 / 89 |
| end-to-end p50 / p95 ms incl. network, short state, 1 question(s) | - | - | 331 / 366 |
| model p50 / p95 ms, short state, 10 question(s) | 94 / 96 | 152 / 157 | 50 / 88 |
| end-to-end p50 / p95 ms incl. network, short state, 10 question(s) | - | - | 342 / 581 |
| model p50 / p95 ms, short state, 50 question(s) | 423 / 446 | 1177 / 1212 | 72 / 120 |
| end-to-end p50 / p95 ms incl. network, short state, 50 question(s) | - | - | 357 / 441 |
| model p50 / p95 ms, long state, 1 question(s) | 94 / 97 | 97 / 98 | 54 / 90 |
| end-to-end p50 / p95 ms incl. network, long state, 1 question(s) | - | - | 331 / 367 |
| model p50 / p95 ms, long state, 10 question(s) | 152 / 160 | 199 / 207 | 64 / 100 |
| end-to-end p50 / p95 ms incl. network, long state, 10 question(s) | - | - | 343 / 474 |
| model p50 / p95 ms, long state, 50 question(s) | 500 / 523 | 1178 / 1215 | 79 / 103 |
| end-to-end p50 / p95 ms incl. network, long state, 50 question(s) | - | - | 358 / 384 |
