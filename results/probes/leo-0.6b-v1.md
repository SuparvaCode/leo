Device: NVIDIA GeForce RTX 3050

| probe | leo-0.6b-v1 bf16 | leo-0.6b-v1 fp32 | Jev (live) |
|---|---|---|---|
| option-order flip rate, emotion (300 rows x 5 shuffles) | 0.105 | 0.100 | 0.018 |
| mean total-variation shift, emotion | 0.093 | 0.093 | 0.036 |
| option-order flip rate, fin_topic (300 rows x 5 shuffles) | 0.153 | 0.159 | 0.069 |
| mean total-variation shift, fin_topic | 0.095 | 0.095 | 0.063 |
| 4 questions together vs one at a time: max prob. difference (100 states) | 0.0583 | 0.0001 | 0.1300 |
| ... top answers that changed | 2 | 0 | 1 |
| model p50 / p95 ms, short state, 1 question(s) | 38 / 49 | 32 / 40 | 54 / 93 |
| end-to-end p50 / p95 ms incl. network, short state, 1 question(s) | - | - | 331 / 627 |
| model p50 / p95 ms, short state, 10 question(s) | 46 / 46 | 90 / 91 | 71 / 117 |
| end-to-end p50 / p95 ms incl. network, short state, 10 question(s) | - | - | 359 / 429 |
| model p50 / p95 ms, short state, 50 question(s) | 216 / 217 | 560 / 563 | 82 / 100 |
| end-to-end p50 / p95 ms incl. network, short state, 50 question(s) | - | - | 366 / 426 |
| model p50 / p95 ms, long state, 1 question(s) | 42 / 44 | 85 / 86 | 66 / 97 |
| end-to-end p50 / p95 ms incl. network, long state, 1 question(s) | - | - | 348 / 604 |
| model p50 / p95 ms, long state, 10 question(s) | 71 / 72 | 170 / 171 | 66 / 105 |
| end-to-end p50 / p95 ms incl. network, long state, 10 question(s) | - | - | 354 / 648 |
| model p50 / p95 ms, long state, 50 question(s) | 256 / 257 | 659 / 661 | 90 / 174 |
| end-to-end p50 / p95 ms incl. network, long state, 50 question(s) | - | - | 378 / 608 |
