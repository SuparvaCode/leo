Device: NVIDIA GeForce RTX 3050

| probe | leo-1.7b-v3 bf16 | leo-1.7b-v3 bf16 2 views | Jev (live) |
|---|---|---|---|
| option-order flip rate, emotion (300 rows x 5 shuffles) | 0.100 | 0.055 | 0.018 |
| mean total-variation shift, emotion | 0.063 | 0.032 | 0.036 |
| option-order flip rate, fin_topic (300 rows x 5 shuffles) | 0.217 | 0.116 | 0.069 |
| mean total-variation shift, fin_topic | 0.141 | 0.058 | 0.063 |
| 4 questions together vs one at a time: max prob. difference (100 states) | 0.0236 | 0.0195 | 0.1300 |
| ... top answers that changed | 2 | 3 | 1 |
| model p50 / p95 ms, short state, 1 question(s) | 37 / 46 | 36 / 41 | 56 / 84 |
| end-to-end p50 / p95 ms incl. network, short state, 1 question(s) | - | - | 324 / 365 |
| model p50 / p95 ms, short state, 10 question(s) | 91 / 92 | 142 / 144 | 64 / 83 |
| end-to-end p50 / p95 ms incl. network, short state, 10 question(s) | - | - | 336 / 357 |
| model p50 / p95 ms, short state, 50 question(s) | 392 / 393 | 1082 / 1086 | 78 / 133 |
| end-to-end p50 / p95 ms incl. network, short state, 50 question(s) | - | - | 358 / 439 |
| model p50 / p95 ms, long state, 1 question(s) | 86 / 87 | 91 / 92 | 60 / 105 |
| end-to-end p50 / p95 ms incl. network, long state, 1 question(s) | - | - | 333 / 373 |
| model p50 / p95 ms, long state, 10 question(s) | 142 / 142 | 186 / 187 | 62 / 113 |
| end-to-end p50 / p95 ms incl. network, long state, 10 question(s) | - | - | 336 / 415 |
| model p50 / p95 ms, long state, 50 question(s) | 460 / 461 | 1082 / 1085 | 72 / 110 |
| end-to-end p50 / p95 ms incl. network, long state, 50 question(s) | - | - | 351 / 382 |
