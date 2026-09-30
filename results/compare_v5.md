| benchmark | leo-1.7b-v3 | leo-4b-v5 | leo-4b-v5.1 | leo-4b-soup3 | jev-live |
|---|---|---|---|---|---|
| naturalcodz drop-in, 37 scenarios | 32/37 | 36/37 | 33/37 | 35/37 | 36/37 |
| Browser tasks passed (7 tasks x 3) | 18/21 | 15/21 | 18/21 | 18/21 | 18/21 |
| Short suite, 211 conditions (as sent) | 0.872 | 0.957 | 0.943 | 0.953 | 0.976 |
| Short suite (short-condition rewrite on) | 0.948 | 0.972 | 0.957 | 0.972 | - |
| Short suite ECE (as sent) | 0.093 | 0.036 | 0.035 | 0.029 | 0.075 |
| JevBench all | 0.697 | 0.701 | 0.697 | 0.701 | 0.861 |
| JevBench easy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| JevBench standard | 0.958 | 0.931 | 0.931 | 0.944 | 0.986 |
| JevBench hard | 0.396 | 0.423 | 0.414 | 0.414 | 0.721 |
| JevBench ECE | 0.101 | 0.103 | 0.141 | 0.110 | 0.057 |
| Held-out emotion (acc) | 0.564 | 0.541 | 0.527 | 0.547 | 0.587 |
| Held-out tweet_topic (acc) | 0.845 | 0.839 | 0.837 | 0.840 | 0.790 |
| Held-out fin_topic (acc) | 0.456 | 0.556 | 0.564 | 0.567 | 0.669 |
| Held-out daily_dialog (acc) | 0.647 | 0.629 | 0.668 | 0.646 | 0.710 |
| Held-out mean accuracy | 0.628 | 0.641 | 0.649 | 0.650 | 0.689 |
| Held-out mean macro-F1 | 0.493 | 0.505 | 0.504 | 0.511 | 0.552 |
| Held-out mean ECE | 0.099 | 0.069 | 0.065 | 0.069 | 0.167 |
| Multilingual belebele | 0.692 | 0.675 | 0.640 | 0.681 | 0.917 |
| Multilingual mmmlu | 0.479 | 0.476 | 0.459 | 0.481 | 0.861 |
| Multilingual include | 0.491 | 0.563 | 0.554 | 0.576 | 0.775 |
| Multilingual all | 0.560 | 0.574 | 0.553 | 0.582 | 0.857 |
| False DONE, held-out browser screens | 0.067 | 0.000 | 0.000 | 0.000 | - |
| Browser step accuracy, held-out screens | 0.665 | 1.000 | 0.990 | 0.995 | - |
| Browser step accuracy, v3 simulator | 0.995 | 1.000 | 0.990 | 0.990 | - |
| Flights: unsubmitted form P(DONE) | 0.38 CLICK | 0.00 CLICK | 0.01 CLICK | 0.00 CLICK | - |
| Flights: seat-class list P(DONE) | 0.99 DONE | 0.00 CLICK | 0.00 CLICK | 0.00 CLICK | - |
| Flights: blank page P(DONE) | 1.00 DONE | 0.00 WAIT | 0.04 WAIT | 0.00 WAIT | - |
