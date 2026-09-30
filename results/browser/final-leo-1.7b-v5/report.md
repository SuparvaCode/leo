| task | leo:leo-1.7b-v5 passed | leo:leo-1.7b-v5 median time / decisions |
|---|---|---|
| travel-casa-flora | 3/3 | 5.1 s / 6 |
| research-finite-choices | 3/3 | 0.7 s / 2 |
| travel-glasshouse | 3/3 | 6.7 s / 7 |
| travel-serra-lodge | 3/3 | 4.3 s / 5 |
| research-confidence | 3/3 | 3.0 s / 7 |
| wikipedia-godel (live site) | 0/3 | 2.0 s / 2 |
| google-flights (live site) | 0/3 | 91.2 s / 40 |
| **all** | **15/21** |  |

Failures:

- leo:leo-1.7b-v5 / wikipedia-godel #0: ValueError: Text helper returned no valid field value; nothing typed.; 0 actions, last: -
- leo:leo-1.7b-v5 / wikipedia-godel #1: ValueError: Text helper returned no valid field value; nothing typed.; 0 actions, last: -
- leo:leo-1.7b-v5 / wikipedia-godel #2: ValueError: Text helper returned no valid field value; nothing typed.; 0 actions, last: -
- leo:leo-1.7b-v5 / google-flights #0: stopped at the 120 s wall-clock limit; 31 actions, last: Done. 
- leo:leo-1.7b-v5 / google-flights #1: status blocked, checks search_page,one_way,origin,destination,date,results; 24 actions, last: Return
- leo:leo-1.7b-v5 / google-flights #2: status blocked, checks search_page,one_way,origin,destination,date,results; 24 actions, last: Return
