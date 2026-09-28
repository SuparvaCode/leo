| task | jev passed | jev median time / decisions | leo:leo-1.7b-v3 passed | leo:leo-1.7b-v3 median time / decisions |
|---|---|---|---|---|
| travel-casa-flora | 3/3 | 3.6 s / 6 | 3/3 | 5.0 s / 6 |
| research-finite-choices | 3/3 | 0.8 s / 2 | 3/3 | 5.7 s / 12 |
| travel-glasshouse | 3/3 | 3.5 s / 6 | 3/3 | 5.0 s / 6 |
| travel-serra-lodge | 3/3 | 3.0 s / 5 | 3/3 | 4.3 s / 5 |
| research-confidence | 3/3 | 0.8 s / 2 | 3/3 | 0.7 s / 2 |
| wikipedia-godel (live site) | 3/3 | 4.0 s / 4 | 3/3 | 11.6 s / 5 |
| google-flights (live site) | 0/3 | 1.6 s / 3 | 0/3 | 18.7 s / 14 |
| **all** | **18/21** |  | **18/21** |  |

Failures:

- jev / google-flights #0: status blocked, checks search_page,one_way,origin,destination,date,year,results; 1 actions, last: Change ticket type. Round trip
- leo:leo-1.7b-v3 / google-flights #0: status done, checks search_page,one_way,origin,destination,date,results; 8 actions, last: Change seating class. Economy
- leo:leo-1.7b-v3 / google-flights #1: status done, checks search_page,one_way,origin,destination,date,results; 8 actions, last: Change seating class. Economy
- jev / google-flights #1: status blocked, checks search_page,one_way,origin,destination,date,year,results; 2 actions, last: Change ticket type. Round trip
- jev / google-flights #2: status blocked, checks search_page,one_way,origin,destination,date,year,results; 1 actions, last: Change ticket type. Round trip
- leo:leo-1.7b-v3 / google-flights #2: status done, checks search_page,one_way,origin,destination,date,results; 8 actions, last: Change seating class. Economy
