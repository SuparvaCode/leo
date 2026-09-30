| task | jev passed | jev median time / decisions | leo:leo-4b-v5 passed | leo:leo-4b-v5 median time / decisions |
|---|---|---|---|---|
| travel-casa-flora | 3/3 | 3.6 s / 6 | 3/3 | 6.9 s / 6 |
| research-finite-choices | 3/3 | 0.8 s / 2 | 3/3 | 1.2 s / 2 |
| travel-glasshouse | 3/3 | 3.6 s / 6 | 0/3 | 7.1 s / 4 |
| travel-serra-lodge | 3/3 | 3.1 s / 5 | 3/3 | 5.4 s / 5 |
| research-confidence | 3/3 | 0.8 s / 2 | 3/3 | 1.3 s / 2 |
| wikipedia-godel (live site) | 3/3 | 4.4 s / 5 | 3/3 | 14.0 s / 5 |
| google-flights (live site) | 0/3 | 1.5 s / 3 | 0/3 | 119.5 s / 54 |
| **all** | **18/21** |  | **15/21** |  |

Failures:

- leo:leo-4b-v5 / travel-glasshouse #0: status blocked, checks opened,filters; 4 actions, last: Destination
- leo:leo-4b-v5 / travel-glasshouse #1: status blocked, checks opened,filters; 4 actions, last: Destination
- leo:leo-4b-v5 / travel-glasshouse #2: status blocked, checks opened,filters; 4 actions, last: Destination
- jev / google-flights #0: status blocked, checks search_page,one_way,origin,destination,date,year,results; 1 actions, last: Change ticket type. Round trip
- leo:leo-4b-v5 / google-flights #0: stopped at the 120 s wall-clock limit; 28 actions, last: Search
- leo:leo-4b-v5 / google-flights #1: stopped at the 120 s wall-clock limit; 28 actions, last: Search
- jev / google-flights #1: status blocked, checks search_page,one_way,origin,destination,date,year,results; 2 actions, last: Change ticket type. Round trip
- jev / google-flights #2: status blocked, checks search_page,one_way,origin,destination,date,year,results; 2 actions, last: Change ticket type. Round trip
- leo:leo-4b-v5 / google-flights #2: stopped at the 120 s wall-clock limit; 28 actions, last: Search
