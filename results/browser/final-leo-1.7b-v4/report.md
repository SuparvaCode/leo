| task | jev passed | jev median time / decisions | leo:leo-1.7b-v4 passed | leo:leo-1.7b-v4 median time / decisions |
|---|---|---|---|---|
| travel-casa-flora | 3/3 | 3.5 s / 6 | 3/3 | 5.3 s / 6 |
| research-finite-choices | 3/3 | 0.9 s / 2 | 0/3 | 33.0 s / 61 |
| travel-glasshouse | 2/3 | 3.5 s / 6 | 3/3 | 5.3 s / 6 |
| travel-serra-lodge | 3/3 | 3.1 s / 5 | 3/3 | 4.4 s / 5 |
| research-confidence | 3/3 | 0.9 s / 2 | 0/3 | 31.6 s / 61 |
| wikipedia-godel (live site) | 3/3 | 4.3 s / 5 | 3/3 | 10.1 s / 4 |
| google-flights (live site) | 1/3 | 13.5 s / 17 | 0/3 | 48.5 s / 36 |
| **all** | **18/21** |  | **12/21** |  |

Failures:

- leo:leo-1.7b-v4 / research-finite-choices #0: ValueError: Stopped at the 60-action demo budget; 60 actions, last: ← Reading room
- leo:leo-1.7b-v4 / research-finite-choices #1: ValueError: Stopped at the 60-action demo budget; 60 actions, last: ← Reading room
- leo:leo-1.7b-v4 / research-finite-choices #2: ValueError: Stopped at the 60-action demo budget; 60 actions, last: ← Reading room
- jev / travel-glasshouse #2: status done, checks filters; 7 actions, last: View The Glasshouse
- leo:leo-1.7b-v4 / research-confidence #0: ValueError: Stopped at the 60-action demo budget; 60 actions, last: Reading room
- leo:leo-1.7b-v4 / research-confidence #1: ValueError: Stopped at the 60-action demo budget; 60 actions, last: Reading room
- leo:leo-1.7b-v4 / research-confidence #2: ValueError: Stopped at the 60-action demo budget; 60 actions, last: Reading room
- leo:leo-1.7b-v4 / google-flights #0: status blocked, checks search_page,one_way,origin,results; 26 actions, last: Lucknow
- leo:leo-1.7b-v4 / google-flights #1: status blocked, checks search_page,one_way,origin,results; 28 actions, last: Lucknow
- jev / google-flights #1: status blocked, checks search_page,one_way,origin,destination,date,year,results; 2 actions, last: Change ticket type. Round trip
- jev / google-flights #2: status blocked, checks search_page,one_way,origin,destination,date,year,results; 12 actions, last: Scroll down
- leo:leo-1.7b-v4 / google-flights #2: status blocked, checks search_page,one_way,origin,destination,date,year,results; 11 actions, last: Lucknow
