| run | final val loss | best val loss (step) | final train loss | train time (s) | wall time incl. eval (s) | ms/step |
|---|---|---|---|---|---|---|
| AdamW (2017) | 1.7005 | 1.4659 (1750) | 0.6183 | 241.6 | 363.5 | 48.5 |
| Shampoo (2018, Adam graft) | 1.8481 | 1.4745 (1500) | 0.4341 | 661.5 | 784.1 | 132.8 |
| SOAP (2024) | 1.7869 | 1.4746 (1750) | 0.4912 | 386.6 | 514.3 | 77.6 |
| Muon (2024, +aux AdamW) | 1.6071 | 1.4385 (1750) | 0.6538 | 277.2 | 399.0 | 55.7 |

| val loss threshold | AdamW (2017): step (time s) | Shampoo (2018, Adam graft): step (time s) | SOAP (2024): step (time s) | Muon (2024, +aux AdamW): step (time s) |
|---|---|---|---|---|
| ≤ 2.0 | 500 (24s) | 250 (33s) | 250 (19s) | 250 (14s) |
| ≤ 1.8 | 500 (24s) | 500 (67s) | 500 (39s) | 250 (14s) |
| ≤ 1.7 | 750 (36s) | 500 (67s) | 500 (39s) | 500 (28s) |
| ≤ 1.6 | 750 (36s) | 750 (100s) | 750 (58s) | 500 (28s) |
| ≤ 1.5 | 1250 (60s) | 1000 (133s) | 1250 (97s) | 1000 (55s) |
