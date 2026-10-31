| run | final val loss | best val loss (step) | final train loss | train time (s) | wall time incl. eval (s) | ms/step |
|---|---|---|---|---|---|---|
| AdamW | 1.7005 | 1.4659 (1750) | 0.6183 | 241.6 | 363.5 | 48.5 |
| Muon (+aux AdamW) | 1.6071 | 1.4385 (1750) | 0.6538 | 277.2 | 399.0 | 55.7 |
| SOAP | 1.7869 | 1.4746 (1750) | 0.4912 | 386.6 | 514.3 | 77.6 |
| Shampoo (Adam graft) | 1.8481 | 1.4745 (1500) | 0.4341 | 661.5 | 784.1 | 132.8 |

| val loss threshold | AdamW: step (time s) | Muon (+aux AdamW): step (time s) | SOAP: step (time s) | Shampoo (Adam graft): step (time s) |
|---|---|---|---|---|
| ≤ 2.0 | 500 (24s) | 250 (14s) | 250 (19s) | 250 (33s) |
| ≤ 1.8 | 500 (24s) | 250 (14s) | 500 (39s) | 500 (67s) |
| ≤ 1.7 | 750 (36s) | 500 (28s) | 500 (39s) | 500 (67s) |
| ≤ 1.6 | 750 (36s) | 500 (28s) | 750 (58s) | 750 (100s) |
| ≤ 1.5 | 1250 (60s) | 1000 (55s) | 1250 (97s) | 1000 (133s) |
