# P1 pi-Base — 75% action-pool × 20 stability audit

**Status: PASS. Gold reproduction gate passed exactly before subsampling.**

- Primary median pairwise Kendall tau-b: **0.346**
- Mean pairwise tau-b: 0.332
- Median tau vs full Gold ranking: 0.252
- Verdict: **TAU_LT_0_4_RANKING_UNSTABLE**

| Method | mean calls diff vs random | paired bootstrap 95% CI | better instances / 20 |
|---|---:|---:|---:|
| small_first | -11.093 | [-13.373, -8.828] | 20 |
| tpe | 2.407 | [-1.240, 6.082] | 8 |
| target_mean | -1.897 | [-3.207, -0.622] | 15 |
| target_ucb | 7.052 | [4.292, 9.395] | 2 |
| target_eps | 0.762 | [-0.488, 2.170] | 8 |
| psc | 0.047 | [-1.670, 1.832] | 12 |
| wpsc | 0.047 | [-1.640, 1.847] | 12 |
| bestfirst | 0.057 | [-1.620, 1.880] | 11 |

Negative call difference means fewer calls than random. Ridge/LinUCB/kNN are N/A in the frozen pi-Base Gold ledger and were not newly invented for this audit.
