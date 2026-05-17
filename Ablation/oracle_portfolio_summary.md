# Oracle Portfolio Sanity Check

- Labels: `/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000.csv`
- Per-instance output: `/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/oracle_portfolio_pi_ref_gpu1_seed2030_vs_alns_3000.csv`
- JSON summary: `/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/oracle_portfolio_pi_ref_gpu1_seed2030_vs_alns_3000_summary.json`

## Overall

- n: `1024`
- j_ref_mean: `957.023249`
- j_teacher_mean: `947.564667`
- j_oracle_mean: `923.034191`
- oracle_gap_abs_mean: `33.989058`
- oracle_gap_pct_ref_mean: `4.196284`
- alns_win_count: `598`
- ppo_win_count: `325`
- tie_count: `101`
- alns_win_ratio: `0.583984`
- ppo_win_ratio: `0.317383`
- tie_ratio: `0.098633`

## High-confidence ALNS-win thresholds

- ALNS improves ref by > 0.5%: `626` (61.13%)
- ALNS improves ref by > 1.0%: `598` (58.40%)
- ALNS improves ref by > 2.0%: `555` (54.20%)
- ALNS improves ref by > 3.0%: `489` (47.75%)
- ALNS improves ref by > 5.0%: `353` (34.47%)

## By Bucket

| bucket    |          n |   j_ref_mean |   j_teacher_mean |   j_oracle_mean |   oracle_gap_abs_mean |   oracle_gap_pct_ref_mean |   alns_win_ratio |   ppo_win_ratio |   tie_ratio |   teacher_win_count |   ppo_win_count |   tie_count |
|:----------|-----------:|-------------:|-----------------:|----------------:|----------------------:|--------------------------:|-----------------:|----------------:|------------:|--------------------:|----------------:|------------:|
| C_narrow  | 190.000000 |   798.082935 |       790.676911 |      769.095082 |             28.987853 |                  3.695047 |         0.568421 |        0.336842 |    0.094737 |          108.000000 |       64.000000 |   18.000000 |
| C_wide    | 214.000000 |   641.509819 |       585.905714 |      583.121176 |             58.388643 |                  9.156611 |         0.897196 |        0.060748 |    0.042056 |          192.000000 |       13.000000 |    9.000000 |
| RC_narrow | 114.000000 |  1085.766485 |      1118.478919 |     1075.035106 |             10.731380 |                  1.005558 |         0.254386 |        0.552632 |    0.192982 |           29.000000 |       63.000000 |   22.000000 |
| RC_wide   |  97.000000 |   882.448733 |       837.003788 |      831.572601 |             50.876132 |                  5.704449 |         0.783505 |        0.154639 |    0.061856 |           76.000000 |       15.000000 |    6.000000 |
| R_narrow  | 193.000000 |  1331.932266 |      1389.487283 |     1320.688956 |             11.243309 |                  0.848395 |         0.222798 |        0.663212 |    0.113990 |           43.000000 |      128.000000 |   22.000000 |
| R_wide    | 216.000000 |  1039.977382 |      1008.457284 |     1000.747692 |             39.229691 |                  3.720908 |         0.694444 |        0.194444 |    0.111111 |          150.000000 |       42.000000 |   24.000000 |

## By Regret Class

| regret_class   |          n |   j_ref_mean |   j_teacher_mean |   j_oracle_mean |   oracle_gap_abs_mean |   oracle_gap_pct_ref_mean |   alns_win_ratio |   ppo_win_ratio |   tie_ratio |   teacher_win_count |   ppo_win_count |   tie_count |
|:---------------|-----------:|-------------:|-----------------:|----------------:|----------------------:|--------------------------:|-----------------:|----------------:|------------:|--------------------:|----------------:|------------:|
| rl_win         | 325.000000 |  1101.851245 |      1178.496223 |     1101.851245 |              0.000000 |                  0.000000 |         0.000000 |        1.000000 |    0.000000 |            0.000000 |      325.000000 |    0.000000 |
| teacher_win    | 598.000000 |   866.654546 |       808.924007 |      808.924007 |             57.730540 |                  7.137609 |         1.000000 |        0.000000 |    0.000000 |          598.000000 |        0.000000 |    0.000000 |
| tie            | 101.000000 |  1026.046871 |      1025.330604 |     1023.255456 |              2.791415 |                  0.284209 |         0.000000 |        0.000000 |    1.000000 |            0.000000 |        0.000000 |  101.000000 |
