# P4 Best-Archive Reference Experiments

Run from the copied Ablation package after activating the environment:

```bash
cd /data/Maojie/Github2/Ablation
conda activate maojie
./auto_run_p4.sh
```

Defaults:

```text
SEED=3003
NUM_UPDATES=300
python=current conda environment python
```

Override example:

```bash
SEED=3004 NUM_UPDATES=300 ./auto_run_p4.sh
```

## GPU layout

| GPU | Experiment |
|---|---|
| GPU0 | ALNS5k-only fixed reference baseline |
| GPU1 | best-archive reference, all offline states |
| GPU2 | best-archive reference, dominant states only |
| GPU3 | best-archive reference, dominant states + linear gate |

Common settings:

```text
group_adv_coef=0.30
reference_adv_coef=0.10
reference_adv_rho=0.05
reference_adv_clip=2.0
incumbent_ratio=0.20
n_traj=50
test_agent=8
```

Reference source definitions:

```text
teacher:      ALNS5k teacher objective only
best_archive: min(ALNS teacher, current incumbent archive, policy/POMO50 best recorded during training)
```

Dominant states used by GPU2/GPU3:

```text
stable_alns_win, stable_ppo_win, lucky_ppo_win
```

Outputs:

```text
LOGS/p4_bestarchive_seed${SEED}_u${NUM_UPDATES}/
IMGS/p4_bestarchive_seed${SEED}_u${NUM_UPDATES}/
checkpoint/p4_bestarchive_seed${SEED}_u${NUM_UPDATES}/
```

`run_p4.py` waits for all four jobs and then calls `plot_ablation.py` to produce CSV, summary, and plots.
