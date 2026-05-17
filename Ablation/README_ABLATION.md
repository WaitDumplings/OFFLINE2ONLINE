# Ablation run package

This directory is designed to be portable. It contains:

- current best code snapshot;
- Cus_50 eval pickle;
- ALNS5k offline buffer;
- one-command runner for a 4-GPU reference-adv ablation.

## Run

```bash
cd /data/Maojie/Github2/Ablation
./all_run.sh
```

Default settings:

```text
SEED=3003
NUM_UPDATES=300
PYTHON_BIN=/home/npg/miniconda3/envs/maojie/bin/python
```

Override example:

```bash
PYTHON_BIN=/path/to/conda/env/bin/python SEED=3004 NUM_UPDATES=300 ./all_run.sh
```

## GPU layout

| GPU | Experiment |
|---|---|
| GPU0 | Vanilla PPO 2-head |
| GPU1 | Exp3, reference_adv_coef=0.05, reference_adv_rho=0.05 |
| GPU2 | Exp3, reference_adv_coef=0.10, reference_adv_rho=0.10 |
| GPU3 | Exp3, reference_adv_coef=0.15, reference_adv_rho=0.10 |

All GPU1-3 experiments keep:

```text
group_adv_coef=0.30
group_adv_clip=3.0
reference_adv_clip=2.0
reference_adv_alns_win_only=true
incumbent_ratio=0.20
n_traj=50
test_agent=8
```

## Outputs

For `SEED=3003` and `NUM_UPDATES=300`:

```text
LOGS/ref_sweep_seed3003_u300/
IMGS/ref_sweep_seed3003_u300/
checkpoint/ref_sweep_seed3003_u300/
```

The runner waits for all four jobs to finish and then writes:

```text
IMGS/ref_sweep_seed3003_u300/eval_points.csv
IMGS/ref_sweep_seed3003_u300/summary.md
IMGS/ref_sweep_seed3003_u300/*objective*.png
IMGS/ref_sweep_seed3003_u300/*solved_rate.png
IMGS/ref_sweep_seed3003_u300/*avg_cs.png
```
