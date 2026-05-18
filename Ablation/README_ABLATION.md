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
python = current conda environment python
```

Override example:

```bash
conda activate maojie
SEED=3004 NUM_UPDATES=300 ./all_run.sh
```


`all_run.sh` intentionally uses plain `python`. Activate the desired conda environment before running it.

The top-level path is a symlink to `/data/Maojie/Github2/OFFLINE2ONLINE/Ablation` in this machine; both paths point to the same package.

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


## Reference gate support

This package includes the latest adaptive reference-advantage gate code. `train_incumbent.py` now accepts:

```text
--reference-adv-gate-mode fixed|linear|hard
--reference-adv-gate-temp 0.05
--reference-adv-hard-threshold 0.03
```

The default `all_run.sh` ablation still runs the fixed-ref coefficient/rho sweep. To run P3-style gate experiments, add these flags to the corresponding `train_incumbent.py` command or launcher.


## P4 entry point

P4 best-archive reference experiments are configured in this package:

```bash
conda activate maojie
./auto_run_p4.sh
```

See `README_P4.md` for the GPU layout and reference-source definitions.

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
