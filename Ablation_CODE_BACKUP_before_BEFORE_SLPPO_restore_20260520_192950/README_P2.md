# Priority 2: Decision-Fork / High-Entropy Action Gating

This ablation keeps the Exp3 soft-gate baseline fixed and changes only where the route-level archive signal is injected.

Base actor advantage:

```text
A_step = A_GAE + h_t * (0.30 * A_group + 0.10 * g_ref * A_ref)
```

GPU mapping in `run_p2.py`:

| GPU | Experiment | Gate |
| --- | --- | --- |
| 0 | Exp3 soft gate baseline | `h_t = 1` full broadcast |
| 1 | Entropy fork | `h_t = 1[norm_entropy >= 0.60]` |
| 2 | Feasible-count fork | `h_t = 1[feasible_actions >= 5]` |
| 3 | Route-structure fork | route start, depot boundary, RS, continue-vs-return decisions |

Run after activating the target conda environment:

```bash
cd /data/Maojie/Github2/OFFLINE2ONLINE/Ablation
SEED=3005 NUM_UPDATES=300 ./auto_run_p2.sh
```

Plot after logs have eval points:

```bash
python plot_p2.py --log-dir LOGS/rgae_p2_fork_seed3005_u300 --out-dir IMGS/rgae_p2_fork_seed3005_u300
```

Important environment note:

`auto_run_p2.sh` now refuses to run under conda `base` unless `ALLOW_BASE_PYTHON=1` is set. Prefer activating the training environment first, or pass `PYTHON_BIN` explicitly:

```bash
conda activate maojie
SEED=3005 NUM_UPDATES=300 ./auto_run_p2.sh

# or without activating:
PYTHON_BIN=/home/npg0/anaconda3/envs/maojie/bin/python SEED=3005 NUM_UPDATES=300 ./auto_run_p2.sh
```

Check child training logs after launch:

```bash
ps -ef | grep p2_exp3 | grep -v grep
tail -n 60 LOGS/rgae_p2_fork_seed3005_u300/gpu0_p2_exp3_softgate_full_seed3005_u300.txt
```
