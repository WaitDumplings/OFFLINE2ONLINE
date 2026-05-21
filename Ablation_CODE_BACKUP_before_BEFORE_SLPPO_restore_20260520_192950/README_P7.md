# P7 Milestone Reference Ablation

P7 tests whether a reference route can provide dense quality guidance without step-level imitation.
It builds on the current P5 best setting:

- `n_traj=50`, eval `test_agent=8`
- POMO50 regret-aware sampler
- `incumbent_ratio=0.10`
- `group_adv_coef=0.30`
- `reference_adv_source=best_archive`
- `reference_adv_coef=0.10`, `reference_adv_rho=0.10`
- linear reference gate
- incumbent archive updates enabled

## Milestone Signal

For each offline/reference instance, the reference route is converted to a cumulative distance table:

```text
B_ref[k] = reference cumulative objective after serving the kth customer
```

During a PPO rollout, when a trajectory newly reaches served count `k`, P7 adds:

```text
A_mile_raw(t) = (B_ref[k] - C_policy(t)) / (rho_mile * B_ref[k])
A_mile_used(t) = clip(A_mile_raw(t), -clip, clip) * ref_gate
A_actor += milestone_ref_coef * A_mile_used
```

This does not imitate the reference action. It only says whether the current partial solution is ahead of or behind the reference cumulative-cost milestone.

## Four GPU Jobs

- GPU0: control, P5 best sampler, no milestone
- GPU1: `milestone_ref_coef=0.05`, `milestone_ref_rho=0.10`
- GPU2: `milestone_ref_coef=0.10`, `milestone_ref_rho=0.10`
- GPU3: `milestone_ref_coef=0.10`, `milestone_ref_rho=0.05`

## Run

```bash
cd /data/Maojie/Github2/OFFLINE2ONLINE/Ablation
conda activate maojie
./auto_run_p7.sh
```

Override seed or epoch count:

```bash
SEED=3005 NUM_UPDATES=300 ./auto_run_p7.sh
```

Outputs:

- logs: `LOGS/p7_milestone_seed${SEED}_u${NUM_UPDATES}`
- figures: `IMGS/p7_milestone_seed${SEED}_u${NUM_UPDATES}`
- checkpoints: `checkpoint/p7_milestone_seed${SEED}_u${NUM_UPDATES}`
