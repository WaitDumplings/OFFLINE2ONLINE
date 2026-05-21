# SL-PPO Advantage Split Ablation

This ablation starts from the BEFORE_SLPPO codebase plus a narrow route-level SL-PPO auxiliary branch.

Run from this directory after activating the intended conda environment:

```bash
SEED=3007 NUM_UPDATES=300 ./auto_run_slppo_advsplit.sh
```

Outputs:

- Logs: `LOGS/slppo_advsplit_seed${SEED}_u${NUM_UPDATES}`
- Images: `IMGS/slppo_advsplit_seed${SEED}_u${NUM_UPDATES}`
- Checkpoints: `checkpoint/slppo_advsplit_seed${SEED}_u${NUM_UPDATES}`

GPU layout:

| GPU | Experiment | Actor step advantage | SL-PPO route advantage | Purpose |
|---|---|---|---|---|
| 0 | Vanilla PPO 2-head | GAE/decomposed env advantage | none | Base RL reference |
| 1 | PPO archive-only | group + soft-gated ref only | none | Test whether route-level archive signal alone can train step PPO |
| 2 | PPO GAE+ref | GAE + soft-gated ref | none | Test whether group is the noisy part and ref is sufficient |
| 3 | GAE step + SL-PPO group/ref | GAE only | group + soft-gated ref | Test the elegant split: local credit in PPO, route-quality signal in SL-PPO |

Shared incumbent settings for GPU1-3:

- `incumbent_ratio=0.20`
- regret-aware sampler on
- POMO50 regret state on
- `reference_adv_alns_win_only=true`
- `reference_adv_gate_mode=linear`
- `reference_adv_gate_temp=0.05`
- `group_adv_coef=0.30` where group is enabled
- `reference_adv_coef=0.10`, `reference_adv_rho=0.05`

GPU3 SL-PPO settings:

- `step_adv_mode=base_only`
- `use_route_level_loss=true`
- `route_loss_coef=1.0`
- `route_adv_source=group_ref`
- `route_clip_eps=0.20`
- `route_ratio_normalize=mean_logprob`
- `only_success_route_loss=true`

Plot after or during training:

```bash
python plot_slppo_advsplit.py --seed 3007 --num-updates 300 --zoom-low 940 --zoom-high 1040
```
