# P5 Sampler Ablation

P5 fixes the self-teacher/reference setting and ablates only archive sampling.

Fixed base:

- `reference_adv_source=best_archive`
- `reference_adv_gate_mode=linear`
- `reference_adv_rho=0.10`
- `reference_adv_coef=0.10`
- `group_adv_coef=0.30`
- `n_traj=50`
- `test_agent=8`

GPU layout:

| GPU | Experiment | Incumbent Ratio | Regret Kappa | Unknown Weight | Uncertain Weight |
|---|---|---:|---:|---:|---:|
| GPU0 | current sampler baseline | 0.20 | 25 | 1.0 | 0.5 |
| GPU1 | stronger high-regret sampler | 0.20 | 50 | 1.0 | 0.5 |
| GPU2 | mixed hard/random archive sampler | 0.20 | 25 | 1.5 | 1.0 |
| GPU3 | lower archive ratio sampler | 0.10 | 25 | 1.0 | 0.5 |

Run:

```bash
cd /data/Maojie/Github2/OFFLINE2ONLINE/Ablation
SEED=3003 NUM_UPDATES=300 ./auto_run_p5.sh
```

Outputs:

- Logs: `LOGS/p5_sampler_seed${SEED}_u${NUM_UPDATES}`
- Images: `IMGS/p5_sampler_seed${SEED}_u${NUM_UPDATES}`
- Metadata: `LOGS/p5_sampler_seed${SEED}_u${NUM_UPDATES}/run_metadata.json`
