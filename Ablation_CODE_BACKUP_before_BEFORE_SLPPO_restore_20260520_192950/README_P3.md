# RGAE Priority 3: Asymmetric Clipping

This ablation keeps the Exp3 soft-gated archive-guided PPO baseline fixed and only changes PPO clipping.

- GPU0: Exp3 soft gate, symmetric clipping 0.20/0.20.
- GPU1: whole actor loss asymmetric clipping 0.20/0.30.
- GPU2: whole actor loss asymmetric clipping 0.20/0.40.
- GPU3: decomposed actor loss, GAE branch symmetric 0.20/0.20 and archive group/ref branch asymmetric 0.20/0.40.

Run:

```bash
conda activate maojie
SEED=3006 NUM_UPDATES=300 ./auto_run_p3.sh
```

Plot after logs exist:

```bash
python plot_p3.py --log-dir LOGS/rgae_p3_asymclip_seed3006_u300 --out-dir IMGS/rgae_p3_asymclip_seed3006_u300
```
