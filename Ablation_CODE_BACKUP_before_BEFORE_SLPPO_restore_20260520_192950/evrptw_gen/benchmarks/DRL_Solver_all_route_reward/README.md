## References for learning-based and reward-shaping components

- **Learning-based solver (TERRAN++)**  
  The design and training paradigm of the DRL solver are inspired by:  
  https://ieeexplore.ieee.org/document/11245586

- **Progress-based shaping bonus (PBSB)**  
  The reward shaping formulation follows standard progress-based shaping principles as discussed in:  
  https://www.teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf

These references are provided for background and conceptual context only.  
All implementations in this repository are self-contained and adapted to the
EVRP-TW benchmark semantics and protocols.

## DRL training (run from repository root)

To run DRL training, **first `cd` to the repository root**, where `train.py` and
`eval.py` are located. The DRL solver uses a shared argument interface, so the
same set of CLI flags can be passed through the root-level entry scripts.

> **Important:** always launch training from the **repo root** (not from inside
> `evrptw_gen/benchmarks/DRL_Solver/`) to ensure relative paths (configs,
> datasets, checkpoints) resolve correctly.

## DRL (PPO) hyperparameters

The DRL baseline uses PPO with a largely shared hyperparameter setup across
training scales. The following table summarizes the **scale-dependent** and
**shared** settings used in our experiments.

| Hyperparameter | C5 | C15 | C100 | C1K |
|---|---:|---:|---:|---:|
| # Parallel envs | 1024 | 512 | 256 | 64 |
| Max decoding steps per env | 20 | 40 | 150 | 1400 |
| # Minibatches per update | 1 | 8 | 16 | 64 |
| Grad. accumulation steps | 1 | 1 | 4 | 6 |
| PBSB magnitude (\(\alpha\)) | 1 | 1 | 5 | 10 |

**Shared across scales:**

- Training iterations (updates): **1,000**
- Learning rate (actor): **\(3e-5\)**
- Learning rate (critic): **\(3e-5\)**
- Weight decay: **\(1e-5\)**
- Discount (\(\gamma\)): **0.99**
- GAE (\(\lambda\)): **0.95**
- PPO epochs per rollout: **3**
- PPO clip (\(\epsilon\)): **0.2**
- Entropy coefficient: **0.001**
- Value coefficient: **0.5**
- Target KL: **0.01**
- Max grad norm (backbone): **2.0**
- Max grad norm (critic): **10.0**
- #Encoder layers: **3**
- Tanh clipping: **10.0**
- PBSB power (\(\beta\)): **0.5**

### Train with on-the-fly evaluation (recommended)

The training script supports **on-the-fly evaluation** on a fixed benchmark set.
This is the recommended workflow for reproducing the DRL (TERRAN++) results.

#### Example: train and evaluate on **Cus5**

```bash
cd <PATH_TO_EVRP_TW_D_B_REPO>

python train.py \
  --env_mode train \
  --config_path ./evrptw_gen/configs/config.yaml \
  --train_cus_num 5 \
  --train_cs_num 2 \
  --eval_data_path ./dataset/anchored/Cus_5/pickle/evrptw_5C_2R.pkl \
  --eval_batch_size 100 \
  --seed 1234 \
  --save-dir ./checkpointve-dir ./checkpoint
```

This command:
- trains **TERRAN++** on **Cus5** instances generated on the fly,
- performs periodic evaluation on a fixed **Cus5** benchmark set,
- saves model checkpoints and evaluation logs to `./checkpoint`.

### CLI arguments (training / evaluation)

Below is a **minimal reference table** for commonly used CLI arguments in the
training and evaluation scripts (e.g., `train.py`,
`evrptw_gen/benchmark_train.py`).  
Low-level engineering options (e.g., WandB, entry points) are omitted for clarity.

| Argument | Default | Description |
|--------|---------|-------------|
| `--config_path` | `./evrptw_gen/configs/config.yaml` | Path to EVRP-TW generator configuration |
| `--train_cus_num` | `5` | Number of customers in training instances |
| `--train_cs_num` | `2` | Number of charging stations in training instances |
| `--num-updates` | `1500` | Number of PPO update iterations |
| `--num-envs` | `256` | Number of parallel vectorized environments |
| `--num-steps` | `150` | Rollout length per environment |
| `--learning-rate` | `3e-5` | Learning rate for policy network |
| `--critic-lr` | `3e-5` | Learning rate for value network |
| `--gamma` | `0.99` | Discount factor |
| `--gae-lambda` | `0.95` | GAE parameter |
| `--target-success` | `0.95` | Target feasibility (success) rate |
| `--lambda-fail-init` | `10.0` | Initial constraint multiplier |
| `--lambda-max` | `50.0` | Maximum constraint multiplier |
| `--test-sample-mode` | `greedy` | Decoding strategy during evaluation |
| `--multi-greedy-inference` | `True` | Enable multi-trajectory greedy inference |
| `--eval_data_path` | `./dataset/anchored/Cus_5/pickle/evrptw_5C_2R.pkl` | Path to evaluation dataset (pickle) |
| `--eval_batch_size` | `100` | Batch size for evaluation |
| `--seed` | `1234` | Random seed for reproducibility |
| `--cuda-id` | `0` | CUDA device index |
| `--save-dir` | `./checkpoint` | Directory for saving checkpoints |

> This table lists only the **most relevant parameters** for reproducing

## Cross-scale evaluation (e.g, train on Cus5, evaluate on Cus100)
To evaluate a trained model on a different scale, run the standalone evaluation script.

**Example: train on Cus5, evaluate on Cus100**
```shell
cd <PATH_TO_EVPR_TW_D_B_REPO>

python eval.py \
  --checkpoint_path ./checkpoint/Cus_5_CS_2/best_model.pth \
  --eval_data_path ./edataset/anchoredval/Cus_100/pickle/evrptw_100C_20R.pkl \
  --config_path ./evrptw_gen/configs/config.yaml \
  --eval_batch_size 1000 \
  --cuda-id 0
```

This command:
- loads the trained TERRAN++ checkpoint,
- evaluates it on the Cus100 benchmark instances,
- runs the environment in evaluation mode under unified EVRP-TW semantics,
- reports feasibility and reward statistics.

### Training / evaluation outputs

During training and evaluation, the script prints a concise summary to `stdout`
and saves model checkpoints to disk.

#### Console output

For each evaluation phase, the following information is reported:

- **Number of Customers / Charging Stations**  
  The instance scale used in the current evaluation batch.

- **Average reward**  
  Mean episode reward over all evaluated episodes (computed only when all episodes terminate normally).

- **Step**  
  Current training step at which the evaluation is performed.

- **Avg Done Step**  
  Average number of steps until episode termination (proxy for route length / decoding depth).

- **#CS visited**  
  Average number of charging stations visited per episode.

- **Action trace**  
  A human-readable sequence of visited nodes (for debugging and qualitative inspection).

- **Evaluation time**  
  Wall-clock time for the evaluation batch.

#### Model checkpoints

The following checkpoints are written to `--save-dir`:

- `best_model.pth`  
  Saved when the current evaluation achieves the **best average reward** so far
  and all evaluation episodes terminate successfully.

- `cur_model.pth`  
  Snapshot of the model after the current evaluation phase.

These logs allow reviewers to verify training stability, feasibility behavior,
and scale progression without inspecting internal code.
