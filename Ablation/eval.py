import os
import argparse
from distutils.util import strtobool
import time
import numpy as np
from tqdm import tqdm
import pickle
from evrptw_gen.configs.load_config import Config
import warnings
import pandas as pd
import json

warnings.filterwarnings(
    "ignore",
    message="WARN: A Box observation space has an unconventional shape*",
    category=UserWarning,
)

import gym
import torch
from evrptw_gen.benchmarks.DRL_Solver.models.graph_attention_model_wrapper import Agent
from evrptw_gen.benchmarks.DRL_Solver.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.DRL_Solver.wrappers.syncVectorEnvPomo import SyncVectorEnv

def count_vehicle_num(actions):
    depot_num = 0
    id_action = actions.transpose(1,0)
    for route in id_action:
        for i in range(1, len(route)):
            if route[i] == 0:
                if i == len(route)-1 or route[i+1] != 0:
                    depot_num += 1
    return depot_num / id_action.shape[0]

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="seed of the experiment",
    )
    parser.add_argument(
        "--cuda-id", type=int, default=0,
        help="cuda device id",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="cleanRL",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )
    # Algorithm specific arguments
    parser.add_argument(
        "--problem",
        type=str,
        default="evrptw",
        help="the OR problem we are trying to solve, it will be passed to the agent",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="evrptw-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--env-entry-point",
        type=str,
        default="evrptw_gen.benchmarks.DRL_Solver.envs.evrp_vector_env:EVRPTWVectorEnv",
        help="the path to the definition of the environment",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="tanh clipping in the agent",
    )
    parser.add_argument(
        "--n_encode_layers",
        type=int,
        default=3,
        help="number of encoder layers",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=1024,
        help="number of trajectories(players) in a vectorized sub-environment",
    )
    parser.add_argument(
        "--test_agent",
        type=int,
        default=1,
        help="test agent",
    )
    parser.add_argument("--value_heads", type=int, default=2, help="number of critic value heads in checkpoint")
    parser.add_argument(
        "--use-residual-adapter",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Enable the residual adapter module when loading a CARD Stage D checkpoint.",
    )
    parser.add_argument("--adapter-hidden-dim", type=int, default=64)
    parser.add_argument("--adapter-beta-max", type=float, default=0.0)
    parser.add_argument(
        "--adapter-gate-threshold",
        type=float,
        default=-1.0,
        help=(
            "Eval-only hard gate for residual adapter. Values >=0 make the "
            "adapter active only when regret gate >= threshold."
        ),
    )
    parser.add_argument(
        "--adapter-contextual",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Enable contextual residual adapter weights used by full CARD Stage D.",
    )
    parser.add_argument(
        "--adapter-node-contextual",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Enable node-aware action residual adapter weights.",
    )
    parser.add_argument(
        "--adapter-scale-mode",
        choices=["none", "std", "zscore"],
        default="none",
        help=(
            "How to calibrate residual logits before adding them to actor logits. "
            "'zscore' normalizes actor logits for the adapter and rescales the "
            "delta by the masked actor-logit std."
        ),
    )
    parser.add_argument("--adapter-scale-min", type=float, default=0.05)
    parser.add_argument("--adapter-scale-max", type=float, default=10.0)
    parser.add_argument(
        "--use-regret-gate",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Enable the regret gate module when loading a CARD Stage D checkpoint.",
    )
    # '1.26.4' np
    # /data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/alns_teacher_successcoef_freq5/Cus_50_CS_12/best_model.pth
    parser.add_argument("--env_mode", type=str, default="eval", help="env mode: train / eval")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="the batch size for evaluation")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/alns_teacher_successcoef_freq5/Cus_50_CS_12/best_model.pth", help="path to load model checkpoint")
    parser.add_argument("--eval_data_path", type=str, default="./dataset/unanchored/Cus_50/buffer/pickle/evrptw_50C_12R.pkl", help="path to evaluation data when eval_env_mode is solomon_txt")
    parser.add_argument("--save_log_dir", type=str, default="checkpoint", help="directory to save models and logs")
    parser.add_argument("--output_csv", type=str, default="rl_results.csv", help="path to save per-instance RL objective results")
    parser.add_argument("--objective_csv", type=str, default="objective_results.csv", help="path to save objective-only CSV")
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="sampling",
        choices=["greedy", "sampling", "pomo"],
        help="evaluation decode mode; sampling/pomo with test_agent > 1 uses best objective among trajectories",
    )
    parser.add_argument(
        "--adaptive-decode",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help=(
            "Stage E: use the regret gate to assign per-instance neural decoding "
            "budgets and residual strength. No ALNS is run at inference time."
        ),
    )
    parser.add_argument(
        "--adaptive-budgets",
        type=str,
        default="4,8,16",
        help="Comma-separated low,mid,high POMO budgets for --adaptive-decode.",
    )
    parser.add_argument(
        "--adaptive-thresholds",
        type=str,
        default="0.35,0.65",
        help="Comma-separated low/high gate thresholds for --adaptive-decode.",
    )
    parser.add_argument(
        "--adaptive-residual-scales",
        type=str,
        default="0.0,0.5,1.0",
        help=(
            "Comma-separated low,mid,high multipliers for adapter_beta_max. "
            "Low=0 recovers the frozen PPO actor on easy instances."
        ),
    )
    parser.add_argument(
        "--adaptive-probe-batch-size",
        type=int,
        default=256,
        help="Batch size used for initial gate probing.",
    )
    parser.add_argument(
        "--adaptive-gate-csv",
        type=str,
        default="",
        help="Optional CSV path for per-instance Stage E gate/budget assignments.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./evrptw_gen/configs/config.yaml",
        help="path to evrptw_gen config file",
    )

    args = parser.parse_args()
    # fmt: on
    return args

def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

def make_env(env_id, seed, cfg=None):
    if cfg is None:
        cfg = {}

    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(int(seed))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def _parse_float_list(text, expected, name):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values.")
    return [float(p) for p in parts]

def _parse_int_list(text, expected, name):
    values = _parse_float_list(text, expected, name)
    ints = [int(v) for v in values]
    if any(v <= 0 for v in ints):
        raise ValueError(f"{name} values must be positive.")
    return ints

def _initial_gate_scores(agent, args, config, eval_data, device):
    if not bool(args.use_residual_adapter) or not bool(args.use_regret_gate):
        return np.zeros(len(eval_data), dtype=np.float32)

    scores = np.zeros(len(eval_data), dtype=np.float32)
    batch_size = max(1, int(args.adaptive_probe_batch_size))
    agent.eval()

    old_beta = float(getattr(agent, "adapter_beta_max", 0.0))
    # Gate probing should not depend on the residual strength.
    agent.adapter_beta_max = max(old_beta, 1.0)
    try:
        for start in range(0, len(eval_data), batch_size):
            idxs = list(range(start, min(start + batch_size, len(eval_data))))
            envs = SyncVectorEnv(
                [
                    make_env(
                        args.env_id,
                        int(args.seed + 700000 + i),
                        cfg={
                            "env_mode": args.env_mode,
                            "config": config,
                            "n_traj": 1,
                            "eval_data": eval_data[i],
                        },
                    )
                    for i in idxs
                ]
            )
            try:
                obs = envs.reset()
                with torch.no_grad():
                    cached = agent.backbone.encode(obs)
                    backbone_output = agent.backbone.decode(
                        obs,
                        cached,
                        use_mask=False,
                    )
                    gate = agent.get_residual_gate(backbone_output)
                    if gate is None:
                        batch_scores = np.zeros(len(idxs), dtype=np.float32)
                    else:
                        batch_scores = (
                            gate.reshape(len(idxs), -1)
                            .mean(dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                scores[idxs] = batch_scores
            finally:
                envs.close()
    finally:
        agent.adapter_beta_max = old_beta

    return scores

def _stage_e_assignments(scores, args, base_beta):
    budgets = _parse_int_list(args.adaptive_budgets, 3, "adaptive_budgets")
    thresholds = _parse_float_list(args.adaptive_thresholds, 2, "adaptive_thresholds")
    scales = _parse_float_list(args.adaptive_residual_scales, 3, "adaptive_residual_scales")
    if thresholds[0] > thresholds[1]:
        raise ValueError("adaptive_thresholds must be low,high with low <= high.")

    assignments = []
    for score in scores:
        score = float(score)
        if score < thresholds[0]:
            tier_idx, tier = 0, "low"
        elif score < thresholds[1]:
            tier_idx, tier = 1, "mid"
        else:
            tier_idx, tier = 2, "high"
        assignments.append(
            {
                "gate_score": score,
                "tier": tier,
                "budget": int(budgets[tier_idx]),
                "residual_beta": float(base_beta) * float(scales[tier_idx]),
            }
        )
    return assignments

def _run_eval_indices(
    agent,
    args,
    config,
    eval_data,
    indices,
    n_traj,
    residual_beta,
    decode_mode,
    device,
    assignments=None,
):
    rows = []
    if len(indices) == 0:
        return rows

    old_beta = float(getattr(agent, "adapter_beta_max", 0.0))
    agent.adapter_beta_max = float(residual_beta)
    try:
        for batch_start in range(0, len(indices), int(args.eval_batch_size)):
            batch_test_env_id = indices[
                batch_start : min(batch_start + int(args.eval_batch_size), len(indices))
            ]
            test_envs = SyncVectorEnv(
                [
                    make_env(
                        args.env_id,
                        int(args.seed + i),
                        cfg={
                            "env_mode": args.env_mode,
                            "config": config,
                            "n_traj": int(n_traj),
                            "eval_data": eval_data[i],
                        },
                    )
                    for i in batch_test_env_id
                ]
            )

            try:
                test_obs = test_envs.reset()
                batch_actions = []
                final_done = None
                for _step in tqdm(range(0, args.eval_steps)):
                    with torch.no_grad():
                        if decode_mode == "sampling":
                            action, _, _, _ = agent.get_action_and_value(test_obs)
                        else:
                            action, _ = agent(test_obs)
                    action = action.to("cpu").numpy()
                    batch_actions.append(action.copy())
                    test_obs, _, test_done, _test_info = test_envs.step(action)
                    final_done = np.asarray(test_done)
                    if test_done.all():
                        break

                action_arr = (
                    np.stack(batch_actions, axis=0)
                    if batch_actions
                    else np.zeros((0, 0, 0), dtype=np.int32)
                )
                for local_idx, data_idx in enumerate(batch_test_env_id):
                    base_env = unwrap_env(test_envs.envs[local_idx])
                    objective = np.asarray(base_env.objective, dtype=np.float64).reshape(-1)
                    done_vec = (
                        np.asarray(final_done[local_idx], dtype=bool).reshape(-1)
                        if final_done is not None and final_done.ndim >= 2
                        else np.zeros_like(objective, dtype=bool)
                    )
                    valid = np.isfinite(objective) & (objective > 0)
                    solved = bool(np.any(valid & done_vec))
                    if solved:
                        candidate = np.where(valid & done_vec)[0]
                    else:
                        candidate = np.where(valid)[0]

                    if candidate.size:
                        best_traj = int(candidate[np.argmin(objective[candidate])])
                        best_objective = float(objective[best_traj]) * 100.0
                    else:
                        best_traj = 0
                        best_objective = float("inf")

                    route_actions = (
                        action_arr[:, local_idx, best_traj].astype(int).tolist()
                        if action_arr.size and best_traj < action_arr.shape[2]
                        else []
                    )
                    assign = assignments[data_idx] if assignments is not None else {}
                    rows.append(
                        {
                            "index": int(data_idx),
                            "file": eval_data[data_idx].get("file", f"solomon_dataset_{data_idx}.txt"),
                            "instance_id": eval_data[data_idx].get("instance_id", f"solomon_dataset_{data_idx}"),
                            "objective_value": best_objective,
                            "solved": solved,
                            "best_traj": best_traj,
                            "routes": json.dumps(route_actions),
                            "adaptive_gate": assign.get("gate_score", np.nan),
                            "adaptive_tier": assign.get("tier", "fixed"),
                            "adaptive_budget": int(n_traj),
                            "adaptive_residual_beta": float(residual_beta),
                        }
                    )
            finally:
                test_envs.close()
    finally:
        agent.adapter_beta_max = old_beta

    return rows

def main(args):
    #########################
    #### Env Definition #####
    #########################
    # Register the environment.
    # Note: entry_point must be a fully-qualified import path 
    # (details explained in the discussion above).
    gym.envs.register(
        id=args.env_id,
        entry_point=args.env_entry_point,
    )
    print(args.checkpoint_path)
    print(args.eval_data_path)
    decode_mode = str(args.decode_mode).lower()
    if decode_mode == "pomo":
        decode_mode = "sampling"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    #########################
    ### Model Definition ####
    #########################
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"

    agent = Agent(device=device, 
                  name=args.problem, 
                  tanh_clipping = args.tanh_clipping, 
                  n_encode_layers = args.n_encode_layers,
                  value_heads = args.value_heads,
                  use_residual_adapter=args.use_residual_adapter,
                  adapter_hidden_dim=args.adapter_hidden_dim,
                  adapter_beta_max=args.adapter_beta_max,
                  use_regret_gate=args.use_regret_gate,
                  adapter_contextual=args.adapter_contextual,
                  adapter_node_contextual=args.adapter_node_contextual,
                  adapter_gate_threshold=args.adapter_gate_threshold,
                  adapter_scale_mode=args.adapter_scale_mode,
                  adapter_scale_min=args.adapter_scale_min,
                  adapter_scale_max=args.adapter_scale_max).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    try:
        agent.load_state_dict(ckpt)
    except RuntimeError as exc:
        message = str(exc)
        adapter_enabled = bool(args.use_residual_adapter)
        legacy_route_event = (
            "route_event_gru" in message
            or "route_event_seq_norm" in message
        )
        if adapter_enabled:
            current = agent.state_dict()
            compatible = {
                k: v
                for k, v in ckpt.items()
                if k in current and tuple(v.shape) == tuple(current[k].shape)
            }
            skipped = sorted(set(ckpt.keys()) - set(compatible.keys()))
            current.update(compatible)
            agent.load_state_dict(current)
            print(
                "Loaded compatible adapter checkpoint; "
                f"loaded={len(compatible)}, skipped={len(skipped)}"
            )
        elif not legacy_route_event:
            raise
        else:
            fusion = getattr(agent.backbone.decoder, "structured_state_fusion", None)
            if fusion is None or not hasattr(fusion, "use_route_event_gru"):
                raise

            fusion.use_route_event_gru = False
            incompatible = agent.load_state_dict(ckpt, strict=False)
            allowed_missing = {
                "backbone.decoder.structured_state_fusion.route_event_gru.weight_ih_l0",
                "backbone.decoder.structured_state_fusion.route_event_gru.weight_hh_l0",
                "backbone.decoder.structured_state_fusion.route_event_gru.bias_ih_l0",
                "backbone.decoder.structured_state_fusion.route_event_gru.bias_hh_l0",
                "backbone.decoder.structured_state_fusion.route_event_seq_norm.weight",
                "backbone.decoder.structured_state_fusion.route_event_seq_norm.bias",
            }
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            if not missing.issubset(allowed_missing) or unexpected:
                raise RuntimeError(
                    "Unsupported checkpoint mismatch while loading legacy route-event checkpoint: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                ) from exc
            print("Loaded legacy weighted-sum route-event checkpoint; disabled route_event_gru.")
    agent.to(device)
    print("Loaded model from {}".format(args.checkpoint_path))
    agent.eval()

    save_log_dir = args.save_log_dir
    os.makedirs(save_log_dir, exist_ok=True)

    # num_updates = args.total_timesteps // args.batch_size
    config = Config(args.config_path)
    eval_data = pickle.load(open(args.eval_data_path, "rb"))

    num_test_envs = len(eval_data)
    t_eval_start = time.time()

    if bool(args.adaptive_decode):
        base_beta = float(getattr(agent, "adapter_beta_max", 0.0))
        gate_scores = _initial_gate_scores(
            agent=agent,
            args=args,
            config=config,
            eval_data=eval_data,
            device=device,
        )
        assignments = _stage_e_assignments(gate_scores, args, base_beta)
        if args.adaptive_gate_csv:
            gate_df = pd.DataFrame(
                [
                    {
                        "index": idx,
                        "file": eval_data[idx].get("file", f"solomon_dataset_{idx}.txt"),
                        "instance_id": eval_data[idx].get("instance_id", f"solomon_dataset_{idx}"),
                        **assignments[idx],
                    }
                    for idx in range(num_test_envs)
                ]
            )
            gate_df.to_csv(args.adaptive_gate_csv, index=False)

        rows = []
        print("[StageE] adaptive decode enabled")
        for tier in ["low", "mid", "high"]:
            tier_indices = [
                idx for idx, item in enumerate(assignments) if item["tier"] == tier
            ]
            if not tier_indices:
                continue
            budget = int(assignments[tier_indices[0]]["budget"])
            residual_beta = float(assignments[tier_indices[0]]["residual_beta"])
            print(
                f"[StageE] tier={tier} | n={len(tier_indices)} | "
                f"budget={budget} | residual_beta={residual_beta:.4f} | "
                f"gate_mean={float(np.mean(gate_scores[tier_indices])):.4f}"
            )
            rows.extend(
                _run_eval_indices(
                    agent=agent,
                    args=args,
                    config=config,
                    eval_data=eval_data,
                    indices=tier_indices,
                    n_traj=budget,
                    residual_beta=residual_beta,
                    decode_mode="sampling",
                    device=device,
                    assignments=assignments,
                )
            )
        rows = sorted(
            rows,
            key=lambda row: int(row.get("index", 0)),
        )
    else:
        rows = _run_eval_indices(
            agent=agent,
            args=args,
            config=config,
            eval_data=eval_data,
            indices=list(range(num_test_envs)),
            n_traj=int(args.test_agent),
            residual_beta=float(getattr(agent, "adapter_beta_max", 0.0)),
            decode_mode=decode_mode,
            device=device,
            assignments=None,
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    df[["file", "objective_value"]].to_csv(args.objective_csv, index=False)

    finite_obj = pd.to_numeric(df["objective_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    solved_rate = float(df["solved"].mean()) if len(df) else 0.0
    print("Eval Time: {:.4f}s".format(time.time() - t_eval_start))
    print("Instances: {}".format(len(df)))
    print("Solved Rate: {:.3f}%".format(solved_rate * 100.0))
    print("Average Objective: {:.6f}".format(float(finite_obj.mean())))
    print("Best Objective: {:.6f}".format(float(finite_obj.min())))
    print("Saved RL results to: {}".format(args.output_csv))
    print("Saved objective results to: {}".format(args.objective_csv))

if __name__ == "__main__":
    args = parse_args()
    main(args)
