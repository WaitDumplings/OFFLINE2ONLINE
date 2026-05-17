#!/usr/bin/env python3
"""Evaluate one neural policy with configurable sampling budget/temperature."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evrptw_gen.benchmarks.DRL_Solver.models.graph_attention_model_wrapper import Agent
from evrptw_gen.benchmarks.DRL_Solver.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.DRL_Solver.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.configs.load_config import Config


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def make_env(env_id, seed, cfg):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(int(seed))
        env.action_space.seed(int(seed))
        env.observation_space.seed(int(seed))
        return env

    return thunk


def load_agent(args, device):
    agent = Agent(
        device=device,
        name=args.problem,
        tanh_clipping=args.tanh_clipping,
        n_encode_layers=args.n_encode_layers,
        value_heads=args.value_heads,
    ).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt)
    agent.eval()
    return agent


@torch.no_grad()
def select_action(agent, obs, cached_embeddings, decode_mode, temperature):
    backbone_output = agent.backbone.decode(obs, cached_embeddings)
    logits = agent.actor(backbone_output)
    mode = str(decode_mode).lower()
    if mode == "greedy":
        return logits.max(dim=2)[1]
    temp = max(float(temperature), 1e-6)
    probs = torch.distributions.Categorical(logits=logits / temp)
    return probs.sample()


def eval_policy(args):
    try:
        gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    except Exception:
        pass

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() and args.cuda else "cpu"
    config = Config(args.config_path)
    with open(args.eval_data_path, "rb") as f:
        eval_data = pickle.load(f)
    if int(args.limit) > 0:
        eval_data = eval_data[: int(args.limit)]

    agent = load_agent(args, device)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    rows = []
    start_time = time.time()
    for batch_start in range(0, len(eval_data), int(args.eval_batch_size)):
        batch_indices = list(
            range(batch_start, min(batch_start + int(args.eval_batch_size), len(eval_data)))
        )
        envs = SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    int(args.seed + args.seed_offset + idx),
                    cfg={
                        "env_mode": "eval",
                        "config": config,
                        "n_traj": int(args.n_traj),
                        "eval_data": eval_data[idx],
                    },
                )
                for idx in batch_indices
            ]
        )
        try:
            obs = envs.reset()
            cached_embeddings = agent.backbone.encode(obs)
            batch_actions = []
            final_done = None
            for step in range(int(args.eval_steps)):
                action = select_action(
                    agent,
                    obs,
                    cached_embeddings,
                    args.decode_mode,
                    args.temperature,
                )
                action_np = action.detach().cpu().numpy()
                batch_actions.append(action_np.copy())
                obs, _reward, done, _info = envs.step(action_np)
                final_done = np.asarray(done)
                if bool(np.asarray(done).all()):
                    break

            actions = (
                np.stack(batch_actions, axis=0)
                if batch_actions
                else np.zeros((0, len(batch_indices), int(args.n_traj)), dtype=np.int64)
            )
            for local_idx, data_idx in enumerate(batch_indices):
                base_env = unwrap_env(envs.envs[local_idx])
                objective = np.asarray(base_env.objective, dtype=np.float64).reshape(-1)
                done_vec = (
                    np.asarray(final_done[local_idx], dtype=bool).reshape(-1)
                    if final_done is not None and final_done.ndim >= 2
                    else np.zeros_like(objective, dtype=bool)
                )
                valid = np.isfinite(objective) & (objective > 0)
                solved = bool(np.any(valid & done_vec))
                candidate = np.where(valid & done_vec)[0] if solved else np.where(valid)[0]
                if candidate.size:
                    best_traj = int(candidate[np.argmin(objective[candidate])])
                    best_obj = float(objective[best_traj]) * 100.0
                else:
                    best_traj = 0
                    best_obj = float("inf")
                route = (
                    actions[:, local_idx, best_traj].astype(int).tolist()
                    if actions.size and best_traj < actions.shape[2]
                    else []
                )
                inst = eval_data[data_idx]
                rows.append(
                    {
                        "index": int(data_idx),
                        "file": inst.get("file", f"instance_{data_idx}.txt"),
                        "instance_id": inst.get("instance_id", f"instance_{data_idx}"),
                        "objective_value": best_obj,
                        "solved": int(solved),
                        "best_traj": best_traj,
                        "n_traj": int(args.n_traj),
                        "decode_mode": str(args.decode_mode),
                        "temperature": float(args.temperature),
                        "route": json.dumps(route),
                    }
                )
        finally:
            envs.close()

        print(
            f"[PolicyEval] {len(rows)}/{len(eval_data)} done | "
            f"elapsed={time.time() - start_time:.1f}s",
            flush=True,
        )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    finite = np.asarray(
        [r["objective_value"] for r in rows if np.isfinite(r["objective_value"])],
        dtype=np.float64,
    )
    summary = {
        "checkpoint_path": args.checkpoint_path,
        "eval_data_path": args.eval_data_path,
        "instances": len(rows),
        "n_traj": int(args.n_traj),
        "decode_mode": str(args.decode_mode),
        "temperature": float(args.temperature),
        "solved_rate": float(np.mean([r["solved"] for r in rows])) if rows else 0.0,
        "avg_objective": float(np.mean(finite)) if finite.size else float("nan"),
        "median_objective": float(np.median(finite)) if finite.size else float("nan"),
        "min_objective": float(np.min(finite)) if finite.size else float("nan"),
        "max_objective": float(np.max(finite)) if finite.size else float("nan"),
        "elapsed_sec": float(time.time() - start_time),
        "output_csv": args.output_csv,
    }
    summary_path = args.summary_json or os.path.splitext(args.output_csv)[0] + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[PolicyEvalSummary] " + json.dumps(summary, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--cuda", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--problem", default="evrptw")
    parser.add_argument("--env-id", default="evrptw-v0")
    parser.add_argument(
        "--env-entry-point",
        default="evrptw_gen.benchmarks.DRL_Solver.envs.evrp_vector_env:EVRPTWVectorEnv",
    )
    parser.add_argument("--config-path", default=str(ROOT / "evrptw_gen/configs/config.yaml"))
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--n-encode-layers", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--n-traj", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--decode-mode", choices=["sampling", "greedy"], default="sampling")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    eval_policy(parse_args())
