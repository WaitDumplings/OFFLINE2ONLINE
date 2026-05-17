#!/usr/bin/env python3
"""Evaluate frozen base policy plus a bounded residual proposal head."""

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
from mode_proposal_v1.train_bounded_residual_proposal import (
    BoundedResidualHead,
    build_features,
    compose_proposal_logits,
)


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


def load_base(args, device):
    agent = Agent(
        device=device,
        name=args.problem,
        tanh_clipping=args.tanh_clipping,
        n_encode_layers=args.n_encode_layers,
        value_heads=args.value_heads,
    ).to(device)
    agent.load_state_dict(torch.load(args.base_checkpoint_path, map_location=device))
    agent.eval()
    return agent


def load_head(args, device):
    payload = torch.load(args.head_path, map_location=device)
    head = BoundedResidualHead(
        in_dim=int(payload.get("in_dim", 7)),
        hidden=int(payload.get("hidden", 64)),
        max_residual=float(payload.get("max_residual", args.max_residual)),
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    args.feature_mode = str(payload.get("feature_mode", args.feature_mode))
    args.final_logit_mode = str(payload.get("final_logit_mode", args.final_logit_mode))
    return head


@torch.no_grad()
def select_action(args, base, head, obs, cached, step_idx=0):
    base_out = base.backbone.decode(obs, cached)
    base_logits_raw = base.actor(base_out)
    batch_size, n_traj, num_nodes_from_logits = base_logits_raw.shape
    base_logits = base_logits_raw.reshape(batch_size * n_traj, num_nodes_from_logits)
    num_nodes = int(args.num_customers + args.num_charging_stations + 1)
    action_mask_np = np.asarray(obs["action_mask"], dtype=bool).reshape(batch_size * n_traj, num_nodes)
    action_mask = torch.as_tensor(action_mask_np, dtype=torch.bool, device=base_logits.device)
    feats = build_features(
        base_logits,
        action_mask,
        args.num_customers,
        args.num_charging_stations,
        obs=obs,
        feature_mode=args.feature_mode,
        cached_embeddings=cached,
        glimpse=base_out[1],
    )
    residual = head(feats).masked_fill(~action_mask, 0.0)
    masked_base_logits = base_logits.masked_fill(~action_mask, -1e9)
    base_probs = torch.softmax(masked_base_logits, dim=-1).masked_fill(~action_mask, 0.0)
    base_max_prob = base_probs.max(dim=-1).values
    base_entropy = -(base_probs.clamp_min(1e-12).log() * base_probs).sum(dim=-1)
    gate_mode = str(args.residual_gate).lower()
    if gate_mode == "all":
        gate = torch.ones_like(base_max_prob)
    elif gate_mode == "maxprob":
        gate = (base_max_prob <= float(args.max_prob_threshold)).to(residual.dtype)
    elif gate_mode == "entropy":
        gate = (base_entropy >= float(args.entropy_threshold)).to(residual.dtype)
    elif gate_mode == "maxprob_or_entropy":
        gate = (
            (base_max_prob <= float(args.max_prob_threshold))
            | (base_entropy >= float(args.entropy_threshold))
        ).to(residual.dtype)
    else:
        raise ValueError(f"Unknown residual gate: {args.residual_gate}")
    step_limit = int(args.residual_step_limit)
    if step_limit >= 0 and int(step_idx) >= step_limit:
        gate = torch.zeros_like(gate)
    residual = residual * gate[:, None]
    stats = getattr(args, "_gate_stats", None)
    if stats is not None:
        stats["steps"] += int(gate.numel())
        stats["active"] += int(gate.sum().item())
        stats["max_prob_sum"] += float(base_max_prob.sum().item())
        stats["entropy_sum"] += float(base_entropy.sum().item())
    logits = compose_proposal_logits(
        base_logits,
        residual,
        action_mask,
        args.final_logit_mode,
    )
    if str(args.decode_mode).lower() == "greedy":
        return logits.max(dim=-1)[1].reshape(batch_size, n_traj)
    dist = torch.distributions.Categorical(logits=logits / max(float(args.temperature), 1e-6))
    return dist.sample().reshape(batch_size, n_traj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--head-path", required=True)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--cuda", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--problem", default="evrptw")
    parser.add_argument("--env-id", default="evrptw-v0")
    parser.add_argument("--env-entry-point", default="evrptw_gen.benchmarks.DRL_Solver.envs.evrp_vector_env:EVRPTWVectorEnv")
    parser.add_argument("--config-path", default=str(ROOT / "evrptw_gen/configs/config.yaml"))
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--n-encode-layers", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--num-customers", type=int, default=50)
    parser.add_argument("--num-charging-stations", type=int, default=12)
    parser.add_argument("--max-residual", type=float, default=1.0)
    parser.add_argument("--feature-mode", choices=["logit", "rich", "rich_norm", "embed"], default="logit")
    parser.add_argument(
        "--final-logit-mode",
        choices=["raw_residual", "z_residual"],
        default="raw_residual",
    )
    parser.add_argument("--n-traj", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--decode-mode", choices=["sampling", "greedy"], default="sampling")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--residual-gate",
        choices=["all", "maxprob", "entropy", "maxprob_or_entropy"],
        default="all",
    )
    parser.add_argument("--max-prob-threshold", type=float, default=0.75)
    parser.add_argument("--entropy-threshold", type=float, default=1.0)
    parser.add_argument(
        "--residual-step-limit",
        type=int,
        default=-1,
        help="If nonnegative, only apply residual on steps < residual_step_limit.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args._gate_stats = {"steps": 0, "active": 0, "max_prob_sum": 0.0, "entropy_sum": 0.0}

    try:
        gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    except Exception:
        pass
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() and args.cuda else "cpu"
    config = Config(args.config_path)
    with open(args.eval_data_path, "rb") as f:
        eval_data = pickle.load(f)
    if int(args.limit) > 0:
        eval_data = eval_data[: int(args.limit)]
    base = load_base(args, device)
    head = load_head(args, device)
    rows = []
    start = time.time()
    for batch_start in range(0, len(eval_data), int(args.eval_batch_size)):
        batch_indices = list(range(batch_start, min(batch_start + int(args.eval_batch_size), len(eval_data))))
        envs = SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    int(args.seed + idx),
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
            cached = base.backbone.encode(obs)
            batch_actions = []
            final_done = None
            for _step in range(int(args.eval_steps)):
                action = select_action(args, base, head, obs, cached, step_idx=_step)
                action_np = action.detach().cpu().numpy()
                batch_actions.append(action_np.copy())
                obs, _reward, done, _info = envs.step(action_np)
                final_done = np.asarray(done)
                if bool(np.asarray(done).all()):
                    break
            actions = np.stack(batch_actions, axis=0)
            for local_idx, data_idx in enumerate(batch_indices):
                base_env = unwrap_env(envs.envs[local_idx])
                objective = np.asarray(base_env.objective, dtype=np.float64).reshape(-1)
                done_vec = np.asarray(final_done[local_idx], dtype=bool).reshape(-1)
                valid = np.isfinite(objective) & (objective > 0)
                solved = bool(np.any(valid & done_vec))
                candidate = np.where(valid & done_vec)[0] if solved else np.where(valid)[0]
                best_traj = int(candidate[np.argmin(objective[candidate])]) if candidate.size else 0
                best_obj = float(objective[best_traj]) * 100.0 if candidate.size else float("inf")
                route = actions[:, local_idx, best_traj].astype(int).tolist() if candidate.size else []
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
                        "decode_mode": "bounded_residual",
                        "residual_gate": str(args.residual_gate),
                        "max_prob_threshold": float(args.max_prob_threshold),
                        "entropy_threshold": float(args.entropy_threshold),
                        "residual_step_limit": int(args.residual_step_limit),
                        "feature_mode": str(args.feature_mode),
                        "temperature": float(args.temperature),
                        "route": json.dumps(route),
                    }
                )
        finally:
            envs.close()
        print(f"[BoundedResidualEval] {len(rows)}/{len(eval_data)} done", flush=True)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    finite = np.asarray([r["objective_value"] for r in rows if np.isfinite(r["objective_value"])])
    summary = {
        "instances": len(rows),
        "avg_objective": float(np.mean(finite)) if finite.size else float("nan"),
        "median_objective": float(np.median(finite)) if finite.size else float("nan"),
        "solved_rate": float(np.mean([r["solved"] for r in rows])) if rows else 0.0,
        "output_csv": args.output_csv,
        "base_checkpoint_path": args.base_checkpoint_path,
        "head_path": args.head_path,
        "residual_gate": str(args.residual_gate),
        "max_prob_threshold": float(args.max_prob_threshold),
        "entropy_threshold": float(args.entropy_threshold),
        "residual_step_limit": int(args.residual_step_limit),
        "feature_mode": str(args.feature_mode),
        "final_logit_mode": str(args.final_logit_mode),
        "gate_active_rate": (
            float(args._gate_stats["active"] / args._gate_stats["steps"])
            if args._gate_stats["steps"]
            else 0.0
        ),
        "base_max_prob_mean": (
            float(args._gate_stats["max_prob_sum"] / args._gate_stats["steps"])
            if args._gate_stats["steps"]
            else float("nan")
        ),
        "base_entropy_mean": (
            float(args._gate_stats["entropy_sum"] / args._gate_stats["steps"])
            if args._gate_stats["steps"]
            else float("nan")
        ),
        "elapsed_sec": float(time.time() - start),
    }
    summary_path = args.summary_json or str(Path(args.output_csv).with_suffix(".summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[BoundedResidualEvalSummary] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
