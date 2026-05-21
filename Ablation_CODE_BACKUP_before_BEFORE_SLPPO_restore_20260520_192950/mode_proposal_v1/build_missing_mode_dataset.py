#!/usr/bin/env python3
"""Build missing-mode ALNS records for offline proposal training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evrptw_gen.benchmarks.DRL_Solver.models.graph_attention_model_wrapper import Agent
from evrptw_gen.benchmarks.DRL_Solver.train import (
    _alns_record_to_action_sequence,
    _load_alns_buffer_from_dir,
)
from evrptw_gen.benchmarks.DRL_Solver.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.DRL_Solver.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.configs.load_config import Config


def _stem(value):
    if value is None:
        return None
    return os.path.splitext(os.path.basename(str(value)))[0]


def make_env(env_id, seed, cfg):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(int(seed))
        env.action_space.seed(int(seed))
        env.observation_space.seed(int(seed))
        return env

    return thunk


def load_frontier(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = [
                row.get("file"),
                row.get("instance_id"),
                _stem(row.get("file")),
                _stem(row.get("instance_id")),
                row.get("index"),
            ]
            for key in keys:
                if key is not None and str(key) != "":
                    rows[str(key)] = row
    return rows


def match_frontier(record, frontier):
    keys = [
        record.get("file"),
        record.get("instance_id"),
        record.get("key"),
        record.get("matched_key"),
        _stem(record.get("file")),
        _stem(record.get("instance_id")),
        _stem(record.get("key")),
        _stem(record.get("matched_key")),
    ]
    for key in keys:
        if key is not None and str(key) in frontier:
            return frontier[str(key)]
    return None


def load_agent(args, device):
    agent = Agent(
        device=device,
        name=args.problem,
        tanh_clipping=args.tanh_clipping,
        n_encode_layers=args.n_encode_layers,
        value_heads=args.value_heads,
    ).to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


@torch.no_grad()
def replay_batch(args, config, agent, records, device, batch_start):
    num_nodes = int(args.num_customers + args.num_charging_stations + 1)
    sequences = [
        _alns_record_to_action_sequence(
            record,
            num_customers=int(args.num_customers),
            num_nodes=num_nodes,
        )[: int(args.max_steps)]
        for record in records
    ]
    envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                int(args.seed + 700000 + batch_start + i),
                cfg={
                    "env_mode": "eval",
                    "config": config,
                    "n_traj": 1,
                    "eval_data": deepcopy(record["instance"]),
                },
            )
            for i, record in enumerate(records)
        ]
    )
    out = []
    try:
        obs = envs.reset()
        cached = agent.backbone.encode(obs)
        n = len(records)
        invalid = np.zeros(n, dtype=bool)
        done_seen = np.zeros(n, dtype=bool)
        nll_sum = np.zeros(n, dtype=np.float64)
        rank_sum = np.zeros(n, dtype=np.float64)
        entropy_sum = np.zeros(n, dtype=np.float64)
        lowprob_sum = np.zeros(n, dtype=np.float64)
        step_count = np.zeros(n, dtype=np.int64)

        for step in range(int(args.max_steps)):
            active = np.asarray([step < len(seq) for seq in sequences], dtype=bool)
            if not active.any():
                break
            logits, _ = agent.backbone.decode(obs, cached)
            logits = agent.actor((logits, None)).squeeze(1)
            logp = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * logp).sum(dim=-1)
            action_mask = np.asarray(obs["action_mask"], dtype=bool).reshape(n, 1, num_nodes)[:, 0, :]
            action_np = np.zeros((n, 1), dtype=np.int64)

            logits_np = logits.detach().cpu().numpy()
            logp_np = logp.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            entropy_np = entropy.detach().cpu().numpy()

            for i, seq in enumerate(sequences):
                if not active[i]:
                    action_np[i, 0] = 0
                    continue
                action = int(seq[step])
                if action < 0 or action >= num_nodes or not bool(action_mask[i, action]):
                    invalid[i] = True
                    feasible_idx = np.where(action_mask[i])[0]
                    action = int(feasible_idx[np.argmax(logits_np[i, feasible_idx])]) if feasible_idx.size else 0
                    action_np[i, 0] = action
                    continue
                target_logit = logits_np[i, action]
                feasible_logits = logits_np[i, action_mask[i]]
                rank = 1 + int(np.sum(feasible_logits > target_logit + 1e-9))
                prob = float(probs_np[i, action])
                nll_sum[i] += float(-logp_np[i, action])
                rank_sum[i] += float(rank)
                entropy_sum[i] += float(entropy_np[i])
                lowprob_sum[i] += float(prob < float(args.lowprob_threshold))
                step_count[i] += 1
                action_np[i, 0] = action

            obs, _reward, done, _info = envs.step(action_np)
            done_arr = np.asarray(done, dtype=bool).reshape(n, 1)[:, 0]
            done_seen |= done_arr
        for i, record in enumerate(records):
            denom = max(int(step_count[i]), 1)
            out.append(
                {
                    "valid_replay": int((not invalid[i]) and bool(done_seen[i]) and step_count[i] > 0),
                    "mean_nll": float(nll_sum[i] / denom),
                    "mean_rank": float(rank_sum[i] / denom),
                    "mean_entropy": float(entropy_sum[i] / denom),
                    "lowprob_ratio": float(lowprob_sum[i] / denom),
                    "replay_steps": int(step_count[i]),
                    "sequence_len": int(len(sequences[i])),
                }
            )
    finally:
        envs.close()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-dir", required=True)
    parser.add_argument("--frontier-csv", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-pkl", required=True)
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
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--novelty-quantile", type=float, default=0.5)
    parser.add_argument("--novelty-threshold", type=float, default=-1.0)
    parser.add_argument("--lowprob-threshold", type=float, default=0.1)
    args = parser.parse_args()

    try:
        gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    except Exception:
        pass

    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() and args.cuda else "cpu"
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    config = Config(args.config_path)
    frontier = load_frontier(args.frontier_csv)
    records = _load_alns_buffer_from_dir(args.buffer_dir)
    agent = load_agent(args, device)

    table = []
    matched = []
    for record in records:
        frow = match_frontier(record, frontier)
        if frow is None:
            continue
        try:
            base_obj = float(frow["objective_value"])
            alns_obj = float(record["raw_teacher_obj"])
        except Exception:
            continue
        if not np.isfinite(base_obj) or not np.isfinite(alns_obj) or base_obj <= 0:
            continue
        improvement_abs = base_obj - alns_obj
        improvement_rel = improvement_abs / max(base_obj, 1e-8)
        matched.append((record, frow, base_obj, alns_obj, improvement_abs, improvement_rel))

    for start in range(0, len(matched), int(args.replay_batch_size)):
        batch = matched[start : start + int(args.replay_batch_size)]
        replay = replay_batch(args, config, agent, [x[0] for x in batch], device, start)
        for (record, frow, base_obj, alns_obj, improvement_abs, improvement_rel), rep in zip(batch, replay):
            row = {
                "file": record.get("file", ""),
                "instance_id": record.get("instance_id", ""),
                "frontier_index": frow.get("index", ""),
                "base_obj50": float(base_obj),
                "alns_obj": float(alns_obj),
                "improvement_abs": float(improvement_abs),
                "improvement_rel": float(improvement_rel),
                **rep,
            }
            table.append((record, row))
        print(f"[MissingMode] replayed {min(start + len(batch), len(matched))}/{len(matched)}", flush=True)

    positive_nll = [
        row["mean_nll"]
        for _record, row in table
        if row["valid_replay"] and row["improvement_rel"] >= float(args.margin)
    ]
    if float(args.novelty_threshold) >= 0:
        novelty_threshold = float(args.novelty_threshold)
    elif positive_nll:
        novelty_threshold = float(np.quantile(positive_nll, float(args.novelty_quantile)))
    else:
        novelty_threshold = float("inf")

    selected = []
    rows = []
    for record, row in table:
        selected_flag = (
            bool(row["valid_replay"])
            and row["improvement_rel"] >= float(args.margin)
            and row["mean_nll"] >= novelty_threshold
        )
        row["novelty_threshold"] = float(novelty_threshold)
        row["selected"] = int(selected_flag)
        row["route_weight"] = float(
            max(row["improvement_rel"] - float(args.margin), 0.0)
            * max(row["mean_nll"] - novelty_threshold + 1.0, 0.0)
        )
        rows.append(row)
        if selected_flag:
            rec = deepcopy(record)
            rec["mode_proposal"] = dict(row)
            selected.append(rec)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    out_pkl = Path(args.output_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(selected, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "buffer_dir": args.buffer_dir,
        "frontier_csv": args.frontier_csv,
        "records_loaded": len(records),
        "matched": len(matched),
        "rows": len(rows),
        "valid_replay": int(sum(r["valid_replay"] for r in rows)),
        "margin": float(args.margin),
        "novelty_threshold": float(novelty_threshold),
        "selected": len(selected),
        "selected_improvement_rel_mean": float(np.mean([r["improvement_rel"] for r in rows if r["selected"]])) if selected else float("nan"),
        "selected_mean_nll": float(np.mean([r["mean_nll"] for r in rows if r["selected"]])) if selected else float("nan"),
        "output_csv": str(out_csv),
        "output_pkl": str(out_pkl),
    }
    summary_path = Path(args.summary_json or str(out_csv).replace(".csv", ".summary.json"))
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[MissingModeSummary] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
