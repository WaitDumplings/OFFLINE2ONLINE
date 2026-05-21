#!/usr/bin/env python3
"""Train a tiny bounded residual proposal head on top of a frozen base policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evrptw_gen.benchmarks.DRL_Solver.models.graph_attention_model_wrapper import Agent
from evrptw_gen.benchmarks.DRL_Solver.train import _alns_record_to_action_sequence
from evrptw_gen.benchmarks.DRL_Solver.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.DRL_Solver.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.configs.load_config import Config


class BoundedResidualHead(nn.Module):
    def __init__(self, in_dim=7, hidden=64, max_residual=1.5):
        super().__init__()
        self.max_residual = float(max_residual)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats):
        return self.max_residual * torch.tanh(self.net(feats).squeeze(-1))


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
    agent.load_state_dict(torch.load(args.base_checkpoint_path, map_location=device))
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent


def node_type_onehot(batch_size, num_customers, num_charging_stations, device):
    node_types = torch.zeros(1 + num_customers + num_charging_stations, dtype=torch.long, device=device)
    node_types[1 : 1 + num_customers] = 1
    node_types[1 + num_customers :] = 2
    return F.one_hot(node_types, num_classes=3).float().unsqueeze(0).expand(batch_size, -1, -1)


def _flatten_traj_tensor(x, batch_size, n_traj, device, dtype=torch.float32):
    t = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    if t.shape[0] == batch_size and t.numel() >= batch_size * n_traj:
        if t.ndim >= 2 and t.shape[1] == n_traj:
            return t.reshape(batch_size * n_traj, *t.shape[2:])
        return t.repeat_interleave(n_traj, dim=0)
    if t.shape[0] == batch_size * n_traj:
        return t
    return t.reshape(batch_size * n_traj, *t.shape[1:])


def _repeat_static_tensor(x, batch_size, n_traj, device, dtype=torch.float32):
    t = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    if t.ndim >= 1 and t.shape[0] == batch_size:
        return t.repeat_interleave(n_traj, dim=0)
    if t.ndim >= 1 and t.shape[0] == batch_size * n_traj:
        return t
    if t.ndim >= 1:
        return t.unsqueeze(0).expand(batch_size * n_traj, *t.shape)
    return t.reshape(1).expand(batch_size * n_traj)


def _rich_action_features(obs, base_logits, action_mask, num_customers, num_charging_stations):
    device = base_logits.device
    flat_batch, num_nodes = base_logits.shape
    raw_mask = np.asarray(obs["action_mask"], dtype=bool)
    batch_size = int(raw_mask.shape[0])
    n_traj = max(1, flat_batch // max(batch_size, 1))

    depot = _repeat_static_tensor(obs["depot_loc"], batch_size, n_traj, device)
    customers = _repeat_static_tensor(obs["cus_loc"], batch_size, n_traj, device)
    stations = _repeat_static_tensor(obs["rs_loc"], batch_size, n_traj, device)
    node_xy = torch.cat([depot, customers, stations], dim=1)
    demand = _repeat_static_tensor(obs["demand"], batch_size, n_traj, device)
    service = _repeat_static_tensor(obs["service_time"], batch_size, n_traj, device)
    tw = _repeat_static_tensor(obs["time_window"], batch_size, n_traj, device)
    edge_energy = _repeat_static_tensor(obs["edge_energy"], batch_size, n_traj, device)
    visit_count = _flatten_traj_tensor(obs["node_visit_count"], batch_size, n_traj, device)

    last_idx = _flatten_traj_tensor(obs["last_node_idx"], batch_size, n_traj, device, dtype=torch.long).reshape(-1)
    last_idx = last_idx.clamp(0, num_nodes - 1)
    batch_arange = torch.arange(flat_batch, device=device)
    last_xy = node_xy[batch_arange, last_idx]
    dist = torch.linalg.vector_norm(node_xy - last_xy[:, None, :], dim=-1)
    energy = edge_energy[batch_arange, last_idx, :]
    current_time = _flatten_traj_tensor(obs["current_time"], batch_size, n_traj, device).reshape(-1, 1)
    current_battery = _flatten_traj_tensor(obs["current_battery"], batch_size, n_traj, device).reshape(-1, 1)
    current_load = _flatten_traj_tensor(obs["current_load"], batch_size, n_traj, device).reshape(-1, 1)
    slack_start = tw[..., 0] - current_time - energy
    slack_end = tw[..., 1] - current_time - energy - service
    battery_after = current_battery - energy
    load_after = current_load + demand
    ratios = []
    for key in ("visited_customers_ratio", "remain_feasible_customers_ratio", "rs_streak_ratio"):
        if key in obs:
            ratios.append(_flatten_traj_tensor(obs[key], batch_size, n_traj, device).reshape(flat_batch, -1)[:, :1])
    ratio_feat = torch.cat(ratios, dim=-1) if ratios else torch.zeros(flat_batch, 0, device=device)
    ratio_feat = ratio_feat[:, None, :].expand(-1, num_nodes, -1)

    rich = torch.cat(
        [
            node_xy,
            demand.unsqueeze(-1),
            service.unsqueeze(-1),
            tw,
            dist.unsqueeze(-1),
            energy.unsqueeze(-1),
            slack_start.unsqueeze(-1),
            slack_end.unsqueeze(-1),
            battery_after.unsqueeze(-1),
            load_after.unsqueeze(-1),
            visit_count.unsqueeze(-1),
            ratio_feat,
        ],
        dim=-1,
    )
    rich = torch.where(action_mask[..., None], rich, torch.zeros_like(rich))
    return rich


def _normalize_action_features(feats, action_mask, clip=5.0):
    feasible = action_mask.unsqueeze(-1).float()
    count = feasible.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (feats * feasible).sum(dim=1, keepdim=True) / count
    var = (((feats - mean) * feasible) ** 2).sum(dim=1, keepdim=True) / count
    normed = (feats - mean) / torch.sqrt(var + 1e-6)
    normed = normed.clamp(-float(clip), float(clip))
    return torch.where(action_mask[..., None], normed, torch.zeros_like(normed))


def _embedding_action_features(base_logits, action_mask, cached_embeddings, glimpse):
    flat_batch, num_nodes = base_logits.shape
    node_embed = cached_embeddings[0]
    if node_embed.shape[0] != flat_batch:
        if flat_batch % node_embed.shape[0] != 0:
            raise ValueError(
                f"Cannot align node embeddings {node_embed.shape} with logits {base_logits.shape}"
            )
        repeat = flat_batch // node_embed.shape[0]
        node_embed = node_embed.repeat_interleave(repeat, dim=0)
    if node_embed.shape[1] != num_nodes:
        raise ValueError(
            f"Node embedding count {node_embed.shape[1]} does not match logits nodes {num_nodes}"
        )

    if glimpse.dim() == 3:
        glimpse_flat = glimpse.reshape(flat_batch, glimpse.shape[-1])
    elif glimpse.dim() == 2:
        glimpse_flat = glimpse
    else:
        raise ValueError(f"Unexpected glimpse shape: {glimpse.shape}")
    if glimpse_flat.shape[0] != flat_batch:
        if flat_batch % glimpse_flat.shape[0] != 0:
            raise ValueError(
                f"Cannot align glimpse {glimpse.shape} with logits {base_logits.shape}"
            )
        repeat = flat_batch // glimpse_flat.shape[0]
        glimpse_flat = glimpse_flat.repeat_interleave(repeat, dim=0)

    node_embed = F.layer_norm(node_embed, (node_embed.shape[-1],))
    glimpse_flat = F.layer_norm(glimpse_flat, (glimpse_flat.shape[-1],))
    glimpse_expand = glimpse_flat[:, None, :].expand(-1, num_nodes, -1)
    feats = torch.cat([node_embed, glimpse_expand], dim=-1)
    return torch.where(action_mask[..., None], feats, torch.zeros_like(feats))


def normalized_base_logits(base_logits, action_mask):
    feasible = action_mask.float()
    count = feasible.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (base_logits.masked_fill(~action_mask, 0.0) * feasible).sum(dim=-1, keepdim=True) / count
    var = (((base_logits - mean).masked_fill(~action_mask, 0.0) ** 2) * feasible).sum(dim=-1, keepdim=True) / count
    std = torch.sqrt(var + 1e-6)
    z = (base_logits - mean) / std
    return torch.where(action_mask, z, torch.zeros_like(z))


def build_features(
    base_logits,
    action_mask,
    num_customers,
    num_charging_stations,
    obs=None,
    feature_mode="logit",
    cached_embeddings=None,
    glimpse=None,
):
    logits = base_logits.masked_fill(~action_mask, -1e9)
    logp = torch.log_softmax(logits, dim=-1)
    prob = torch.softmax(logits, dim=-1)
    feasible = action_mask.float()
    count = feasible.sum(dim=-1, keepdim=True).clamp_min(1.0)
    z = normalized_base_logits(base_logits, action_mask)
    better = (base_logits[:, None, :] > base_logits[:, :, None]) & action_mask[:, None, :]
    rank = better.sum(dim=-1).float() + 1.0
    rank_norm = rank / count
    z = torch.where(action_mask, z, torch.zeros_like(z))
    logp = torch.where(action_mask, logp, torch.zeros_like(logp))
    prob = torch.where(action_mask, prob, torch.zeros_like(prob))
    rank_norm = torch.where(action_mask, rank_norm, torch.zeros_like(rank_norm))
    type_feat = node_type_onehot(base_logits.size(0), num_customers, num_charging_stations, base_logits.device)
    feats = torch.cat(
        [
            z.unsqueeze(-1),
            logp.unsqueeze(-1),
            prob.unsqueeze(-1),
            rank_norm.unsqueeze(-1),
            type_feat,
        ],
        dim=-1,
    )
    mode = str(feature_mode).lower()
    if mode in {"rich", "rich_norm"}:
        if obs is None:
            raise ValueError(f"obs is required for feature_mode={feature_mode}")
        rich = _rich_action_features(obs, base_logits, action_mask, num_customers, num_charging_stations)
        if mode == "rich_norm":
            rich = _normalize_action_features(rich, action_mask)
        feats = torch.cat([feats, rich], dim=-1)
    elif mode == "embed":
        if cached_embeddings is None or glimpse is None:
            raise ValueError("cached_embeddings and glimpse are required for feature_mode=embed")
        embed = _embedding_action_features(base_logits, action_mask, cached_embeddings, glimpse)
        feats = torch.cat([feats, embed], dim=-1)
    return feats


def masked_kl(logits_new, logits_base, action_mask):
    logits_new = logits_new.masked_fill(~action_mask, -1e9)
    logits_base = logits_base.masked_fill(~action_mask, -1e9)
    logp_new = torch.log_softmax(logits_new, dim=-1)
    logp_base = torch.log_softmax(logits_base, dim=-1)
    p_new = torch.softmax(logits_new, dim=-1)
    return (p_new * (logp_new - logp_base)).sum(dim=-1)


def compose_proposal_logits(base_logits, residual, action_mask, final_logit_mode):
    mode = str(final_logit_mode).lower()
    if mode == "raw_residual":
        logits = base_logits + residual
    elif mode == "z_residual":
        logits = normalized_base_logits(base_logits, action_mask) + residual
    else:
        raise ValueError(f"Unknown final_logit_mode: {final_logit_mode}")
    return logits.masked_fill(~action_mask, -1e9)


def make_batch_envs(args, config, records, seed_base):
    return SyncVectorEnv(
        [
            make_env(
                args.env_id,
                int(seed_base + i),
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


def train_batch(args, config, base, head, optimizer, records, device, seed_base):
    num_nodes = int(args.num_customers + args.num_charging_stations + 1)
    sequences = [
        _alns_record_to_action_sequence(record, args.num_customers, num_nodes)[: int(args.max_steps)]
        for record in records
    ]
    route_weights = torch.as_tensor(
        [max(float(r.get("mode_proposal", {}).get("route_weight", 1.0)), 1e-3) for r in records],
        dtype=torch.float32,
        device=device,
    )
    route_weights = route_weights / route_weights.mean().clamp_min(1e-6)
    envs = make_batch_envs(args, config, records, seed_base)
    try:
        obs = envs.reset()
        cached = base.backbone.encode(obs)
        n = len(records)
        bc_terms = []
        kl_terms = []
        res_terms = []
        mask_rates = []
        invalid_steps = 0
        for step in range(int(args.max_steps)):
            active = np.asarray([step < len(seq) for seq in sequences], dtype=bool)
            if not active.any():
                break
            with torch.no_grad():
                base_out = base.backbone.decode(obs, cached)
                base_logits = base.actor(base_out).squeeze(1)
            action_mask_np = np.asarray(obs["action_mask"], dtype=bool).reshape(n, 1, num_nodes)[:, 0, :]
            action_mask = torch.as_tensor(action_mask_np, dtype=torch.bool, device=device)
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
            final_logits = compose_proposal_logits(
                base_logits,
                residual,
                action_mask,
                args.final_logit_mode,
            )

            actions_np = np.zeros((n, 1), dtype=np.int64)
            targets = torch.zeros(n, dtype=torch.long, device=device)
            valid = torch.as_tensor(active, dtype=torch.bool, device=device)
            base_logits_np = base_logits.detach().cpu().numpy()
            for i, seq in enumerate(sequences):
                if not active[i]:
                    actions_np[i, 0] = 0
                    continue
                action = int(seq[step])
                if action < 0 or action >= num_nodes or not bool(action_mask_np[i, action]):
                    invalid_steps += 1
                    valid[i] = False
                    feasible_idx = np.where(action_mask_np[i])[0]
                    action = int(feasible_idx[np.argmax(base_logits_np[i, feasible_idx])]) if feasible_idx.size else 0
                targets[i] = action
                actions_np[i, 0] = action

            logp_final = torch.log_softmax(final_logits, dim=-1)
            target_logp = logp_final.gather(1, targets[:, None]).squeeze(1)
            with torch.no_grad():
                base_prob = torch.softmax(base_logits.masked_fill(~action_mask, -1e9), dim=-1)
                target_prob = base_prob.gather(1, targets[:, None]).squeeze(1)
                target_logit = base_logits.gather(1, targets[:, None]).squeeze(1)
                target_rank = (((base_logits > target_logit[:, None]) & action_mask).sum(dim=-1) + 1)
                credit = valid & (
                    (target_prob <= float(args.base_prob_threshold))
                    | (target_rank >= int(args.rank_threshold))
                )
            train_step_limit = int(args.train_step_limit)
            train_this_step = train_step_limit < 0 or int(step) < train_step_limit
            if train_this_step and credit.any():
                bc_terms.append((-target_logp * route_weights)[credit].mean())
                mask_rates.append(float(credit.float().mean().item()))
            if train_this_step and valid.any():
                kl_terms.append(masked_kl(final_logits, base_logits, action_mask)[valid].mean())
                res_terms.append((residual[valid] ** 2).mean())

            obs, _reward, _done, _info = envs.step(actions_np)

        bc_loss = torch.stack(bc_terms).mean() if bc_terms else torch.zeros((), device=device)
        kl_loss = torch.stack(kl_terms).mean() if kl_terms else torch.zeros((), device=device)
        res_loss = torch.stack(res_terms).mean() if res_terms else torch.zeros((), device=device)
        loss = bc_loss + float(args.kl_coef) * kl_loss + float(args.res_coef) * res_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), float(args.max_grad_norm))
        optimizer.step()
        return {
            "loss": float(loss.detach().cpu().item()),
            "bc_loss": float(bc_loss.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "res_loss": float(res_loss.detach().cpu().item()),
            "credit_mask_mean": float(np.mean(mask_rates)) if mask_rates else 0.0,
            "invalid_steps": int(invalid_steps),
        }
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-pkl", required=True)
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--output-head", required=True)
    parser.add_argument("--log-csv", required=True)
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
    parser.add_argument(
        "--train-step-limit",
        type=int,
        default=-1,
        help="If nonnegative, only apply training losses on replay steps < train_step_limit.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-coef", type=float, default=0.2)
    parser.add_argument("--res-coef", type=float, default=0.01)
    parser.add_argument("--max-residual", type=float, default=1.0)
    parser.add_argument(
        "--final-logit-mode",
        choices=["raw_residual", "z_residual"],
        default="raw_residual",
        help="How the proposal branch composes base logits and learned residual.",
    )
    parser.add_argument("--feature-mode", choices=["logit", "rich", "rich_norm", "embed"], default="logit")
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--base-prob-threshold", type=float, default=0.15)
    parser.add_argument("--rank-threshold", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    try:
        gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    except Exception:
        pass
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() and args.cuda else "cpu"
    config = Config(args.config_path)
    with open(args.records_pkl, "rb") as f:
        records = pickle.load(f)
    base = load_agent(args, device)
    if str(args.feature_mode) == "logit":
        feature_dim = 7
    elif str(args.feature_mode) in {"rich", "rich_norm"}:
        feature_dim = 23
    elif str(args.feature_mode) == "embed":
        feature_dim = 7 + 2 * int(base.backbone.decoder.embedding_dim)
    else:
        raise ValueError(f"Unknown feature_mode={args.feature_mode}")
    head = BoundedResidualHead(
        in_dim=feature_dim,
        hidden=int(args.head_hidden),
        max_residual=float(args.max_residual),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.learning_rate))
    log_rows = []
    start = time.time()
    rng = random.Random(int(args.seed))
    step_id = 0
    for epoch in range(1, int(args.epochs) + 1):
        order = list(records)
        rng.shuffle(order)
        epoch_rows = []
        for batch_start in range(0, len(order), int(args.batch_size)):
            batch = order[batch_start : batch_start + int(args.batch_size)]
            metrics = train_batch(
                args,
                config,
                base,
                head,
                optimizer,
                batch,
                device,
                seed_base=int(args.seed + 950000 + epoch * 10000 + batch_start),
            )
            step_id += 1
            row = {"epoch": epoch, "step": step_id, "batch_size": len(batch), **metrics}
            log_rows.append(row)
            epoch_rows.append(row)
        print(
            f"[BoundedResidualTrain] epoch={epoch}/{args.epochs} | "
            f"loss={np.mean([r['loss'] for r in epoch_rows]):.6f} | "
            f"bc={np.mean([r['bc_loss'] for r in epoch_rows]):.6f} | "
            f"kl={np.mean([r['kl_loss'] for r in epoch_rows]):.6f} | "
            f"res={np.mean([r['res_loss'] for r in epoch_rows]):.6f} | "
            f"mask={np.mean([r['credit_mask_mean'] for r in epoch_rows]):.4f}",
            flush=True,
        )

    Path(args.output_head).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "max_residual": float(args.max_residual),
            "num_customers": int(args.num_customers),
            "num_charging_stations": int(args.num_charging_stations),
            "feature_mode": str(args.feature_mode),
            "in_dim": int(feature_dim),
            "hidden": int(args.head_hidden),
            "final_logit_mode": str(args.final_logit_mode),
        },
        args.output_head,
    )
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    summary = {
        "records": len(records),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "kl_coef": float(args.kl_coef),
        "res_coef": float(args.res_coef),
        "max_residual": float(args.max_residual),
        "feature_mode": str(args.feature_mode),
        "final_logit_mode": str(args.final_logit_mode),
        "in_dim": int(feature_dim),
        "hidden": int(args.head_hidden),
        "train_step_limit": int(args.train_step_limit),
        "final_loss": float(log_rows[-1]["loss"]),
        "elapsed_sec": float(time.time() - start),
        "output_head": args.output_head,
        "log_csv": args.log_csv,
    }
    summary_path = args.summary_json or str(Path(args.log_csv).with_suffix(".summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[BoundedResidualTrainSummary] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
