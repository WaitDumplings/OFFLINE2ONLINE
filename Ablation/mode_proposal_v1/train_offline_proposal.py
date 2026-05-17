#!/usr/bin/env python3
"""Train a deployable offline proposal policy from selected missing-mode routes."""

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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evrptw_gen.benchmarks.DRL_Solver.models.graph_attention_model_wrapper import Agent
from evrptw_gen.benchmarks.DRL_Solver.train import _alns_record_to_action_sequence
from evrptw_gen.benchmarks.DRL_Solver.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.DRL_Solver.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.configs.load_config import Config


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
    state = torch.load(args.base_checkpoint_path, map_location=device)
    agent.load_state_dict(state)
    return agent


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


def masked_kl(logits_new, logits_base, feasible_mask):
    logits_new = logits_new.masked_fill(~feasible_mask, -1e9)
    logits_base = logits_base.masked_fill(~feasible_mask, -1e9)
    new_logp = torch.log_softmax(logits_new, dim=-1)
    base_logp = torch.log_softmax(logits_base, dim=-1)
    new_p = torch.softmax(logits_new, dim=-1)
    kl = new_p * (new_logp - base_logp)
    kl = torch.where(feasible_mask, kl, torch.zeros_like(kl))
    denom = feasible_mask.float().sum(dim=-1).clamp_min(1.0)
    return kl.sum(dim=-1) / denom


def train_batch(args, config, proposal, base, optimizer, records, device, seed_base):
    num_nodes = int(args.num_customers + args.num_charging_stations + 1)
    sequences = [
        _alns_record_to_action_sequence(
            record,
            num_customers=int(args.num_customers),
            num_nodes=num_nodes,
        )[: int(args.max_steps)]
        for record in records
    ]
    route_weights = torch.as_tensor(
        [
            max(float(record.get("mode_proposal", {}).get("route_weight", 1.0)), 1e-3)
            for record in records
        ],
        dtype=torch.float32,
        device=device,
    )
    route_weights = route_weights / route_weights.mean().clamp_min(1e-6)

    envs = make_batch_envs(args, config, records, seed_base)
    try:
        obs = envs.reset()
        prop_cached = proposal.backbone.encode(obs)
        with torch.no_grad():
            base_cached = base.backbone.encode(obs)

        n = len(records)
        bc_terms = []
        kl_terms = []
        mask_counts = []
        invalid = 0

        for step in range(int(args.max_steps)):
            active = np.asarray([step < len(seq) for seq in sequences], dtype=bool)
            if not active.any():
                break

            prop_logits, _ = proposal.backbone.decode(obs, prop_cached)
            prop_logits = proposal.actor((prop_logits, None)).squeeze(1)
            with torch.no_grad():
                base_logits, _ = base.backbone.decode(obs, base_cached)
                base_logits = base.actor((base_logits, None)).squeeze(1)
                base_logp = torch.log_softmax(base_logits, dim=-1)
                base_prob = torch.softmax(base_logits, dim=-1)

            action_mask_np = np.asarray(obs["action_mask"], dtype=bool).reshape(n, 1, num_nodes)[:, 0, :]
            action_mask = torch.as_tensor(action_mask_np, dtype=torch.bool, device=device)
            prop_logits = prop_logits.masked_fill(~action_mask, -1e9)
            base_logits = base_logits.masked_fill(~action_mask, -1e9)
            actions_np = np.zeros((n, 1), dtype=np.int64)
            targets = torch.zeros(n, dtype=torch.long, device=device)
            active_t = torch.as_tensor(active, dtype=torch.bool, device=device)
            valid_t = active_t.clone()

            base_logits_np = base_logits.detach().cpu().numpy()
            for i, seq in enumerate(sequences):
                if not active[i]:
                    actions_np[i, 0] = 0
                    continue
                action = int(seq[step])
                if action < 0 or action >= num_nodes or not bool(action_mask_np[i, action]):
                    invalid += 1
                    valid_t[i] = False
                    feasible_idx = np.where(action_mask_np[i])[0]
                    action = int(feasible_idx[np.argmax(base_logits_np[i, feasible_idx])]) if feasible_idx.size else 0
                actions_np[i, 0] = action
                targets[i] = int(action)

            target_logp = F.log_softmax(prop_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
            with torch.no_grad():
                target_base_prob = base_prob.gather(1, targets[:, None]).squeeze(1)
                target_base_logit = base_logits.gather(1, targets[:, None]).squeeze(1)
                better_count = (base_logits > target_base_logit[:, None]) & action_mask
                target_rank = better_count.sum(dim=-1) + 1
                credit_mask = (
                    valid_t
                    & (
                        (target_base_prob <= float(args.base_prob_threshold))
                        | (target_rank >= int(args.rank_threshold))
                    )
                )
            if credit_mask.any():
                weighted_nll = -target_logp * route_weights
                bc_terms.append(weighted_nll[credit_mask].mean())
                mask_counts.append(float(credit_mask.float().mean().item()))
            kl_state = masked_kl(prop_logits, base_logits, action_mask)
            if valid_t.any():
                kl_terms.append(kl_state[valid_t].mean())

            obs, _reward, _done, _info = envs.step(actions_np)

        if bc_terms:
            bc_loss = torch.stack(bc_terms).mean()
        else:
            bc_loss = torch.zeros((), device=device)
        if kl_terms:
            kl_loss = torch.stack(kl_terms).mean()
        else:
            kl_loss = torch.zeros((), device=device)
        loss = bc_loss + float(args.kl_coef) * kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proposal.parameters(), float(args.max_grad_norm))
        optimizer.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "bc_loss": float(bc_loss.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "credit_mask_mean": float(np.mean(mask_counts)) if mask_counts else 0.0,
            "invalid_steps": int(invalid),
        }
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-pkl", required=True)
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--output-checkpoint", required=True)
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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--base-prob-threshold", type=float, default=0.25)
    parser.add_argument("--rank-threshold", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--train-scope",
        choices=["all", "decoder", "encoder_decoder"],
        default="all",
    )
    args = parser.parse_args()

    try:
        gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    except Exception:
        pass

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() and args.cuda else "cpu"
    config = Config(args.config_path)
    with open(args.records_pkl, "rb") as f:
        records = pickle.load(f)
    if len(records) == 0:
        raise ValueError(f"No selected records in {args.records_pkl}")

    proposal = load_agent(args, device)
    base = load_agent(args, device)
    proposal.train()
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    if str(args.train_scope) == "decoder":
        for p in proposal.parameters():
            p.requires_grad_(False)
        for p in proposal.backbone.decoder.parameters():
            p.requires_grad_(True)
    elif str(args.train_scope) == "encoder_decoder":
        for p in proposal.parameters():
            p.requires_grad_(False)
        for p in proposal.backbone.encoder.parameters():
            p.requires_grad_(True)
        for p in proposal.backbone.decoder.parameters():
            p.requires_grad_(True)
    elif str(args.train_scope) != "all":
        raise ValueError(f"Unknown train_scope={args.train_scope}")
    trainable_params = [p for p in proposal.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters for train_scope={args.train_scope}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    log_rows = []
    start = time.time()
    steps = 0
    rng = random.Random(int(args.seed))
    for epoch in range(1, int(args.epochs) + 1):
        order = list(records)
        rng.shuffle(order)
        epoch_rows = []
        for batch_start in range(0, len(order), int(args.batch_size)):
            batch = order[batch_start : batch_start + int(args.batch_size)]
            metrics = train_batch(
                args,
                config,
                proposal,
                base,
                optimizer,
                batch,
                device,
                seed_base=int(args.seed + 900000 + epoch * 10000 + batch_start),
            )
            steps += 1
            row = {"epoch": epoch, "step": steps, "batch_size": len(batch), **metrics}
            log_rows.append(row)
            epoch_rows.append(row)
        print(
            f"[OfflineProposalTrain] epoch={epoch}/{args.epochs} | "
            f"loss={np.mean([r['loss'] for r in epoch_rows]):.6f} | "
            f"bc={np.mean([r['bc_loss'] for r in epoch_rows]):.6f} | "
            f"kl={np.mean([r['kl_loss'] for r in epoch_rows]):.6f} | "
            f"mask={np.mean([r['credit_mask_mean'] for r in epoch_rows]):.4f} | "
            f"elapsed={time.time() - start:.1f}s",
            flush=True,
        )

    torch.save(proposal.state_dict(), args.output_checkpoint)
    with open(args.log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    summary = {
        "records": len(records),
        "epochs": int(args.epochs),
        "steps": int(steps),
        "learning_rate": float(args.learning_rate),
        "kl_coef": float(args.kl_coef),
        "train_scope": str(args.train_scope),
        "base_prob_threshold": float(args.base_prob_threshold),
        "rank_threshold": int(args.rank_threshold),
        "final_loss": float(log_rows[-1]["loss"]),
        "final_bc_loss": float(log_rows[-1]["bc_loss"]),
        "final_kl_loss": float(log_rows[-1]["kl_loss"]),
        "elapsed_sec": float(time.time() - start),
        "output_checkpoint": args.output_checkpoint,
        "log_csv": args.log_csv,
    }
    summary_path = args.summary_json or str(Path(args.log_csv).with_suffix(".summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[OfflineProposalTrainSummary] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
