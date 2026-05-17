import argparse
import os
import pickle
import random
import time
from copy import deepcopy
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from evrptw_gen.configs.load_config import Config
from evrptw_gen.benchmarks.DRL_Solver.models.competence_agent import (
    CompetenceAgent,
)
from evrptw_gen.benchmarks.DRL_Solver.train import (
    _alns_record_to_action_sequence,
    _extract_objectives_from_info,
    _load_alns_buffer_from_dir,
    _run_evaluation,
    _scheduled_train_configs_for_batch,
)
from evrptw_gen.benchmarks.DRL_Solver.utils.utils import (
    grad_norm,
    make_env,
    update_lambda_fail,
)
from evrptw_gen.benchmarks.DRL_Solver.wrappers.syncVectorEnvPomo import (
    SyncVectorEnv,
)


def str2bool(x):
    if isinstance(x, bool):
        return x
    return bool(strtobool(str(x)))


def _set_global_seeds(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_state_dict_compatible(module, state_dict):
    current = module.state_dict()
    compatible = {
        k: v
        for k, v in state_dict.items()
        if k in current and tuple(v.shape) == tuple(current[k].shape)
    }
    skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
    current.update(compatible)
    module.load_state_dict(current)
    if skipped:
        print(f"[Init] skipped incompatible checkpoint tensors: {len(skipped)}")


def _unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _stack_obs_list(obs_list):
    if not obs_list:
        raise ValueError("obs_list is empty.")
    return {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}


def _clean_env_cfg(args, lambda_fail):
    return {
        "reward_mode": "vanilla",
        "max_route_events": int(args.max_route_events),
        "max_completed_routes": int(args.max_completed_routes),
        "use_teacher_reward": False,
        "teacher_reward_mode": "none",
        "use_direct_progress_pbrs": bool(args.use_direct_progress_pbrs),
        "progress_pbrs_coef": float(args.progress_pbrs_coef),
        "progress_pbrs_beta": float(args.progress_pbrs_beta),
        "use_repair_fail_reward": bool(args.use_repair_fail_reward),
        "repair_fail_coef": float(args.repair_fail_coef),
        "repair_progress_coef": float(args.repair_progress_coef),
        "repair_success_bonus": float(args.repair_success_bonus),
        "lambda_fail": float(lambda_fail),
    }


def _sample_offline_records(records, batch_size, rng, args):
    if batch_size <= 0 or not records:
        return []

    weights = np.ones(len(records), dtype=np.float64)
    kappa = float(args.offline_regret_kappa)
    for idx, record in enumerate(records):
        regret = float(record.get("student_regret", 0.0))
        if np.isfinite(regret) and regret > 0.0:
            weights[idx] += kappa * regret

    weights = weights / weights.sum()
    replace = batch_size > len(records)
    indices = rng.choice(len(records), size=batch_size, replace=replace, p=weights)
    return [records[int(i)] for i in indices]


def _prepare_teacher_action_sequence(record, args):
    seq = record.get("teacher_action_sequence", None)
    if seq is None:
        seq = _alns_record_to_action_sequence(
            record,
            num_customers=int(args.train_cus_num),
            num_nodes=int(args.train_cus_num + args.train_cs_num + 1),
        )
        record["teacher_action_sequence"] = list(seq)
    return list(seq)


def _make_mixed_train_envs(
    args,
    config,
    perturb_dict,
    offline_records,
    num_offline,
    lambda_fail,
    update_step,
    rng,
):
    sampled_offline = _sample_offline_records(
        offline_records,
        batch_size=num_offline,
        rng=rng,
        args=args,
    )
    num_online = int(args.num_envs) - len(sampled_offline)

    env_fns = []
    records_meta = []

    clean_cfg = _clean_env_cfg(args, lambda_fail)

    for i, record in enumerate(sampled_offline):
        instance = deepcopy(record["instance"])
        env_fns.append(
            make_env(
                args.env_id,
                int(args.seed + 200000 + update_step * 1000 + i),
                cfg={
                    "env_mode": "train",
                    "config": config,
                    "n_traj": args.n_traj,
                    "eval_data": instance,
                    "num_customers": args.train_cus_num,
                    "num_charging_stations": args.train_cs_num,
                    "gamma": args.gamma,
                    **clean_cfg,
                },
            )
        )
        records_meta.append(record)

    if num_online > 0:
        if bool(args.train_config_use_online_counter):
            schedule_step = int(getattr(args, "_train_config_online_step", 0))
        else:
            schedule_step = int(update_step)

        train_configs = _scheduled_train_configs_for_batch(
            args,
            config,
            schedule_step,
            batch_size=num_online,
        )
        if bool(args.train_config_use_online_counter):
            args._train_config_online_step = schedule_step + 1

        seed_offset = update_step * int(args.num_envs) if args.train_env_seed_by_update else 0
        for j in range(num_online):
            env_fns.append(
                make_env(
                    args.env_id,
                    int(args.seed + seed_offset + 500000 + j),
                    cfg={
                        "env_mode": "train",
                        "config": train_configs[j],
                        "n_traj": args.n_traj,
                        "num_customers": args.train_cus_num,
                        "num_charging_stations": args.train_cs_num,
                        "gamma": args.gamma,
                        "perturb_dict": perturb_dict["perturb"],
                        **clean_cfg,
                    },
                )
            )
            records_meta.append(None)

    return SyncVectorEnv(env_fns), records_meta


def _choose_prefix_len(seq_len, args, rng):
    if seq_len <= 1:
        return 0
    remaining_min = float(
        getattr(args, "_active_expert_prefix_remaining_min", args.expert_prefix_remaining_min)
    )
    remaining_max = float(
        getattr(args, "_active_expert_prefix_remaining_max", args.expert_prefix_remaining_max)
    )
    remaining = rng.uniform(remaining_min, remaining_max)
    prefix_len = int(round(float(seq_len) * (1.0 - remaining)))
    return int(np.clip(prefix_len, 0, seq_len - 1))


def _apply_prefix_curriculum(args, update_step):
    if not bool(getattr(args, "use_prefix_curriculum", False)):
        args._active_expert_prefix_prob = float(args.expert_prefix_prob)
        args._active_expert_prefix_remaining_min = float(args.expert_prefix_remaining_min)
        args._active_expert_prefix_remaining_max = float(args.expert_prefix_remaining_max)
        args._active_prefix_stage = "fixed"
        return

    frac = float(update_step) / float(max(int(args.num_updates), 1))
    base_prob = float(args.expert_prefix_prob)
    if frac < 0.25:
        stage, prob, lo, hi = "A_tail", base_prob, 0.10, 0.30
    elif frac < 0.50:
        stage, prob, lo, hi = "B_mid", base_prob * 0.85, 0.20, 0.60
    elif frac < 0.75:
        stage, prob, lo, hi = "C_large", base_prob * 0.65, 0.40, 0.90
    else:
        stage, prob, lo, hi = "D_full", base_prob * 0.25, 0.60, 1.00

    args._active_expert_prefix_prob = float(np.clip(prob, 0.0, 1.0))
    args._active_expert_prefix_remaining_min = lo
    args._active_expert_prefix_remaining_max = hi
    args._active_prefix_stage = stage


def _reset_envs_for_rollout(envs, records_meta, args, rng):
    obs_list = []
    prefix_objective = []
    teacher_suffix_obj = []
    teacher_obj = []
    is_offline = []
    prefix_len = []

    for env, record in zip(envs.envs, records_meta):
        base_env = _unwrap_env(env)
        # Mark Gym wrappers as reset before any direct base-env prefix replay.
        obs = env.reset()
        use_prefix = (
            record is not None
            and bool(args.use_expert_prefix)
            and rng.random() < float(getattr(args, "_active_expert_prefix_prob", args.expert_prefix_prob))
        )

        if use_prefix:
            seq = _prepare_teacher_action_sequence(record, args)
            m = _choose_prefix_len(len(seq), args, rng)
            obs = base_env.reset_with_teacher_prefix(seq, m)

        obs_list.append(obs)
        prefix_objective.append(
            np.asarray(getattr(base_env, "prefix_objective", np.zeros(args.n_traj)), dtype=np.float32)
        )
        teacher_suffix_obj.append(
            np.asarray(getattr(base_env, "teacher_suffix_obj", np.full(args.n_traj, np.nan)), dtype=np.float32)
        )
        teacher_obj.append(float(record["teacher_obj"]) if record is not None else np.nan)
        is_offline.append(record is not None)
        prefix_len.append(int(getattr(base_env, "prefix_len", 0)))

    meta = {
        "prefix_objective": torch.tensor(
            np.stack(prefix_objective, axis=0),
            dtype=torch.float32,
        ),
        "teacher_suffix_obj": torch.tensor(
            np.stack(teacher_suffix_obj, axis=0),
            dtype=torch.float32,
        ),
        "teacher_obj": torch.tensor(teacher_obj, dtype=torch.float32),
        "is_offline": torch.tensor(is_offline, dtype=torch.bool),
        "prefix_len": torch.tensor(prefix_len, dtype=torch.long),
    }
    return _stack_obs_list(obs_list), meta


def _rollout_action_value_cached(agent, obs, encoder_state, args):
    backbone_output = agent.backbone.decode(obs, encoder_state)
    logits = agent.actor(backbone_output)
    temperature = max(float(getattr(args, "rollout_temperature", 1.0)), 1e-6)
    behavior_logits = logits / temperature
    probs = torch.distributions.Categorical(logits=behavior_logits)
    action = probs.sample()
    value = agent.critic(backbone_output)
    return action, probs.log_prob(action), probs.entropy(), value


def _student_rollout(
    agent,
    envs,
    args,
    device,
    num_steps,
    initial_obs,
    rollout_meta=None,
):
    num_envs = len(envs.envs)
    obs_buf = [None] * num_steps

    actions = torch.zeros(
        (num_steps, num_envs) + envs.single_action_space.shape,
        dtype=torch.int16,
        device=device,
    )
    logprobs = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    rewards = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    objectives = torch.full((num_envs, args.n_traj), float("inf"), device=device)
    dones = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    values = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    valid_masks = torch.zeros(
        (num_steps, num_envs, args.n_traj),
        dtype=torch.bool,
        device=device,
    )

    agent.train()
    next_obs = initial_obs
    with torch.no_grad():
        encoder_state = agent.backbone.encode(next_obs)

    next_done = torch.zeros(num_envs, args.n_traj, device=device)
    alive = torch.ones(num_envs, args.n_traj, dtype=torch.bool, device=device)
    valid_step = 0
    last_info = None

    for step in range(num_steps):
        obs_buf[step] = next_obs
        dones[step] = next_done
        valid_masks[step] = alive

        with torch.no_grad():
            action, logprob, _, value = _rollout_action_value_cached(
                agent,
                next_obs,
                encoder_state,
                args,
            )
            action = action.view(num_envs, args.n_traj)
            values[step] = value.squeeze(-1).view(num_envs, args.n_traj)

        actions[step] = action
        logprobs[step] = logprob.view(num_envs, args.n_traj)

        if step == num_steps - 1:
            for env in envs.envs:
                setattr(_unwrap_env(env), "terminate", True)

        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        last_info = info

        rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32)
        done_tensor = torch.tensor(done, device=device, dtype=torch.bool)

        maybe_obj = _extract_objectives_from_info(info, args.n_traj, device)
        if maybe_obj is not None:
            objectives = maybe_obj

        next_done = done_tensor.float()
        alive = alive & (~done_tensor)
        valid_step = step + 1

        if done.all():
            break

    result = {
        "obs": obs_buf[:valid_step],
        "actions": actions[:valid_step],
        "logprobs": logprobs[:valid_step],
        "rewards": rewards[:valid_step],
        "objectives": objectives,
        "dones": dones[:valid_step],
        "values": values[:valid_step],
        "valid_masks": valid_masks[:valid_step],
        "next_obs": next_obs,
        "next_done": next_done,
        "encoder_state": encoder_state,
        "valid_step": valid_step,
        "last_info": last_info,
    }
    if rollout_meta:
        result.update({k: v.to(device) for k, v in rollout_meta.items()})
    return result

def _instance_normalize_advantages(advantages, valid_masks):
    out = torch.zeros_like(advantages)
    for env_idx in range(advantages.shape[1]):
        mask = valid_masks[:, env_idx, :].bool()
        vals = advantages[:, env_idx, :][mask]
        if vals.numel() <= 1:
            continue
        std = vals.std()
        if torch.isfinite(std) and std > 1e-8:
            out[:, env_idx, :] = (advantages[:, env_idx, :] - vals.mean()) / (std + 1e-8)
    return out * valid_masks


def _trajectory_rank_advantage(objectives, clip_value):
    rank = torch.zeros_like(objectives)
    for env_idx in range(objectives.shape[0]):
        obj = objectives[env_idx]
        valid = torch.isfinite(obj)
        if valid.sum() <= 1:
            continue
        vals = obj[valid]
        std = vals.std()
        if torch.isfinite(std) and std > 1e-8:
            rank[env_idx, valid] = (vals.mean() - vals) / (std + 1e-8)
    return rank.clamp(-clip_value, clip_value)


def _comparative_suffix_advantage(rollout, clip_value):
    objectives = rollout["objectives"]
    prefix_objective = rollout.get("prefix_objective", None)
    teacher_suffix_obj = rollout.get("teacher_suffix_obj", None)
    is_offline = rollout.get("is_offline", None)

    if prefix_objective is None or teacher_suffix_obj is None or is_offline is None:
        return torch.zeros_like(objectives)

    student_suffix = objectives - prefix_objective
    denom = teacher_suffix_obj.abs().clamp_min(1e-8)
    cmp_adv = (teacher_suffix_obj - student_suffix) / denom

    valid = (
        is_offline[:, None]
        & torch.isfinite(student_suffix)
        & torch.isfinite(teacher_suffix_obj)
        & (teacher_suffix_obj > 0.0)
    )
    cmp_adv = torch.where(valid, cmp_adv, torch.zeros_like(cmp_adv))
    return cmp_adv.clamp(-clip_value, clip_value)


def _compute_gae_and_returns(agent, rollout, args, device):
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    valid_masks = rollout["valid_masks"].bool()
    valid_step = int(rollout["valid_step"])

    with torch.no_grad():
        next_value = agent.get_value_cached(
            rollout["next_obs"],
            rollout["encoder_state"],
        ).squeeze(-1)

    advantages = torch.zeros_like(rewards, device=device)
    lastgaelam = torch.zeros(rewards.shape[1], rewards.shape[2], device=device)

    for t in reversed(range(valid_step)):
        if t == valid_step - 1:
            nextnonterminal = 1.0 - rollout["next_done"]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]

        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta
            + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )

    returns = advantages + values

    if args.adv_norm_scope == "instance":
        actor_advantages = _instance_normalize_advantages(advantages, valid_masks)
    elif args.norm_adv:
        actor_advantages = advantages.clone()
        vals = actor_advantages[valid_masks]
        if vals.numel() > 1:
            actor_advantages = (actor_advantages - vals.mean()) / (vals.std() + 1e-8)
            actor_advantages = actor_advantages * valid_masks
    else:
        actor_advantages = advantages

    if args.rank_adv_coef > 0:
        rank = _trajectory_rank_advantage(
            rollout["objectives"],
            float(args.rank_adv_clip),
        )
        actor_advantages = actor_advantages + float(args.rank_adv_coef) * rank.unsqueeze(0)

    if args.cmp_adv_coef > 0:
        cmp_adv = _comparative_suffix_advantage(
            rollout,
            float(args.cmp_adv_clip),
        )
        actor_advantages = actor_advantages + float(args.cmp_adv_coef) * cmp_adv.unsqueeze(0)

    return actor_advantages * valid_masks, returns


def _ppo_update(agent, optim_backbone, optim_critic, envs, rollout, advantages, returns, args):
    obs = rollout["obs"]
    actions = rollout["actions"]
    old_logprobs = rollout["logprobs"]
    old_values = rollout["values"]
    valid_masks = rollout["valid_masks"].bool()
    valid_step = int(rollout["valid_step"])
    num_envs = len(envs.envs)

    batch_size = int(num_envs * valid_step)
    envsperbatch = num_envs // int(args.num_minibatches)
    flatinds = np.arange(batch_size).reshape(valid_step, num_envs)
    envinds = np.arange(num_envs)

    b_obs = {
        k: np.concatenate([obs_t[k] for obs_t in obs], axis=0)
        for k in envs.single_observation_space
    }
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_old_logprobs = old_logprobs.reshape(-1, args.n_traj)
    b_advantages = advantages.reshape(-1, args.n_traj).detach()
    b_returns = returns.reshape(-1, args.n_traj)
    b_old_values = old_values.reshape(-1, args.n_traj)
    b_valid = valid_masks.reshape(-1, args.n_traj)

    pg_losses = []
    v_losses = []
    entropies = []
    kls = []

    accum_steps = max(1, int(args.accum_steps))
    stop_early = False

    for _epoch in range(args.update_epochs):
        if stop_early:
            break
        np.random.shuffle(envinds)
        optim_backbone.zero_grad(set_to_none=True)
        optim_critic.zero_grad(set_to_none=True)
        accum_counter = 0

        for start in range(0, num_envs, envsperbatch):
            mbenvinds = envinds[start : start + envsperbatch]
            mb_inds = flatinds[:, mbenvinds].ravel()
            r_inds = np.tile(np.arange(envsperbatch), valid_step)

            cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}
            encoder_state_mb = agent.backbone.encode(cur_obs)

            _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value_cached(
                {k: v[mb_inds] for k, v in b_obs.items()},
                b_actions.long()[mb_inds],
                (embedding[r_inds, :] for embedding in encoder_state_mb),
            )

            mb_valid = b_valid[mb_inds]
            valid_count = mb_valid.sum()
            if valid_count == 0:
                continue
            valid_count_f = valid_count.float()

            logratio = newlogprob - b_old_logprobs[mb_inds]
            ratio = logratio.exp()
            with torch.no_grad():
                lr = logratio[mb_valid]
                rr = ratio[mb_valid]
                approx_kl = ((rr - 1.0) - lr).mean().detach()
                kls.append(float(approx_kl.cpu().item()))

            mb_adv = b_advantages[mb_inds]
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(
                ratio,
                1.0 - args.clip_coef,
                1.0 + args.clip_coef,
            )
            pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_valid).sum() / valid_count_f

            newvalue = newvalue.squeeze(-1).view(-1, args.n_traj)
            returns_mb = b_returns[mb_inds]
            old_values_mb = b_old_values[mb_inds]
            value_loss_raw = F.smooth_l1_loss(
                newvalue,
                returns_mb,
                reduction="none",
                beta=1.0,
            )
            if args.clip_vloss:
                v_clipped = old_values_mb + torch.clamp(
                    newvalue - old_values_mb,
                    -args.clip_coef,
                    args.clip_coef,
                )
                value_loss_clip = F.smooth_l1_loss(
                    v_clipped,
                    returns_mb,
                    reduction="none",
                    beta=1.0,
                )
                v_loss = 0.5 * (torch.max(value_loss_raw, value_loss_clip) * mb_valid).sum() / valid_count_f
            else:
                v_loss = 0.5 * (value_loss_raw * mb_valid).sum() / valid_count_f

            entropy_loss = (entropy * mb_valid).sum() / valid_count_f
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
            (loss / accum_steps).backward()
            accum_counter += 1

            if accum_counter % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    agent.backbone.parameters(),
                    args.max_grad_norm_backbone,
                )
                torch.nn.utils.clip_grad_norm_(
                    agent.critic.parameters(),
                    args.max_grad_norm_critic,
                )
                optim_backbone.step()
                optim_critic.step()
                optim_backbone.zero_grad(set_to_none=True)
                optim_critic.zero_grad(set_to_none=True)

            pg_losses.append(float(pg_loss.detach().cpu().item()))
            v_losses.append(float(v_loss.detach().cpu().item()))
            entropies.append(float(entropy_loss.detach().cpu().item()))

        if accum_counter % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                agent.backbone.parameters(),
                args.max_grad_norm_backbone,
            )
            torch.nn.utils.clip_grad_norm_(
                agent.critic.parameters(),
                args.max_grad_norm_critic,
            )
            optim_backbone.step()
            optim_critic.step()

        if kls and float(np.mean(kls[-args.num_minibatches :])) > args.target_kl:
            stop_early = True

    return {
        "pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
        "v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
        "grad_backbone": grad_norm(agent.backbone.parameters()),
        "grad_critic": grad_norm(agent.critic.parameters()),
    }


def _best_objectives(rollout):
    objectives = rollout["objectives"]
    success = torch.isfinite(objectives)
    return objectives.min(dim=1).values.detach().cpu().numpy(), success.float().mean(dim=1).detach().cpu().numpy()



def _best_action_sequences_from_rollout(rollout):
    objectives = rollout["objectives"].detach().cpu()
    actions = rollout["actions"].detach().cpu().numpy()
    valid_masks = rollout["valid_masks"].detach().cpu().numpy().astype(np.bool_)
    if objectives.numel() == 0 or actions.size == 0:
        return []
    best_traj = objectives.argmin(dim=1).numpy().astype(np.int64)
    sequences = []
    for env_idx, traj_idx in enumerate(best_traj):
        seq = []
        for step in range(actions.shape[0]):
            if not bool(valid_masks[step, env_idx, traj_idx]):
                break
            seq.append(int(actions[step, env_idx, traj_idx]))
        sequences.append(seq)
    return sequences


def _fill_infeasible_with_first_valid(obs, actions_np):
    mask = np.asarray(obs["action_mask"], dtype=bool)
    for i in range(actions_np.shape[0]):
        for j in range(actions_np.shape[1]):
            a = int(actions_np[i, j])
            if a < 0 or a >= mask.shape[-1] or not mask[i, j, a]:
                valid = np.flatnonzero(mask[i, j])
                actions_np[i, j] = int(valid[0]) if valid.size else 0
    return actions_np


def _offline_supervised_warmup(agent, optim_backbone, optim_critic, args, config, records, device, rng):
    updates = int(getattr(args, "stage0_bc_updates", 0))
    if updates <= 0 or not records:
        return

    print(f"[Stage0] offline supervised warmup updates={updates}")
    old_use_prefix = bool(args.use_expert_prefix)
    old_prefix_prob = float(args.expert_prefix_prob)
    args.use_expert_prefix = False
    args.expert_prefix_prob = 0.0

    try:
        for update in range(updates):
            batch_size = min(int(args.stage0_bc_batch_size), len(records))
            sampled = _sample_offline_records(records, batch_size, rng, args)
            envs = None
            old_num_envs = int(args.num_envs)
            args.num_envs = batch_size
            try:
                envs, records_meta = _make_mixed_train_envs(
                    args=args,
                    config=config,
                    perturb_dict={"perturb": {}},
                    offline_records=sampled,
                    num_offline=batch_size,
                    lambda_fail=args.lambda_fail_init,
                    update_step=800000 + update,
                    rng=rng,
                )
                obs, _meta = _reset_envs_for_rollout(envs, records_meta, args, rng)
                seqs = [_prepare_teacher_action_sequence(r, args) for r in records_meta]
                max_len = min(int(args.stage0_bc_max_steps), max((len(s) for s in seqs), default=0))
                bc_losses = []
                critic_losses = []

                for step in range(max_len):
                    targets = np.zeros((batch_size, args.n_traj), dtype=np.int64)
                    valid = np.zeros((batch_size, args.n_traj), dtype=np.bool_)
                    value_targets = np.zeros((batch_size, args.n_traj), dtype=np.float32)
                    for env_idx, (env, record, seq) in enumerate(zip(envs.envs, records_meta, seqs)):
                        base_env = _unwrap_env(env)
                        if step < len(seq):
                            a = int(seq[step])
                            if 0 <= a < base_env.n_nodes:
                                targets[env_idx, :] = a
                                valid[env_idx, :] = np.asarray(obs["action_mask"][env_idx, :, a], dtype=np.bool_)
                        teacher_obj = float(record["teacher_obj"])
                        remaining = teacher_obj - np.asarray(base_env.objective, dtype=np.float32)
                        value_targets[env_idx, :] = -remaining.astype(np.float32)

                    if not valid.any():
                        break

                    targets = _fill_infeasible_with_first_valid(obs, targets)
                    target_tensor = torch.as_tensor(targets, device=device, dtype=torch.long)
                    valid_tensor = torch.as_tensor(valid, device=device, dtype=torch.bool)
                    value_target_tensor = torch.as_tensor(value_targets, device=device, dtype=torch.float32)

                    _, logprob, _, value = agent.get_action_and_value(obs, action=target_tensor)
                    valid_count = valid_tensor.sum().float().clamp_min(1.0)
                    bc_loss = -(logprob * valid_tensor).sum() / valid_count
                    value_pred = value.squeeze(-1)
                    critic_loss = (
                        F.smooth_l1_loss(value_pred, value_target_tensor, reduction="none", beta=1.0)
                        * valid_tensor
                    ).sum() / valid_count
                    loss = float(args.stage0_bc_coef) * bc_loss + float(args.stage0_critic_coef) * critic_loss

                    optim_backbone.zero_grad(set_to_none=True)
                    optim_critic.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.backbone.parameters(), args.max_grad_norm_backbone)
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm_critic)
                    optim_backbone.step()
                    optim_critic.step()
                    bc_losses.append(float(bc_loss.detach().cpu().item()))
                    critic_losses.append(float(critic_loss.detach().cpu().item()))

                    obs, _reward, done, _info = envs.step(targets)
                    if np.asarray(done).all():
                        break

                if update % max(1, int(args.stage0_log_freq)) == 0:
                    print(
                        f"[Stage0] update={update} "
                        f"bc={float(np.mean(bc_losses)) if bc_losses else 0.0:.5f} "
                        f"critic={float(np.mean(critic_losses)) if critic_losses else 0.0:.5f}"
                    )
            finally:
                args.num_envs = old_num_envs
                if envs is not None:
                    envs.close()
    finally:
        args.use_expert_prefix = old_use_prefix
        args.expert_prefix_prob = old_prefix_prob

def _update_records_from_rollout(records_meta, rollout, args):
    best_obj, _fr = _best_objectives(rollout)
    best_sequences = _best_action_sequences_from_rollout(rollout)
    for idx, (record, obj) in enumerate(zip(records_meta, best_obj)):
        if record is None:
            continue
        teacher_obj = float(record["teacher_obj"])
        student_obj = float(obj)
        record["student_best_obj"] = student_obj
        record["student_regret"] = student_obj - teacher_obj if np.isfinite(student_obj) else teacher_obj
        record["quality_solved"] = bool(
            np.isfinite(student_obj)
            and student_obj <= (1.0 + float(args.competence_gap_eps)) * teacher_obj
        )
        if (
            bool(getattr(args, "use_self_improved_buffer", False))
            and np.isfinite(student_obj)
            and student_obj < (1.0 - float(args.self_improve_margin)) * teacher_obj
        ):
            best_seen = float(record.get("self_improved_obj", float("inf")))
            if student_obj < best_seen:
                record["self_improved_obj"] = student_obj
                if idx < len(best_sequences):
                    record["self_improved_action_sequence"] = list(best_sequences[idx])


def _probe_competence(agent, args, config, records, device, update_step, rng):
    if not records:
        return 0.0

    probe_size = min(int(args.competence_probe_size), len(records))
    sampled = _sample_offline_records(records, probe_size, rng, args)
    envs = None
    old_use_prefix = bool(args.use_expert_prefix)
    old_prefix_prob = float(args.expert_prefix_prob)
    args.use_expert_prefix = False
    args.expert_prefix_prob = 0.0
    try:
        envs, records_meta = _make_mixed_train_envs(
            args=args,
            config=config,
            perturb_dict={"perturb": {}},
            offline_records=sampled,
            num_offline=probe_size,
            lambda_fail=args.lambda_fail_init,
            update_step=update_step + 900000,
            rng=rng,
        )
        initial_obs, meta = _reset_envs_for_rollout(envs, records_meta, args, rng)
        rollout = _student_rollout(
            agent=agent,
            envs=envs,
            args=args,
            device=device,
            num_steps=int(args.competence_probe_steps),
            initial_obs=initial_obs,
            rollout_meta=meta,
        )
        _update_records_from_rollout(records_meta, rollout, args)
    finally:
        args.use_expert_prefix = old_use_prefix
        args.expert_prefix_prob = old_prefix_prob
        if envs is not None:
            envs.close()

    solved = [
        bool(r.get("quality_solved", False))
        for r in sampled
    ]
    return float(np.mean(solved)) if solved else 0.0


def _success_rate_from_rollout(rollout, customer_numbers):
    actions = rollout["actions"]
    valid_step = rollout["valid_step"]
    visu_actions = actions.reshape((valid_step, -1)).cpu().numpy().copy()
    visu_actions[visu_actions == 0] = customer_numbers + 1
    visu_actions[visu_actions < 1 + customer_numbers] = 1
    visu_actions[visu_actions >= 1 + customer_numbers] = 0
    cus_count_per_traj = visu_actions.sum(axis=0)
    return float((cus_count_per_traj == customer_numbers).mean())


def _print_update_summary(update_step, args, rollout, ppo_info, q_ema, num_offline, lambda_fail):
    best_obj, fr = _best_objectives(rollout)
    finite_obj = best_obj[np.isfinite(best_obj)]
    obj_mean = float(finite_obj.mean()) if finite_obj.size else float("nan")
    prefix_len = rollout.get("prefix_len", torch.zeros(1))
    prefix_mean = float(prefix_len.float().mean().detach().cpu().item())
    print(
        f"[Train] update={update_step} "
        f"offline={num_offline}/{args.num_envs} "
        f"q_ema={q_ema:.4f} "
        f"success={_success_rate_from_rollout(rollout, args.train_cus_num):.4f} "
        f"fr={float(fr.mean()):.4f} "
        f"obj_best_mean={obj_mean:.4f} "
        f"prefix_len_mean={prefix_mean:.2f} "
        f"prefix_stage={getattr(args, '_active_prefix_stage', 'fixed')} "
        f"lambda_fail={lambda_fail:.3f} "
        f"pg={ppo_info['pg_loss']:.5f} "
        f"vf={ppo_info['v_loss']:.5f} "
        f"kl={ppo_info['kl']:.5f}"
    )


def train(args):
    print("---------------- Competence Training Info ----------------")
    for key, value in vars(args).items():
        print(key, value)
    print("----------------------------------------------------------")

    _set_global_seeds(args.seed, deterministic=args.torch_deterministic)

    try:
        gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    except Exception as exc:
        print(f"[GymRegister] skipped or already registered: {exc}")

    device = f"cuda:{args.cuda_id}" if args.cuda and torch.cuda.is_available() else "cpu"
    agent = CompetenceAgent(
        device=device,
        name=args.problem,
        tanh_clipping=args.tanh_clipping,
        n_encode_layers=args.n_encode_layers,
        value_heads=1,
        use_candidate_dynamic_embedding=args.use_candidate_dynamic_embedding,
    ).to(device)

    if args.init_ckpt_path and os.path.exists(args.init_ckpt_path):
        ckpt = _safe_torch_load(args.init_ckpt_path, device)
        _load_state_dict_compatible(agent, ckpt)
        print(f"[Init] loaded compatible tensors from {args.init_ckpt_path}")

    optim_backbone = optim.AdamW(
        list(agent.backbone.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=args.weight_decay,
    )
    optim_critic = optim.AdamW(
        list(agent.critic.parameters()),
        lr=args.critic_lr,
        eps=1e-5,
        weight_decay=args.weight_decay,
    )

    config = Config(args.config_path)
    perturb_dict = Config(args.perturb_dict_path).setup_env_parameters()

    records = []
    if args.use_offline_buffer:
        records = _load_alns_buffer_from_dir(
            buffer_dir=args.alns_buffer_dir,
            progress_name=args.alns_progress_name,
            instance_pickle_name=args.alns_instance_pickle_name,
            obj_scale=args.alns_obj_scale,
        )
    if args.use_offline_buffer and not records:
        raise ValueError("use_offline_buffer=True but no offline ALNS records were loaded.")

    with open(args.eval_data_path, "rb") as f:
        eval_data = pickle.load(f)
    if args.eval_batch_size > len(eval_data):
        raise ValueError(
            f"eval_batch_size={args.eval_batch_size} > len(eval_data)={len(eval_data)}"
        )

    eval_rng = np.random.default_rng(args.seed + 12345)
    eval_ids = eval_rng.choice(len(eval_data), size=args.eval_batch_size, replace=False)
    test_envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                int(args.seed + 800000 + i),
                cfg={
                    "env_mode": "eval",
                    "config": config,
                    "n_traj": args.test_agent,
                    "eval_data": eval_data[int(env_id)],
                    "max_route_events": args.max_route_events,
                    "max_completed_routes": args.max_completed_routes,
                },
            )
            for i, env_id in enumerate(eval_ids)
        ]
    )

    save_dir = os.path.join(
        args.save_dir,
        args.exp_name,
        f"Cus_{args.train_cus_num}_CS_{args.train_cs_num}",
    )
    os.makedirs(save_dir, exist_ok=True)
    best_reward = float("-inf")

    rng = np.random.default_rng(args.seed + 9999)
    q_ema = 0.0
    lambda_fail = float(args.lambda_fail_init)

    _offline_supervised_warmup(
        agent=agent,
        optim_backbone=optim_backbone,
        optim_critic=optim_critic,
        args=args,
        config=config,
        records=records,
        device=device,
        rng=rng,
    )

    for update_step in tqdm(range(args.num_updates)):
        _apply_prefix_curriculum(args, update_step)
        if records and update_step % int(args.competence_probe_freq) == 0:
            q = _probe_competence(
                agent=agent,
                args=args,
                config=config,
                records=records,
                device=device,
                update_step=update_step,
                rng=rng,
            )
            beta = float(args.competence_ema_beta)
            q_ema = beta * q_ema + (1.0 - beta) * q
            print(f"[Competence] update={update_step} q={q:.4f} q_ema={q_ema:.4f}")

        if records:
            rho_online = min(1.0 - float(args.offline_min_ratio), q_ema)
            num_online = int(round(args.num_envs * rho_online))
            num_online = int(np.clip(num_online, 0, args.num_envs))
            num_offline = args.num_envs - num_online
            if args.offline_min_ratio > 0:
                floor = int(round(args.num_envs * args.offline_min_ratio))
                num_offline = max(num_offline, floor)
                num_online = args.num_envs - num_offline
        else:
            num_offline = 0

        envs = None
        try:
            envs, records_meta = _make_mixed_train_envs(
                args=args,
                config=config,
                perturb_dict=perturb_dict,
                offline_records=records,
                num_offline=num_offline,
                lambda_fail=lambda_fail,
                update_step=update_step,
                rng=rng,
            )

            initial_obs, meta = _reset_envs_for_rollout(envs, records_meta, args, rng)
            rollout = _student_rollout(
                agent=agent,
                envs=envs,
                args=args,
                device=device,
                num_steps=args.num_steps,
                initial_obs=initial_obs,
                rollout_meta=meta,
            )
            _update_records_from_rollout(records_meta, rollout, args)

            success_rate = _success_rate_from_rollout(rollout, args.train_cus_num)
            lambda_fail = update_lambda_fail(
                lambda_fail=lambda_fail,
                success_rate=success_rate,
                target_success=args.target_success,
                lambda_max=args.lambda_max,
                lr_up=args.lambda_lr_up,
                lr_down=args.lambda_lr_down,
                tolerance=args.lambda_tolerance,
            )

            advantages, returns = _compute_gae_and_returns(agent, rollout, args, device)
            ppo_info = _ppo_update(
                agent=agent,
                optim_backbone=optim_backbone,
                optim_critic=optim_critic,
                envs=envs,
                rollout=rollout,
                advantages=advantages,
                returns=returns,
                args=args,
            )

            if args.debug or update_step % args.log_freq == 0:
                _print_update_summary(
                    update_step,
                    args,
                    rollout,
                    ppo_info,
                    q_ema,
                    num_offline,
                    lambda_fail,
                )

            if (update_step + 1) % args.eval_freq == 0:
                best_reward = _run_evaluation(
                    agent=agent,
                    test_envs=test_envs,
                    args=args,
                    test_num_cus=args.train_cus_num,
                    test_num_cs=args.train_cs_num,
                    batch_size=args.eval_batch_size,
                    test_max_step=args.num_steps,
                    save_dir=save_dir,
                    best_reward=best_reward,
                    decode_mode=args.eval_decode_mode,
                    eval_label=args.eval_decode_mode,
                    num_agents=args.test_agent,
                    update_best=True,
                )

            if args.save_freq > 0 and (update_step + 1) % args.save_freq == 0:
                torch.save(agent.state_dict(), os.path.join(save_dir, "cur_model.pth"))

            milestones = {
                int(x.strip())
                for x in str(args.checkpoint_milestones).split(",")
                if x.strip()
            }
            if update_step + 1 in milestones:
                torch.save(
                    agent.state_dict(),
                    os.path.join(save_dir, f"model_update{update_step + 1:04d}.pth"),
                )
        finally:
            if envs is not None:
                envs.close()

    torch.save(agent.state_dict(), os.path.join(save_dir, "cur_model.pth"))
    test_envs.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Competence-guided offline-to-online PPO for EVRPTW."
    )

    parser.add_argument("--exp-name", type=str, default="competence_off2on")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--cuda", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--torch-deterministic", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--save-dir", type=str, default="./checkpoint")
    parser.add_argument("--init-ckpt-path", type=str, default=None)

    parser.add_argument("--problem", type=str, default="evrptw")
    parser.add_argument("--env-id", type=str, default="evrptw-competence-v0")
    parser.add_argument(
        "--env-entry-point",
        type=str,
        default="evrptw_gen.benchmarks.DRL_Solver.envs.evrp_vector_env_competence:CompetenceEVRPTWVectorEnv",
    )
    parser.add_argument("--train-cus-num", type=int, default=50)
    parser.add_argument("--train-cs-num", type=int, default=12)
    parser.add_argument("--config-path", type=str, default="./evrptw_gen/configs/config.yaml")
    parser.add_argument("--perturb-dict-path", type=str, default="./evrptw_gen/configs/perturb_config.yaml")
    parser.add_argument(
        "--eval-data-path",
        type=str,
        default="./dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl",
    )
    parser.add_argument("--eval-freq", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-decode-mode", type=str, default="sampling", choices=["greedy", "sampling"])

    parser.add_argument("--num-updates", type=int, default=800)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=75)
    parser.add_argument("--n-traj", type=int, default=50)
    parser.add_argument("--test-agent", type=int, default=8)
    parser.add_argument("--num-minibatches", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--accum-steps", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--norm-adv", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--adv-norm-scope", type=str, default="instance", choices=["batch", "instance", "none"])
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm-backbone", type=float, default=2.0)
    parser.add_argument("--max-grad-norm-critic", type=float, default=10.0)
    parser.add_argument("--target-kl", type=float, default=0.01)

    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--n-encode-layers", type=int, default=2)
    parser.add_argument("--use-candidate-dynamic-embedding", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--max-route-events", type=int, default=16)
    parser.add_argument("--max-completed-routes", type=int, default=4)

    parser.add_argument("--use-direct-progress-pbrs", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--progress-pbrs-coef", type=float, default=2.0)
    parser.add_argument("--progress-pbrs-beta", type=float, default=0.5)
    parser.add_argument("--use-repair-fail-reward", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--repair-fail-coef", type=float, default=1.0)
    parser.add_argument("--repair-progress-coef", type=float, default=1.0)
    parser.add_argument("--repair-success-bonus", type=float, default=1.0)

    parser.add_argument("--lambda-fail-init", type=float, default=10.0)
    parser.add_argument("--target-success", type=float, default=0.95)
    parser.add_argument("--lambda-max", type=float, default=50.0)
    parser.add_argument("--lambda-lr-up", type=float, default=1.0)
    parser.add_argument("--lambda-lr-down", type=float, default=2.0)
    parser.add_argument("--lambda-tolerance", type=float, default=0.01)

    parser.add_argument("--use-offline-buffer", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--alns-buffer-dir", type=str, default="./dataset/unanchored/Cus_50/buffer")
    parser.add_argument("--alns-progress-name", type=str, default="buffer_progress.pkl")
    parser.add_argument("--alns-instance-pickle-name", type=str, default="evrptw_50C_12R.pkl")
    parser.add_argument("--alns-obj-scale", type=float, default=100.0)
    parser.add_argument("--competence-probe-freq", type=int, default=10)
    parser.add_argument("--competence-probe-size", type=int, default=128)
    parser.add_argument("--competence-probe-steps", type=int, default=75)
    parser.add_argument("--competence-gap-eps", type=float, default=0.02)
    parser.add_argument("--competence-ema-beta", type=float, default=0.9)
    parser.add_argument("--offline-min-ratio", type=float, default=0.10)
    parser.add_argument("--offline-regret-kappa", type=float, default=2.0)

    parser.add_argument("--use-expert-prefix", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--expert-prefix-prob", type=float, default=0.0)
    parser.add_argument("--expert-prefix-remaining-min", type=float, default=0.10)
    parser.add_argument("--expert-prefix-remaining-max", type=float, default=0.30)

    parser.add_argument("--rank-adv-coef", type=float, default=0.05)
    parser.add_argument("--rank-adv-clip", type=float, default=3.0)
    parser.add_argument("--cmp-adv-coef", type=float, default=0.05)
    parser.add_argument("--cmp-adv-clip", type=float, default=3.0)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--use-prefix-curriculum", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--stage0-bc-updates", type=int, default=0)
    parser.add_argument("--stage0-bc-batch-size", type=int, default=32)
    parser.add_argument("--stage0-bc-max-steps", type=int, default=75)
    parser.add_argument("--stage0-bc-coef", type=float, default=0.1)
    parser.add_argument("--stage0-critic-coef", type=float, default=0.05)
    parser.add_argument("--stage0-log-freq", type=int, default=5)
    parser.add_argument("--use-self-improved-buffer", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--self-improve-margin", type=float, default=0.0)

    parser.add_argument("--train-config-schedule", type=str, default="batch_cycle", choices=["mixed", "cycle", "random", "batch_cycle", "batch_random"])
    parser.add_argument("--train-config-stratify-keys", type=str, default="instance_type,time_window_policy")
    parser.add_argument("--train-config-cycle-offset", type=int, default=0)
    parser.add_argument("--train-config-fixed-overrides", type=str, default="service_time_policy=cargoweight,cluster_number_policy=random")
    parser.add_argument("--train-config-use-online-counter", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--train-config-shuffle-combos", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--train-config-shuffle-seed", type=int, default=17)
    parser.add_argument("--train-config-log", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--train-config-log-freq", type=int, default=20)
    parser.add_argument("--train-env-seed-by-update", type=str2bool, default=True, nargs="?", const=True)

    parser.add_argument("--save-freq", type=int, default=20)
    parser.add_argument("--checkpoint-milestones", type=str, default="300,500,800")
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--debug-test", type=str2bool, default=False, nargs="?", const=True)

    args = parser.parse_args()

    if args.num_envs % args.num_minibatches != 0:
        raise ValueError("num_envs must be divisible by num_minibatches.")
    if args.num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive.")
    if args.offline_min_ratio < 0 or args.offline_min_ratio >= 1:
        raise ValueError("offline_min_ratio must be in [0, 1).")
    if args.competence_probe_freq <= 0:
        raise ValueError("competence_probe_freq must be positive.")
    if args.competence_ema_beta < 0 or args.competence_ema_beta >= 1:
        raise ValueError("competence_ema_beta must be in [0, 1).")
    if args.expert_prefix_remaining_min < 0 or args.expert_prefix_remaining_max > 1:
        raise ValueError("expert prefix remaining ratios must be within [0, 1].")
    if args.expert_prefix_remaining_min > args.expert_prefix_remaining_max:
        raise ValueError("expert_prefix_remaining_min cannot exceed max.")
    if args.rollout_temperature <= 0:
        raise ValueError("rollout_temperature must be positive.")
    if args.stage0_bc_updates < 0:
        raise ValueError("stage0_bc_updates must be non-negative.")
    if args.stage0_bc_batch_size <= 0:
        raise ValueError("stage0_bc_batch_size must be positive.")
    if args.stage0_bc_max_steps <= 0:
        raise ValueError("stage0_bc_max_steps must be positive.")
    if args.stage0_bc_coef < 0 or args.stage0_critic_coef < 0:
        raise ValueError("stage0 coefficients must be non-negative.")
    if args.self_improve_margin < 0:
        raise ValueError("self_improve_margin must be non-negative.")

    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
