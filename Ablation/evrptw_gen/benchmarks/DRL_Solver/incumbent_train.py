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
    _decomposed_component_names,
    _decomposed_component_weights,
    _extract_objectives_from_info,
    _extract_reward_component_stack,
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


def _incumbent_obj(record):
    value = record.get("incumbent_obj", record.get("teacher_obj", float("inf")))
    try:
        return float(value)
    except Exception:
        return float("inf")


def _incumbent_action_sequence(record, args):
    seq = record.get("incumbent_action_sequence", None)
    if seq is None:
        seq = record.get("self_improved_action_sequence", None)
    if seq is None:
        seq = record.get("teacher_action_sequence", None)
    if seq is None:
        seq = _alns_record_to_action_sequence(
            record,
            num_customers=int(args.train_cus_num),
            num_nodes=int(args.train_cus_num + args.train_cs_num + 1),
        )
    seq = [int(a) for a in list(seq)]
    record["incumbent_action_sequence"] = list(seq)
    return seq


def _classify_policy_vs_incumbent(record, objectives, args):
    obj = np.asarray(objectives, dtype=np.float64).reshape(-1)
    archive = _incumbent_obj(record)
    margin = float(getattr(args, "pomo50_regret_margin_abs", 0.0))
    if np.isfinite(archive):
        margin += float(getattr(args, "pomo50_regret_margin_rel", 0.0)) * abs(float(archive))

    finite = obj[np.isfinite(obj)]
    if finite.size == 0 or not np.isfinite(archive):
        best = float("inf")
        mean = float("inf")
        beat_rate = 0.0
        gap = float("inf")
        state = "stable_alns_win"
    else:
        best = float(np.min(finite))
        mean = float(np.mean(finite))
        beat_rate = float(np.mean(obj < (archive - margin)))
        gap = best - archive
        if gap < -margin:
            if beat_rate >= float(getattr(args, "pomo50_stable_beat_rate", 0.20)):
                state = "stable_ppo_win"
            else:
                state = "lucky_ppo_win"
        elif gap > margin:
            if beat_rate <= float(getattr(args, "pomo50_low_beat_rate", 0.05)):
                state = "stable_alns_win"
            else:
                state = "uncertain"
        else:
            state = "uncertain"

    record["policy_best_obj"] = best
    record["policy_mean_obj"] = mean
    record["policy_beat_rate"] = beat_rate
    record["policy_gap_to_archive"] = gap
    record["regret_state"] = state
    record["student_best_obj"] = best
    record["student_regret"] = gap
    record["student_regret_known"] = True
    return state


def _reference_allowed_for_record(record, args):
    if record is None:
        return False
    if not bool(getattr(args, "reference_adv_alns_win_only", False)):
        return True
    return str(record.get("regret_state", "unknown")) == "stable_alns_win"


def _initialize_incumbent_records(records, args):
    for record in records:
        teacher_obj = float(record.get("teacher_obj", float("inf")))
        incumbent_obj = float(record.get("incumbent_obj", teacher_obj))
        record["incumbent_obj"] = incumbent_obj
        if "incumbent_action_sequence" not in record:
            record["incumbent_action_sequence"] = _incumbent_action_sequence(record, args)
        has_student_best = "student_best_obj" in record and np.isfinite(float(record.get("student_best_obj", float("inf"))))
        record.setdefault("student_regret", float(record.get("student_best_obj", incumbent_obj)) - incumbent_obj if has_student_best else 0.0)
        record.setdefault("student_regret_known", bool(has_student_best))
        record.setdefault("buffer_source", "alns")


def _sample_incumbent_records(records, batch_size, rng, args):
    if batch_size <= 0 or not records:
        return []

    weights = np.ones(len(records), dtype=np.float64)
    kappa = float(args.buffer_regret_kappa)
    for idx, record in enumerate(records):
        if bool(getattr(args, "use_regret_aware_sampler", True)):
            known = bool(record.get("student_regret_known", False))
            regret = float(record.get("student_regret", 0.0))
            margin = float(getattr(args, "buffer_regret_margin", 0.0))
            if not known or not np.isfinite(regret):
                weights[idx] = float(getattr(args, "buffer_unknown_weight", 1.0))
                continue

            if bool(getattr(args, "use_pomo50_regret_state", False)):
                state = str(record.get("regret_state", "unknown"))
                if state == "stable_ppo_win":
                    weights[idx] = float(getattr(args, "buffer_ppo_win_weight", 0.05))
                    continue
                if state == "lucky_ppo_win":
                    weights[idx] = float(getattr(args, "buffer_lucky_ppo_weight", 0.75))
                    continue
                if state == "uncertain":
                    weights[idx] = float(getattr(args, "buffer_uncertain_weight", 0.5))
                    continue

            if regret < -margin:
                weights[idx] = float(getattr(args, "buffer_ppo_win_weight", 0.05))
                continue
            if abs(regret) <= margin:
                weights[idx] = float(getattr(args, "buffer_uncertain_weight", 0.5))
                continue

            scale = 1.0
            if bool(getattr(args, "buffer_regret_relative", False)):
                scale = max(abs(_incumbent_obj(record)), 1e-8)
            regret_term = max(0.0, regret / scale)
            weights[idx] = float(getattr(args, "buffer_alns_win_base_weight", 1.0)) + kappa * regret_term
            continue

        regret = float(record.get("student_regret", 0.0))
        if np.isfinite(regret) and regret > 0.0:
            weights[idx] += kappa * regret
    weights = weights / weights.sum()
    replace = batch_size > len(records)
    indices = rng.choice(len(records), size=batch_size, replace=replace, p=weights)
    return [records[int(i)] for i in indices]


def _split_group_counts(total, fracs):
    total = int(total)
    names = list(fracs.keys())
    raw = np.asarray([max(0.0, float(fracs[n])) for n in names], dtype=np.float64)
    if raw.sum() <= 0.0:
        raw[:] = 1.0
    raw = raw / raw.sum() * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def _prepare_teacher_action_sequence(record, args):
    return _incumbent_action_sequence(record, args)


def _make_incumbent_train_envs(
    args,
    config,
    perturb_dict,
    incumbent_records,
    lambda_fail,
    update_step,
    rng,
):
    """Build strict 50/50 rollout slots.

    The rollout slots are fixed as:
      buffer: normal/temp/prefix groups
      online: normal/temp groups
    All actions collected from those slots are generated by the current policy;
    expert actions are used only for reset/prefix and auxiliary targets.
    """
    buffer_slots = int(round(int(args.num_envs) * float(args.incumbent_ratio)))
    buffer_slots = int(np.clip(buffer_slots, 0, int(args.num_envs)))
    online_slots = int(args.num_envs) - buffer_slots

    buffer_counts = _split_group_counts(
        buffer_slots,
        {
            "buffer_normal": float(args.buffer_normal_frac),
            "buffer_temp": float(args.buffer_temp_frac),
            "buffer_prefix": float(args.buffer_prefix_frac),
        },
    )
    online_counts = _split_group_counts(
        online_slots,
        {
            "online_normal": float(args.online_normal_frac),
            "online_temp": float(args.online_temp_frac),
        },
    )

    env_fns = []
    records_meta = []
    group_meta = []
    clean_cfg = _clean_env_cfg(args, lambda_fail)

    group_specs = [
        ("buffer_normal", buffer_counts.get("buffer_normal", 0), 1.0, False),
        ("buffer_temp", buffer_counts.get("buffer_temp", 0), float(args.temperature_sampling), False),
        ("buffer_prefix", buffer_counts.get("buffer_prefix", 0), 1.0, True),
    ]
    slot_idx = 0
    for group_name, count, temperature, use_prefix in group_specs:
        sampled = _sample_incumbent_records(incumbent_records, count, rng, args)
        for record in sampled:
            instance = deepcopy(record["instance"])
            env_fns.append(
                make_env(
                    args.env_id,
                    int(args.seed + 200000 + update_step * 10000 + slot_idx),
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
            group_meta.append({"kind": group_name, "temperature": float(temperature), "use_prefix": bool(use_prefix)})
            slot_idx += 1

    if online_slots > 0:
        if bool(args.train_config_use_online_counter):
            schedule_step = int(getattr(args, "_train_config_online_step", 0))
        else:
            schedule_step = int(update_step)
        train_configs = _scheduled_train_configs_for_batch(
            args,
            config,
            schedule_step,
            batch_size=online_slots,
        )
        if bool(args.train_config_use_online_counter):
            args._train_config_online_step = schedule_step + 1

        online_specs = []
        online_specs.extend([("online_normal", 1.0)] * online_counts.get("online_normal", 0))
        online_specs.extend([("online_temp", float(args.temperature_sampling))] * online_counts.get("online_temp", 0))
        seed_offset = update_step * int(args.num_envs) if args.train_env_seed_by_update else 0
        for j, (group_name, temperature) in enumerate(online_specs):
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
            group_meta.append({"kind": group_name, "temperature": float(temperature), "use_prefix": False})

    if len(env_fns) != int(args.num_envs):
        raise RuntimeError(f"Expected {args.num_envs} rollout slots, got {len(env_fns)}.")

    return SyncVectorEnv(env_fns), records_meta, group_meta


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


def _reset_envs_for_rollout(envs, records_meta, group_meta, args, rng):
    obs_list = []
    prefix_objective = []
    teacher_suffix_obj = []
    teacher_obj = []
    reference_obj = []
    is_buffer = []
    is_prefix = []
    prefix_len = []
    temperatures = []
    reference_allowed = []

    for env, record, meta_i in zip(envs.envs, records_meta, group_meta):
        base_env = _unwrap_env(env)
        obs = env.reset()
        used_prefix = False

        if record is not None and bool(meta_i.get("use_prefix", False)):
            seq = _incumbent_action_sequence(record, args)
            m = _choose_prefix_len(len(seq), args, rng)
            obs = base_env.reset_with_teacher_prefix(seq, m)
            used_prefix = m > 0

        obs_list.append(obs)
        prefix_objective.append(
            np.asarray(getattr(base_env, "prefix_objective", np.zeros(args.n_traj)), dtype=np.float32)
        )
        teacher_suffix_obj.append(
            np.asarray(getattr(base_env, "teacher_suffix_obj", np.full(args.n_traj, np.nan)), dtype=np.float32)
        )
        teacher_obj.append(float(record.get("teacher_obj", _incumbent_obj(record))) if record is not None else np.nan)
        reference_obj.append(_incumbent_obj(record) if record is not None else np.nan)
        is_buffer.append(record is not None)
        is_prefix.append(used_prefix)
        prefix_len.append(int(getattr(base_env, "prefix_len", 0)))
        temperatures.append(float(meta_i.get("temperature", 1.0)))
        reference_allowed.append(_reference_allowed_for_record(record, args))

    meta = {
        "prefix_objective": torch.tensor(np.stack(prefix_objective, axis=0), dtype=torch.float32),
        "teacher_suffix_obj": torch.tensor(np.stack(teacher_suffix_obj, axis=0), dtype=torch.float32),
        "teacher_obj": torch.tensor(teacher_obj, dtype=torch.float32),
        "reference_obj": torch.tensor(reference_obj, dtype=torch.float32),
        "is_offline": torch.tensor(is_buffer, dtype=torch.bool),
        "is_buffer": torch.tensor(is_buffer, dtype=torch.bool),
        "is_prefix": torch.tensor(is_prefix, dtype=torch.bool),
        "prefix_len": torch.tensor(prefix_len, dtype=torch.long),
        "temperature_by_env": torch.tensor(temperatures, dtype=torch.float32),
        "reference_allowed": torch.tensor(reference_allowed, dtype=torch.bool),
    }
    return _stack_obs_list(obs_list), meta


def _rollout_action_value_cached(agent, obs, encoder_state, temperature_by_env, args):
    backbone_output = agent.backbone.decode(obs, encoder_state)
    logits = agent.actor(backbone_output)
    num_envs = int(temperature_by_env.numel())
    if logits.dim() == 3:
        temperature = temperature_by_env.to(logits.device, dtype=logits.dtype).view(num_envs, 1, 1)
    else:
        temperature = temperature_by_env.to(logits.device, dtype=logits.dtype).view(num_envs, 1)
        temperature = temperature.repeat_interleave(int(args.n_traj), dim=0)
    behavior_logits = logits / temperature.clamp_min(1e-6)
    probs = torch.distributions.Categorical(logits=behavior_logits)
    action = probs.sample()
    value = agent.critic(backbone_output)
    return action, probs.log_prob(action), probs.entropy(), value


def _extract_rollout_reward_components(info, reward, args, device):
    names = _decomposed_component_names(args)
    out = {}
    reward_np = np.asarray(reward, dtype=np.float32)
    objective = _extract_reward_component_stack(info, "objective")
    if objective is None:
        objective = np.zeros_like(reward_np)

    for name in names:
        if name == "objective":
            stack = objective
        elif name == "progress" and tuple(names) == ("objective", "progress"):
            # Two-head mode: head-0 is distance objective, head-1 is all non-objective
            # shaping/terminal/repair reward so actor still sees the full scalar reward.
            stack = reward_np - objective
        else:
            stack = _extract_reward_component_stack(info, name)
            if stack is None:
                stack = np.zeros_like(reward_np)
        out[name] = torch.tensor(stack, device=device, dtype=torch.float32)
    return out


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
    value_heads = int(getattr(agent, "value_heads", 1))

    actions = torch.zeros(
        (num_steps, num_envs) + envs.single_action_space.shape,
        dtype=torch.int16,
        device=device,
    )
    logprobs = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    rewards = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    reward_components = {
        name: torch.zeros((num_steps, num_envs, args.n_traj), device=device)
        for name in _decomposed_component_names(args)
    } if value_heads > 1 else None
    objectives = torch.full((num_envs, args.n_traj), float("inf"), device=device)
    dones = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    if value_heads > 1:
        values = torch.zeros((num_steps, num_envs, args.n_traj, value_heads), device=device)
    else:
        values = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    valid_masks = torch.zeros((num_steps, num_envs, args.n_traj), dtype=torch.bool, device=device)

    if rollout_meta is None:
        rollout_meta = {}
    temperature_by_env = rollout_meta.get("temperature_by_env", torch.ones(num_envs, dtype=torch.float32)).to(device)

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
                temperature_by_env,
                args,
            )
            action = action.view(num_envs, args.n_traj)
            if value_heads > 1:
                values[step] = value.view(num_envs, args.n_traj, value_heads)
            else:
                values[step] = value.squeeze(-1).view(num_envs, args.n_traj)

        actions[step] = action
        logprobs[step] = logprob.view(num_envs, args.n_traj)

        if step == num_steps - 1:
            for env in envs.envs:
                setattr(_unwrap_env(env), "terminate", True)

        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        last_info = info

        reward_tensor = torch.tensor(reward, device=device, dtype=torch.float32)
        rewards[step] = reward_tensor
        if reward_components is not None:
            comp = _extract_rollout_reward_components(info, reward, args, device)
            for name in reward_components:
                reward_components[name][step] = comp[name]

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
    if reward_components is not None:
        result["reward_components"] = {k: v[:valid_step] for k, v in reward_components.items()}
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


def _trajectory_rank_advantage(objectives, clip_value, mode="zscore", std_floor=0.0, top_frac=0.25):
    rank = torch.zeros_like(objectives)
    mode = str(mode).lower()
    top_frac = min(max(float(top_frac), 0.0), 1.0)
    std_floor = max(float(std_floor), 0.0)
    for env_idx in range(objectives.shape[0]):
        obj = objectives[env_idx]
        valid = torch.isfinite(obj)
        if valid.sum() <= 1:
            continue
        vals = obj[valid]
        if mode == "rank":
            order = torch.argsort(vals)
            scores = torch.zeros_like(vals)
            n = int(vals.numel())
            if n == 1:
                scores[order[0]] = 0.0
            else:
                # lower objective is better: best=+1, worst=-1
                positions = torch.arange(n, device=vals.device, dtype=vals.dtype)
                scores[order] = 1.0 - 2.0 * positions / float(n - 1)
            rank[env_idx, valid] = scores
        elif mode == "top_bottom":
            n = int(vals.numel())
            k = max(1, int(np.ceil(top_frac * n)))
            order = torch.argsort(vals)
            scores = torch.zeros_like(vals)
            scores[order[:k]] = 1.0
            scores[order[-k:]] = -1.0
            rank[env_idx, valid] = scores
        else:
            std = vals.std(unbiased=False)
            denom = torch.clamp(std, min=std_floor)
            if torch.isfinite(denom) and denom > 1e-8:
                rank[env_idx, valid] = (vals.mean() - vals) / (denom + 1e-8)
    return rank.clamp(-clip_value, clip_value)


def _decision_fork_gate(rollout, args, valid_masks):
    if not bool(getattr(args, "use_decision_fork_gate", False)):
        return torch.ones_like(valid_masks, dtype=torch.float32)
    obs_seq = rollout.get("obs", [])
    if not obs_seq or "action_mask" not in obs_seq[0]:
        return torch.ones_like(valid_masks, dtype=torch.float32)
    gates = []
    min_actions = float(getattr(args, "decision_fork_min_actions", 4))
    max_actions = float(getattr(args, "decision_fork_max_actions", 12))
    span = max(max_actions - min_actions, 1e-8)
    mode = str(getattr(args, "decision_fork_gate_mode", "soft")).lower()
    device = valid_masks.device
    for obs in obs_seq[: valid_masks.shape[0]]:
        mask = torch.as_tensor(obs["action_mask"], device=device, dtype=torch.bool)
        # action_mask: [env, n_traj, action_dim], True means feasible.
        counts = mask.sum(dim=-1).to(torch.float32)
        if mode == "hard":
            gate = (counts >= min_actions).to(torch.float32)
        else:
            gate = ((counts - min_actions) / span).clamp(0.0, 1.0)
        gates.append(gate)
    if not gates:
        return torch.ones_like(valid_masks, dtype=torch.float32)
    gate_tensor = torch.stack(gates, dim=0)
    if gate_tensor.shape[0] < valid_masks.shape[0]:
        pad = torch.ones_like(valid_masks[gate_tensor.shape[0]:], dtype=torch.float32)
        gate_tensor = torch.cat([gate_tensor, pad], dim=0)
    return gate_tensor.to(valid_masks.device, dtype=torch.float32) * valid_masks.to(torch.float32)


def _variance_group_gate(objectives, args):
    if not bool(getattr(args, "use_variance_group_gate", False)):
        return torch.ones_like(objectives)
    gate = torch.zeros_like(objectives)
    v_min = float(getattr(args, "variance_gate_vmin", 0.005))
    v_max = float(getattr(args, "variance_gate_vmax", 0.030))
    denom_span = max(v_max - v_min, 1e-8)
    for env_idx in range(objectives.shape[0]):
        obj = objectives[env_idx]
        valid = torch.isfinite(obj)
        if valid.sum() <= 1:
            continue
        vals = obj[valid]
        mean_abs = vals.mean().abs().clamp_min(1e-8)
        spread = vals.std(unbiased=False) / mean_abs
        if not torch.isfinite(spread):
            continue
        gate_value = ((spread - v_min) / denom_span).clamp(0.0, 1.0)
        gate[env_idx, valid] = gate_value
    return gate


def _masked_std(tensor, mask):
    vals = tensor[mask]
    if vals.numel() <= 1:
        return 0.0
    std = vals.std()
    if not torch.isfinite(std):
        return 0.0
    return float(std.detach().cpu().item())


def _masked_mean(tensor, mask):
    vals = tensor[mask]
    if vals.numel() == 0:
        return 0.0
    mean = vals.float().mean()
    if not torch.isfinite(mean):
        return 0.0
    return float(mean.detach().cpu().item())


def _comparative_suffix_advantage(rollout, args):
    objectives = rollout["objectives"]
    prefix_objective = rollout.get("prefix_objective", None)
    teacher_suffix_obj = rollout.get("teacher_suffix_obj", None)
    is_offline = rollout.get("is_offline", None)

    if prefix_objective is None or teacher_suffix_obj is None or is_offline is None:
        zero = torch.zeros_like(objectives)
        return zero, zero

    student_suffix = objectives - prefix_objective
    denom = teacher_suffix_obj.abs().clamp_min(1e-8)
    raw = (teacher_suffix_obj - student_suffix) / denom

    valid = (
        is_offline[:, None]
        & torch.isfinite(student_suffix)
        & torch.isfinite(teacher_suffix_obj)
        & (teacher_suffix_obj > 0.0)
    )
    raw = torch.where(valid, raw, torch.zeros_like(raw))
    raw = raw.clamp(-float(args.cmp_raw_clip), float(args.cmp_raw_clip))

    mode = str(getattr(args, "cmp_adv_mode", "raw")).lower()
    if mode == "gap_temp":
        scaled = raw / max(float(args.cmp_gap_temp), 1e-8)
    else:
        scaled = raw
    scaled = scaled.clamp(-float(args.cmp_adv_clip), float(args.cmp_adv_clip))
    return raw, scaled




def _reference_conditioned_advantage(rollout, args):
    objectives = rollout["objectives"]
    reference_obj = rollout.get("reference_obj", None)
    is_offline = rollout.get("is_offline", None)
    reference_allowed = rollout.get("reference_allowed", None)
    if reference_obj is None or is_offline is None:
        zero = torch.zeros_like(objectives)
        return zero, zero, zero

    ref = reference_obj.to(objectives.device, dtype=objectives.dtype).unsqueeze(1)
    denom = (float(args.reference_adv_rho) * ref.abs()).clamp_min(1e-8)
    raw = (ref - objectives) / denom
    if reference_allowed is None:
        allow = is_offline.to(objectives.device).unsqueeze(1)
    else:
        allow = reference_allowed.to(objectives.device).unsqueeze(1)
    valid = allow & is_offline.to(objectives.device).unsqueeze(1) & torch.isfinite(objectives) & torch.isfinite(ref) & (ref > 0.0)
    raw = torch.where(valid, raw, torch.zeros_like(raw))
    clipped = raw.clamp(-float(args.reference_adv_clip), float(args.reference_adv_clip))

    mode = str(getattr(args, "reference_adv_gate_mode", "fixed")).lower()
    if mode == "linear":
        finite_obj = torch.where(torch.isfinite(objectives), objectives, torch.full_like(objectives, float("inf")))
        best = finite_obj.min(dim=1).values.unsqueeze(1)
        regret_rel = (best - ref) / ref.abs().clamp_min(1e-8)
        gate = (regret_rel / max(float(args.reference_adv_gate_temp), 1e-8)).clamp(0.0, 1.0)
    elif mode == "hard":
        finite_obj = torch.where(torch.isfinite(objectives), objectives, torch.full_like(objectives, float("inf")))
        best = finite_obj.min(dim=1).values.unsqueeze(1)
        regret_rel = (best - ref) / ref.abs().clamp_min(1e-8)
        gate = (regret_rel > float(args.reference_adv_hard_threshold)).to(objectives.dtype)
    else:
        gate = torch.ones((objectives.shape[0], 1), device=objectives.device, dtype=objectives.dtype)
    gate = torch.where(valid.any(dim=1, keepdim=True), gate, torch.zeros_like(gate))
    gate = gate.expand_as(objectives)
    used = clipped * gate
    return raw, used, gate


def _effective_group_adv_coef(args):
    group_coef = float(getattr(args, "group_adv_coef", 0.0))
    if abs(group_coef) > 0.0:
        return group_coef
    return float(getattr(args, "rank_adv_coef", 0.0))

def _scheduled_obj_progress_weights(args, device):
    schedule = str(getattr(args, "adv_weight_schedule", "fixed")).lower()
    update_step = int(getattr(args, "_current_update_step", 0))
    use_late = False
    if schedule in ("staged", "epoch_30_70", "updates_30_70"):
        if update_step < int(args.adv_mid_update):
            obj_w = float(args.adv_early_obj_weight)
            prog_w = float(args.adv_early_progress_weight)
            stage = "early"
        elif update_step < int(args.adv_late_update):
            obj_w = float(args.adv_mid_obj_weight)
            prog_w = float(args.adv_mid_progress_weight)
            stage = "mid"
        else:
            obj_w = float(args.adv_late_obj_weight)
            prog_w = float(args.adv_late_progress_weight)
            stage = "late"
    elif schedule == "finish_rate":
        streak = int(getattr(args, "_finish_rate_streak", 0))
        if streak >= int(args.adv_fr_patience):
            obj_w = float(args.adv_late_obj_weight)
            prog_w = float(args.adv_late_progress_weight)
            stage = "fr_late"
            use_late = True
        else:
            obj_w = float(args.adv_objective_weight)
            prog_w = float(args.adv_progress_weight)
            stage = "fixed"
    else:
        obj_w = float(args.adv_objective_weight)
        prog_w = float(args.adv_progress_weight)
        stage = "fixed"

    weights = torch.tensor([max(obj_w, 0.0), max(prog_w, 0.0)], device=device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-8)
    args.last_adv_component_weights = {"objective": float(weights[0].cpu().item()), "progress": float(weights[1].cpu().item())}
    args.last_adv_weight_stage = stage
    args.last_adv_fr_late = use_late
    return weights[0], weights[1], stage


def _compute_gae_and_returns(agent, rollout, args, device):
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    valid_masks = rollout["valid_masks"].bool()
    valid_step = int(rollout["valid_step"])

    def gae_for(reward_tensor, value_tensor, next_value_tensor):
        advantages = torch.zeros_like(reward_tensor, device=device)
        lastgaelam = torch.zeros(reward_tensor.shape[1], reward_tensor.shape[2], device=device)
        for t in reversed(range(valid_step)):
            if t == valid_step - 1:
                nextnonterminal = 1.0 - rollout["next_done"]
                nextvalues = next_value_tensor
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = value_tensor[t + 1]
            delta = reward_tensor[t] + args.gamma * nextvalues * nextnonterminal - value_tensor[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        return advantages, advantages + value_tensor

    adv_parts = {}
    with torch.no_grad():
        next_value = agent.get_value_cached(rollout["next_obs"], rollout["encoder_state"])
        if values.dim() == 4:
            next_value = next_value.view(rewards.shape[1], rewards.shape[2], values.shape[-1])
            names = _decomposed_component_names(args)
            component_rewards = rollout.get("reward_components", None)
            if component_rewards is None:
                raise ValueError("multi-head incumbent PPO requires reward_components in rollout.")
            adv_list = []
            ret_list = []
            for head_idx, name in enumerate(names):
                adv_i, ret_i = gae_for(component_rewards[name], values[..., head_idx], next_value[..., head_idx])
                if args.adv_norm_scope == "instance":
                    adv_i = _instance_normalize_advantages(adv_i, valid_masks)
                elif args.norm_adv:
                    vals = adv_i[valid_masks]
                    if vals.numel() > 1:
                        adv_i = (adv_i - vals.mean()) / (vals.std() + 1e-8)
                        adv_i = adv_i * valid_masks
                adv_list.append(adv_i)
                ret_list.append(ret_i)
            component_adv = torch.stack(adv_list, dim=-1)
            returns = torch.stack(ret_list, dim=-1)

            if tuple(names) == ("objective", "progress"):
                obj_adv = component_adv[..., 0]
                prog_adv = component_adv[..., 1]
                if bool(args.success_gated_progress):
                    success = torch.isfinite(rollout["objectives"]).unsqueeze(0).expand(valid_step, -1, -1)
                    success_obj = float(args.success_obj_weight) * obj_adv
                    fail_obj = float(args.failure_obj_weight) * obj_adv
                    fail_prog = float(args.failure_progress_weight) * prog_adv
                    actor_advantages = torch.where(success, success_obj, fail_obj + fail_prog)
                    args.last_adv_component_weights = {
                        "success_objective": float(args.success_obj_weight),
                        "failure_objective": float(args.failure_obj_weight),
                        "failure_progress": float(args.failure_progress_weight),
                    }
                    args.last_adv_weight_stage = "success_gated"
                    adv_parts["obj_base"] = torch.where(success, success_obj, fail_obj)
                    adv_parts["progress_base"] = torch.where(success, torch.zeros_like(prog_adv), fail_prog)
                else:
                    obj_w, prog_w, stage = _scheduled_obj_progress_weights(args, device)
                    adv_parts["obj_base"] = obj_w * obj_adv
                    adv_parts["progress_base"] = prog_w * prog_adv
                    actor_advantages = adv_parts["obj_base"] + adv_parts["progress_base"]
                adv_parts["obj_norm"] = obj_adv
                adv_parts["progress_norm"] = prog_adv
            else:
                weights = _decomposed_component_weights(args, names, device=device, dtype=torch.float32)
                weights = torch.clamp(weights[: component_adv.shape[-1]], min=0.0)
                weights = weights / weights.sum().clamp_min(1e-8)
                actor_advantages = (component_adv * weights.view(1, 1, 1, -1)).sum(dim=-1)
                args.last_adv_component_weights = {name: float(weights[i].detach().cpu().item()) for i, name in enumerate(names)}
                args.last_adv_weight_stage = "generic"
        else:
            next_value = next_value.squeeze(-1)
            advantages, returns = gae_for(rewards, values, next_value)
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
            adv_parts["base_actor"] = actor_advantages

    base_actor = actor_advantages.clone()
    group_contrib = torch.zeros_like(actor_advantages)
    cmp_contrib = torch.zeros_like(actor_advantages)
    ref_contrib = torch.zeros_like(actor_advantages)
    group_adv = torch.zeros_like(rollout["objectives"])
    cmp_raw = torch.zeros_like(rollout["objectives"])
    cmp_used = torch.zeros_like(rollout["objectives"])
    ref_raw = torch.zeros_like(rollout["objectives"])
    ref_used = torch.zeros_like(rollout["objectives"])
    ref_gate = torch.zeros_like(rollout["objectives"])
    var_gate = _variance_group_gate(rollout["objectives"], args)
    fork_gate = _decision_fork_gate(rollout, args, valid_masks)

    group_coef = _effective_group_adv_coef(args)
    if group_coef > 0:
        group_adv = _trajectory_rank_advantage(
            rollout["objectives"],
            float(args.group_adv_clip),
            mode=getattr(args, "group_adv_mode", "zscore"),
            std_floor=getattr(args, "group_adv_std_floor", 0.0),
            top_frac=getattr(args, "group_adv_top_frac", 0.25),
        )
        group_adv = group_adv * var_gate
        group_contrib = group_coef * group_adv.unsqueeze(0) * fork_gate
        actor_advantages = actor_advantages + group_contrib

    if float(getattr(args, "reference_adv_coef", 0.0)) > 0:
        ref_raw, ref_used, ref_gate = _reference_conditioned_advantage(rollout, args)
        ref_used = ref_used * var_gate
        ref_contrib = float(args.reference_adv_coef) * ref_used.unsqueeze(0) * fork_gate
        actor_advantages = actor_advantages + ref_contrib

    if args.cmp_adv_coef > 0:
        cmp_raw, cmp_used = _comparative_suffix_advantage(rollout, args)
        cmp_contrib = float(args.cmp_adv_coef) * cmp_used.unsqueeze(0)
        actor_advantages = actor_advantages + cmp_contrib

    final_adv = actor_advantages * valid_masks
    adv_parts["base_actor"] = base_actor * valid_masks
    adv_parts["group"] = group_adv.unsqueeze(0).expand_as(actor_advantages) * valid_masks
    adv_parts["group_contrib"] = group_contrib * valid_masks
    adv_parts["rank"] = adv_parts["group"]
    adv_parts["rank_contrib"] = adv_parts["group_contrib"]
    adv_parts["ref_raw"] = ref_raw.unsqueeze(0).expand_as(actor_advantages) * valid_masks
    adv_parts["ref_used"] = ref_used.unsqueeze(0).expand_as(actor_advantages) * valid_masks
    adv_parts["ref_gate"] = ref_gate.unsqueeze(0).expand_as(actor_advantages) * valid_masks
    adv_parts["var_gate"] = var_gate.unsqueeze(0).expand_as(actor_advantages) * valid_masks
    adv_parts["fork_gate"] = fork_gate * valid_masks
    adv_parts["ref_contrib"] = ref_contrib * valid_masks
    adv_parts["cmp_raw"] = cmp_raw.unsqueeze(0).expand_as(actor_advantages) * valid_masks
    adv_parts["cmp_used"] = cmp_used.unsqueeze(0).expand_as(actor_advantages) * valid_masks
    adv_parts["cmp_contrib"] = cmp_contrib * valid_masks
    adv_parts["final"] = final_adv
    rollout["adv_parts"] = {k: v.detach() for k, v in adv_parts.items()}

    if bool(getattr(args, "adv_diag", True)):
        mask = valid_masks
        debug = {
            "stage": str(getattr(args, "last_adv_weight_stage", "unknown")),
            "obj_w": float(getattr(args, "last_adv_component_weights", {}).get("objective", getattr(args, "success_obj_weight", 0.0))),
            "prog_w": float(getattr(args, "last_adv_component_weights", {}).get("progress", getattr(args, "failure_progress_weight", 0.0))),
            "base_std": _masked_std(adv_parts["base_actor"], mask),
            "group_std": _masked_std(adv_parts["group"], mask),
            "group_contrib_std": _masked_std(adv_parts["group_contrib"], mask),
            "rank_std": _masked_std(adv_parts["rank"], mask),
            "rank_contrib_std": _masked_std(adv_parts["rank_contrib"], mask),
            "ref_raw_std": _masked_std(adv_parts["ref_raw"], mask),
            "ref_used_std": _masked_std(adv_parts["ref_used"], mask),
            "ref_gate_mean": _masked_mean(adv_parts["ref_gate"], mask),
            "ref_gate_active": _masked_mean((adv_parts["ref_gate"] > 1e-8).to(torch.float32), mask),
            "var_gate_mean": _masked_mean(adv_parts["var_gate"], mask),
            "var_gate_active": _masked_mean((adv_parts["var_gate"] > 1e-8).to(torch.float32), mask),
            "fork_gate_mean": _masked_mean(adv_parts["fork_gate"], mask),
            "fork_gate_active": _masked_mean((adv_parts["fork_gate"] > 1e-8).to(torch.float32), mask),
            "ref_contrib_std": _masked_std(adv_parts["ref_contrib"], mask),
            "cmp_raw_std": _masked_std(adv_parts["cmp_raw"], mask),
            "cmp_used_std": _masked_std(adv_parts["cmp_used"], mask),
            "cmp_contrib_std": _masked_std(adv_parts["cmp_contrib"], mask),
            "final_std": _masked_std(final_adv, mask),
        }
        if "obj_norm" in adv_parts:
            debug["obj_norm_std"] = _masked_std(adv_parts["obj_norm"], mask)
        if "progress_norm" in adv_parts:
            debug["progress_norm_std"] = _masked_std(adv_parts["progress_norm"], mask)
        rollout["adv_debug"] = debug

    return final_adv, returns


def _policy_loss_from_adv(adv, ratio, valid, clip_coef):
    valid_count = valid.sum().float().clamp_min(1.0)
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    return (torch.max(pg1, pg2) * valid).sum() / valid_count


def _route_loss_effective_coef(args):
    coef = float(getattr(args, "route_loss_coef", 0.0))
    warmup = int(getattr(args, "route_loss_warmup_updates", 0))
    if coef <= 0.0 or warmup <= 0:
        return max(0.0, coef)
    update_step = int(getattr(args, "_current_update_step", 0))
    return coef * min(1.0, max(0.0, float(update_step + 1) / float(warmup)))


def _solution_level_clipped_loss(
    new_logprobs,
    old_logprobs,
    valid_step_mask,
    objectives,
    reference_obj,
    is_offline,
    reference_allowed,
    args,
):
    """Auxiliary route-level PPO loss over complete successful trajectories."""
    device = objectives.device
    dtype = objectives.dtype
    if new_logprobs.numel() == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "route_loss": 0.0,
            "route_ratio_mean": 1.0,
            "route_ratio_std": 0.0,
            "route_clip_frac": 0.0,
            "route_adv_mean": 0.0,
            "route_adv_std": 0.0,
            "route_used": 0,
        }

    mask_f = valid_step_mask.to(dtype)
    valid_counts = mask_f.sum(dim=0).clamp_min(1.0)
    delta_logp = new_logprobs - old_logprobs
    mean_delta = (delta_logp * mask_f).sum(dim=0) / valid_counts
    r_route = torch.exp(mean_delta.clamp(-20.0, 20.0))

    source = str(getattr(args, "route_adv_source", "group_ref")).lower()
    use_group = source in ("group", "group_ref", "ref_group")
    use_ref = source in ("ref", "group_ref", "ref_group")

    group_adv = torch.zeros_like(objectives)
    if use_group:
        std_floor = max(float(getattr(args, "route_adv_std_floor", 0.0)), 0.0)
        for env_idx in range(objectives.shape[0]):
            obj = objectives[env_idx]
            finite = torch.isfinite(obj)
            if finite.sum() <= 1:
                continue
            vals = obj[finite]
            std = vals.std(unbiased=False)
            denom = torch.clamp(std, min=std_floor)
            if torch.isfinite(denom) and float(denom.detach().cpu().item()) > 1e-12:
                group_adv[env_idx, finite] = (vals.mean() - vals) / (denom + 1e-8)
        group_adv = group_adv.clamp(-float(args.group_adv_clip), float(args.group_adv_clip))

    ref_used = torch.zeros_like(objectives)
    if use_ref and reference_obj is not None and is_offline is not None:
        sub_rollout = {
            "objectives": objectives,
            "reference_obj": reference_obj,
            "is_offline": is_offline,
        }
        if reference_allowed is not None:
            sub_rollout["reference_allowed"] = reference_allowed
        _ref_raw, ref_used, _ref_gate = _reference_conditioned_advantage(sub_rollout, args)

    route_adv = (
        float(args.group_adv_coef) * group_adv
        + float(args.reference_adv_coef) * ref_used
    ).detach()

    route_mask = torch.isfinite(objectives) & (valid_counts > 0)
    if bool(getattr(args, "only_success_route_loss", True)):
        route_mask = route_mask & torch.isfinite(objectives)

    mask_mode = str(getattr(args, "route_mask_mode", "all")).lower()
    if mask_mode in ("positive", "positive_elite", "elite_positive"):
        route_mask = route_mask & (route_adv > float(getattr(args, "route_positive_eps", 0.0)))
    if mask_mode in ("elite", "positive_elite", "elite_positive"):
        elite_mask = torch.zeros_like(route_mask)
        elite_frac = min(max(float(getattr(args, "route_elite_frac", 0.25)), 0.0), 1.0)
        for env_idx in range(objectives.shape[0]):
            obj = objectives[env_idx]
            finite = torch.isfinite(obj)
            if finite.sum() == 0:
                continue
            idx = torch.nonzero(finite, as_tuple=False).flatten()
            vals = obj[idx]
            k = max(1, int(np.ceil(elite_frac * int(idx.numel()))))
            elite_idx = idx[torch.argsort(vals)[:k]]
            elite_mask[env_idx, elite_idx] = True
        route_mask = route_mask & elite_mask

    if route_mask.sum() == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "route_loss": 0.0,
            "route_ratio_mean": 1.0,
            "route_ratio_std": 0.0,
            "route_clip_frac": 0.0,
            "route_adv_mean": 0.0,
            "route_adv_std": 0.0,
            "route_used": 0,
        }

    r = r_route[route_mask]
    adv = route_adv[route_mask]
    clipped_r = torch.clamp(
        r,
        1.0 - float(args.route_clip_eps),
        1.0 + float(args.route_clip_eps),
    )
    route_loss = -torch.min(r * adv, clipped_r * adv).mean()
    clip_frac = ((r > 1.0 + float(args.route_clip_eps)) | (r < 1.0 - float(args.route_clip_eps))).float().mean()
    logs = {
        "route_loss": float(route_loss.detach().cpu().item()),
        "route_ratio_mean": float(r.detach().mean().cpu().item()),
        "route_ratio_std": float(r.detach().std(unbiased=False).cpu().item()) if r.numel() > 1 else 0.0,
        "route_clip_frac": float(clip_frac.detach().cpu().item()),
        "route_adv_mean": float(adv.detach().mean().cpu().item()),
        "route_adv_std": float(adv.detach().std(unbiased=False).cpu().item()) if adv.numel() > 1 else 0.0,
        "route_used": int(route_mask.sum().detach().cpu().item()),
    }
    return route_loss, logs


def _flat_grad_vector(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    pieces = [g.detach().reshape(-1) for g in grads if g is not None]
    if not pieces:
        return None
    return torch.cat(pieces)


def _grad_cosine(a, b):
    if a is None or b is None:
        return float("nan")
    denom = a.norm() * b.norm()
    if float(denom.detach().cpu().item()) <= 1e-12:
        return float("nan")
    return float((torch.dot(a, b) / denom).detach().cpu().item())


def _grad_cosine_diagnostics(agent, ratio, valid, adv_parts, mb_inds, args):
    params = [p for p in agent.backbone.parameters() if p.requires_grad]
    if not params:
        return {}

    def grad_for(key):
        part = adv_parts.get(key, None)
        if part is None:
            return None
        adv = part[mb_inds]
        if not torch.isfinite(adv).any() or float(adv.abs().sum().detach().cpu().item()) <= 1e-12:
            return None
        loss = _policy_loss_from_adv(adv, ratio, valid, float(args.clip_coef))
        return _flat_grad_vector(loss, params)

    obj_g = grad_for("obj_base")
    prog_g = grad_for("progress_base")
    base_g = grad_for("base_actor")
    group_g = grad_for("group_contrib")
    ref_g = grad_for("ref_contrib")
    cmp_g = grad_for("cmp_contrib")
    return {
        "cos_obj_progress": _grad_cosine(obj_g, prog_g),
        "cos_base_group": _grad_cosine(base_g, group_g),
        "cos_base_ref": _grad_cosine(base_g, ref_g),
        "cos_base_cmp": _grad_cosine(base_g, cmp_g),
    }


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

    b_obs = {k: np.concatenate([obs_t[k] for obs_t in obs], axis=0) for k in envs.single_observation_space}
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_old_logprobs = old_logprobs.reshape(-1, args.n_traj)
    b_advantages = advantages.reshape(-1, args.n_traj).detach()
    b_adv_parts = {
        k: v.reshape(-1, args.n_traj).detach()
        for k, v in rollout.get("adv_parts", {}).items()
    }
    if old_values.dim() == 4:
        value_heads = old_values.shape[-1]
        b_returns = returns.reshape(-1, args.n_traj, value_heads)
        b_old_values = old_values.reshape(-1, args.n_traj, value_heads)
    else:
        value_heads = 1
        b_returns = returns.reshape(-1, args.n_traj)
        b_old_values = old_values.reshape(-1, args.n_traj)
    b_valid = valid_masks.reshape(-1, args.n_traj)

    pg_losses = []
    v_losses = []
    entropies = []
    kls = []
    route_losses = []
    route_ratio_means = []
    route_ratio_stds = []
    route_clip_fracs = []
    route_adv_means = []
    route_adv_stds = []
    route_used_counts = []
    route_effective_coefs = []
    grad_cos_info = {}

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
            pg_loss = _policy_loss_from_adv(mb_adv, ratio, mb_valid, float(args.clip_coef))

            if (
                bool(getattr(args, "grad_cos_diag", False))
                and not grad_cos_info
                and int(getattr(args, "_current_update_step", 0)) % max(1, int(args.grad_cos_freq)) == 0
            ):
                grad_cos_info = _grad_cosine_diagnostics(
                    agent=agent,
                    ratio=ratio,
                    valid=mb_valid,
                    adv_parts=b_adv_parts,
                    mb_inds=mb_inds,
                    args=args,
                )

            if value_heads > 1:
                newvalue = newvalue.view(-1, args.n_traj, value_heads)
                value_valid = mb_valid.unsqueeze(-1).expand_as(newvalue)
                value_valid_count = value_valid.sum().float().clamp_min(1.0)
            else:
                newvalue = newvalue.squeeze(-1).view(-1, args.n_traj)
                value_valid = mb_valid
                value_valid_count = valid_count_f

            returns_mb = b_returns[mb_inds]
            old_values_mb = b_old_values[mb_inds]
            value_loss_raw = F.smooth_l1_loss(newvalue, returns_mb, reduction="none", beta=1.0)
            if args.clip_vloss:
                v_clipped = old_values_mb + torch.clamp(newvalue - old_values_mb, -args.clip_coef, args.clip_coef)
                value_loss_clip = F.smooth_l1_loss(v_clipped, returns_mb, reduction="none", beta=1.0)
                v_loss = 0.5 * (torch.max(value_loss_raw, value_loss_clip) * value_valid).sum() / value_valid_count
            else:
                v_loss = 0.5 * (value_loss_raw * value_valid).sum() / value_valid_count

            entropy_loss = (entropy * mb_valid).sum() / valid_count_f

            route_loss = torch.zeros((), device=pg_loss.device, dtype=pg_loss.dtype)
            route_coef_eff = 0.0
            if bool(getattr(args, "use_route_level_loss", False)):
                route_coef_eff = _route_loss_effective_coef(args)
                route_objectives = rollout["objectives"][mbenvinds]
                route_reference_obj = rollout.get("reference_obj", None)
                if route_reference_obj is not None:
                    route_reference_obj = route_reference_obj[mbenvinds]
                route_is_offline = rollout.get("is_offline", None)
                if route_is_offline is not None:
                    route_is_offline = route_is_offline[mbenvinds]
                route_reference_allowed = rollout.get("reference_allowed", None)
                if route_reference_allowed is not None:
                    route_reference_allowed = route_reference_allowed[mbenvinds]
                route_loss, route_info = _solution_level_clipped_loss(
                    new_logprobs=newlogprob.reshape(valid_step, envsperbatch, args.n_traj),
                    old_logprobs=b_old_logprobs[mb_inds].reshape(valid_step, envsperbatch, args.n_traj),
                    valid_step_mask=mb_valid.reshape(valid_step, envsperbatch, args.n_traj),
                    objectives=route_objectives,
                    reference_obj=route_reference_obj,
                    is_offline=route_is_offline,
                    reference_allowed=route_reference_allowed,
                    args=args,
                )
                route_losses.append(route_info["route_loss"])
                route_ratio_means.append(route_info["route_ratio_mean"])
                route_ratio_stds.append(route_info["route_ratio_std"])
                route_clip_fracs.append(route_info["route_clip_frac"])
                route_adv_means.append(route_info["route_adv_mean"])
                route_adv_stds.append(route_info["route_adv_std"])
                route_used_counts.append(route_info["route_used"])
                route_effective_coefs.append(route_coef_eff)

            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + route_coef_eff * route_loss
            (loss / accum_steps).backward()
            accum_counter += 1

            if accum_counter % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(agent.backbone.parameters(), args.max_grad_norm_backbone)
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm_critic)
                optim_backbone.step()
                optim_critic.step()
                optim_backbone.zero_grad(set_to_none=True)
                optim_critic.zero_grad(set_to_none=True)

            pg_losses.append(float(pg_loss.detach().cpu().item()))
            v_losses.append(float(v_loss.detach().cpu().item()))
            entropies.append(float(entropy_loss.detach().cpu().item()))

        if accum_counter % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(agent.backbone.parameters(), args.max_grad_norm_backbone)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm_critic)
            optim_backbone.step()
            optim_critic.step()

        if kls and float(np.mean(kls[-args.num_minibatches :])) > args.target_kl:
            stop_early = True

    return {
        "pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
        "v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
        "route_loss": float(np.mean(route_losses)) if route_losses else 0.0,
        "route_coef": float(np.mean(route_effective_coefs)) if route_effective_coefs else 0.0,
        "route_ratio_mean": float(np.mean(route_ratio_means)) if route_ratio_means else 1.0,
        "route_ratio_std": float(np.mean(route_ratio_stds)) if route_ratio_stds else 0.0,
        "route_clip_frac": float(np.mean(route_clip_fracs)) if route_clip_fracs else 0.0,
        "route_adv_mean": float(np.mean(route_adv_means)) if route_adv_means else 0.0,
        "route_adv_std": float(np.mean(route_adv_stds)) if route_adv_stds else 0.0,
        "route_used": float(np.mean(route_used_counts)) if route_used_counts else 0.0,
        "grad_backbone": grad_norm(agent.backbone.parameters()),
        "grad_critic": grad_norm(agent.critic.parameters()),
        "grad_cos": grad_cos_info,
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


def _full_sequence_from_rollout(record, suffix_seq, prefix_len, args):
    suffix_seq = [int(a) for a in list(suffix_seq)]
    if record is None or int(prefix_len) <= 0:
        return suffix_seq
    prefix = _incumbent_action_sequence(record, args)[: int(prefix_len)]
    return [int(a) for a in prefix] + suffix_seq


def _update_incumbents_from_rollout(records_meta, envs, rollout, args, online_buffer):
    best_obj, fr = _best_objectives(rollout)
    best_sequences = _best_action_sequences_from_rollout(rollout)
    prefix_len = rollout.get("prefix_len", torch.zeros(len(records_meta), dtype=torch.long))
    prefix_len_np = prefix_len.detach().cpu().numpy().astype(np.int64)

    online_candidates = []
    for idx, (record, obj) in enumerate(zip(records_meta, best_obj)):
        student_obj = float(obj)
        suffix_seq = best_sequences[idx] if idx < len(best_sequences) else []
        full_seq = _full_sequence_from_rollout(record, suffix_seq, prefix_len_np[idx], args)

        if record is not None:
            incumbent_obj = _incumbent_obj(record)
            if bool(getattr(args, "use_pomo50_regret_state", False)):
                _classify_policy_vs_incumbent(record, rollout["objectives"][idx].detach().cpu().numpy(), args)
            else:
                record["student_best_obj"] = student_obj
                record["student_regret"] = student_obj - incumbent_obj if np.isfinite(student_obj) else incumbent_obj
                record["student_regret_known"] = True
            record["student_action_sequence"] = list(full_seq)
            if np.isfinite(student_obj) and student_obj + float(args.incumbent_update_eps) < incumbent_obj:
                record["incumbent_obj"] = student_obj
                record["incumbent_action_sequence"] = list(full_seq)
                record["incumbent_source"] = "policy"
                record["incumbent_update_count"] = int(record.get("incumbent_update_count", 0)) + 1
                record["student_regret"] = 0.0
        else:
            if np.isfinite(student_obj) and len(full_seq) > 0:
                online_candidates.append((idx, student_obj, full_seq))

    added_online = 0
    if online_candidates and bool(args.use_self_generated_buffer):
        objs = np.asarray([x[1] for x in online_candidates], dtype=np.float64)
        threshold = float(np.quantile(objs, float(args.online_elite_quantile)))
        threshold = min(threshold, float(args.online_elite_max_obj))
        online_candidates = [x for x in online_candidates if x[1] <= threshold]
        online_candidates = sorted(online_candidates, key=lambda x: x[1])[: int(args.online_elite_add_per_update)]
        for env_idx, student_obj, seq in online_candidates:
            if len(online_buffer) >= int(args.self_buffer_max_records):
                break
            base_env = _unwrap_env(envs.envs[env_idx])
            record = {
                "instance": deepcopy(getattr(base_env, "current_instance", None)),
                "teacher_obj": float(student_obj),
                "incumbent_obj": float(student_obj),
                "teacher_action_sequence": list(seq),
                "incumbent_action_sequence": list(seq),
                "buffer_source": "self_generated",
                "student_regret": 0.0,
            }
            if record["instance"] is not None:
                online_buffer.append(record)
                added_online += 1

    return {
        "updated_incumbents": int(sum(1 for r in records_meta if r is not None and int(r.get("incumbent_update_count", 0)) > 0)),
        "added_online": int(added_online),
        "fr_mean": float(np.mean(fr)) if len(fr) else 0.0,
    }


def _cache_student_sequences(records_meta, rollout, args):
    best_obj, _fr = _best_objectives(rollout)
    best_sequences = _best_action_sequences_from_rollout(rollout)
    prefix_len = rollout.get("prefix_len", torch.zeros(len(records_meta), dtype=torch.long))
    prefix_len_np = prefix_len.detach().cpu().numpy().astype(np.int64)
    seen = set()
    for idx, record in enumerate(records_meta):
        if record is None:
            continue
        rid = id(record)
        if rid in seen:
            continue
        seen.add(rid)
        suffix_seq = best_sequences[idx] if idx < len(best_sequences) else []
        full_seq = _full_sequence_from_rollout(record, suffix_seq, prefix_len_np[idx], args)
        if bool(getattr(args, "use_pomo50_regret_state", False)):
            _classify_policy_vs_incumbent(record, rollout["objectives"][idx].detach().cpu().numpy(), args)
        else:
            record["student_best_obj"] = float(best_obj[idx])
            record["student_regret"] = float(best_obj[idx]) - _incumbent_obj(record) if np.isfinite(float(best_obj[idx])) else _incumbent_obj(record)
            record["student_regret_known"] = True
        record["student_action_sequence"] = list(full_seq)


def _make_aux_envs(args, config, records, update_step, salt=0):
    if not records:
        return None
    clean_cfg = _clean_env_cfg(args, args.lambda_fail_init)
    return SyncVectorEnv(
        [
            make_env(
                args.env_id,
                int(args.seed + 700000 + update_step * 1000 + salt + i),
                cfg={
                    "env_mode": "train",
                    "config": config,
                    "n_traj": 1,
                    "eval_data": deepcopy(record["instance"]),
                    "num_customers": args.train_cus_num,
                    "num_charging_stations": args.train_cs_num,
                    "gamma": args.gamma,
                    **clean_cfg,
                },
            )
            for i, record in enumerate(records)
        ]
    )


def _forced_sequence_logprobs(agent, envs, sequences, device, max_steps, length_norm=True):
    if envs is None or not sequences:
        return None
    obs = envs.reset()
    cached_embeddings = agent.backbone.encode(obs)
    batch_size = len(sequences)
    active = np.ones(batch_size, dtype=np.bool_)
    logprob_sums = torch.zeros(batch_size, device=device, dtype=torch.float32)
    lengths = torch.zeros(batch_size, device=device, dtype=torch.float32)
    infeasible_actions = 0

    for step in range(max_steps):
        target = np.zeros((batch_size, 1), dtype=np.int64)
        has_target = np.zeros(batch_size, dtype=np.bool_)
        for i, seq in enumerate(sequences):
            if active[i] and step < len(seq):
                target[i, 0] = int(seq[step])
                has_target[i] = True
        if not np.any(has_target):
            break
        action_mask = np.asarray(obs["action_mask"], dtype=np.bool_)
        feasible = action_mask[np.arange(batch_size), 0, target[:, 0]]
        valid = has_target & active & feasible
        infeasible_actions += int(np.sum(has_target & active & (~feasible)))

        if np.any(valid):
            target_t = torch.tensor(target, device=device, dtype=torch.long)
            _, logprob, _, _, _ = agent.get_action_and_value_cached(
                obs,
                action=target_t,
                cached_embeddings=cached_embeddings,
            )
            valid_t = torch.tensor(valid, device=device, dtype=torch.bool)
            logprob_sums[valid_t] += logprob.squeeze(-1)[valid_t]
            lengths[valid_t] += 1.0

        forced_action = target.copy()
        forced_action[~valid, 0] = 0
        obs, _, done, _ = envs.step(forced_action)
        active = active & valid & (~done.reshape(-1))
        if not np.any(active):
            break

    score = logprob_sums / lengths.clamp_min(1.0) if length_norm else logprob_sums
    return score, lengths > 0, lengths, infeasible_actions


def _incumbent_aux_update(agent, optim_backbone, args, config, records_meta, device, update_step):
    if not (bool(args.use_route_preference) or bool(args.use_selective_bc)):
        return None

    unique = []
    seen = set()
    for record in records_meta:
        if record is None:
            continue
        rid = id(record)
        if rid in seen:
            continue
        seen.add(rid)
        student_obj = float(record.get("student_best_obj", float("inf")))
        incumbent_obj = _incumbent_obj(record)
        if not np.isfinite(student_obj) or not np.isfinite(incumbent_obj):
            continue
        if student_obj <= incumbent_obj + float(args.selective_bc_margin):
            continue
        if not record.get("student_action_sequence"):
            continue
        unique.append(record)

    if not unique:
        return None
    unique = sorted(unique, key=lambda r: float(r.get("student_regret", 0.0)), reverse=True)
    unique = unique[: int(args.aux_batch_size)]

    incumbent_seqs = [_incumbent_action_sequence(r, args)[: int(args.aux_max_steps)] for r in unique]
    student_seqs = [list(r.get("student_action_sequence", []))[: int(args.aux_max_steps)] for r in unique]
    valid = [len(a) > 0 and len(b) > 0 for a, b in zip(incumbent_seqs, student_seqs)]
    unique = [r for r, keep in zip(unique, valid) if keep]
    incumbent_seqs = [s for s, keep in zip(incumbent_seqs, valid) if keep]
    student_seqs = [s for s, keep in zip(student_seqs, valid) if keep]
    if not unique:
        return None

    pos_envs = None
    neg_envs = None
    agent.train()
    losses = []
    info = {"batch": len(unique), "bc_loss": 0.0, "pref_loss": 0.0, "pref_delta": 0.0, "infeasible": 0}
    try:
        if bool(args.use_selective_bc):
            pos_envs = _make_aux_envs(args, config, unique, update_step, salt=11)
            pos = _forced_sequence_logprobs(agent, pos_envs, incumbent_seqs, device, int(args.aux_max_steps), length_norm=False)
            if pos is not None:
                logp, valid_pos, lengths, infeasible = pos
                if torch.any(valid_pos):
                    bc_loss = -(logp[valid_pos] / lengths[valid_pos].clamp_min(1.0)).mean()
                    losses.append(float(args.selective_bc_coef) * bc_loss)
                    info["bc_loss"] = float(bc_loss.detach().cpu().item())
                    info["infeasible"] += int(infeasible)
        if bool(args.use_route_preference):
            if pos_envs is not None:
                pos_envs.close()
                pos_envs = None
            pos_envs = _make_aux_envs(args, config, unique, update_step, salt=21)
            neg_envs = _make_aux_envs(args, config, unique, update_step, salt=31)
            pos = _forced_sequence_logprobs(agent, pos_envs, incumbent_seqs, device, int(args.aux_max_steps), length_norm=True)
            neg = _forced_sequence_logprobs(agent, neg_envs, student_seqs, device, int(args.aux_max_steps), length_norm=True)
            if pos is not None and neg is not None:
                pos_lp, pos_valid, _pos_len, pos_inf = pos
                neg_lp, neg_valid, _neg_len, neg_inf = neg
                valid_pair = pos_valid & neg_valid
                if torch.any(valid_pair):
                    delta = pos_lp - neg_lp
                    pref_loss = -F.logsigmoid(float(args.route_pref_beta) * delta[valid_pair]).mean()
                    losses.append(float(args.route_pref_coef) * pref_loss)
                    info["pref_loss"] = float(pref_loss.detach().cpu().item())
                    info["pref_delta"] = float(delta[valid_pair].detach().mean().cpu().item())
                    info["infeasible"] += int(pos_inf + neg_inf)

        if not losses:
            return None
        loss = torch.stack(losses).sum()
        optim_backbone.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.backbone.parameters(), args.max_grad_norm_backbone)
        optim_backbone.step()
        optim_backbone.zero_grad(set_to_none=True)
        info["loss"] = float(loss.detach().cpu().item())
        return info
    finally:
        if pos_envs is not None:
            pos_envs.close()
        if neg_envs is not None:
            neg_envs.close()


def _recompute_regret_probe(agent, args, config, records, device, update_step, rng):
    if int(args.regret_recompute_freq) <= 0 or not records:
        return None
    if update_step % int(args.regret_recompute_freq) != 0:
        return None

    probe_size = min(int(args.regret_probe_size), len(records))
    indices = rng.choice(len(records), size=probe_size, replace=False)
    sampled = [records[int(i)] for i in indices]
    clean_cfg = _clean_env_cfg(args, args.lambda_fail_init)
    envs = None
    try:
        envs = SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    int(args.seed + 900000 + update_step * 1000 + i),
                    cfg={
                        "env_mode": "train",
                        "config": config,
                        "n_traj": args.n_traj,
                        "eval_data": deepcopy(record["instance"]),
                        "num_customers": args.train_cus_num,
                        "num_charging_stations": args.train_cs_num,
                        "gamma": args.gamma,
                        **clean_cfg,
                    },
                )
                for i, record in enumerate(sampled)
            ]
        )
        group_meta = [{"kind": "regret_probe", "temperature": 1.0, "use_prefix": False} for _ in sampled]
        initial_obs, meta = _reset_envs_for_rollout(envs, sampled, group_meta, args, rng)
        rollout = _student_rollout(
            agent=agent,
            envs=envs,
            args=args,
            device=device,
            num_steps=int(args.num_steps),
            initial_obs=initial_obs,
            rollout_meta=meta,
        )
        best_obj, _fr = _best_objectives(rollout)
        for probe_idx, (record, obj) in enumerate(zip(sampled, best_obj)):
            student_obj = float(obj)
            inc_obj = _incumbent_obj(record)
            if bool(getattr(args, "use_pomo50_regret_state", False)):
                _classify_policy_vs_incumbent(record, rollout["objectives"][probe_idx].detach().cpu().numpy(), args)
            else:
                record["student_best_obj"] = student_obj
                record["student_regret"] = student_obj - inc_obj if np.isfinite(student_obj) else inc_obj
                record["student_regret_known"] = True
        finite = best_obj[np.isfinite(best_obj)]
        return {
            "probe": int(probe_size),
            "obj_mean": float(finite.mean()) if finite.size else float("nan"),
            "regret_mean": float(np.mean([float(r.get("student_regret", 0.0)) for r in sampled])),
        }
    finally:
        if envs is not None:
            envs.close()


def _success_rate_from_rollout(rollout, customer_numbers):
    actions = rollout["actions"]
    valid_step = rollout["valid_step"]
    visu_actions = actions.reshape((valid_step, -1)).cpu().numpy().copy()
    visu_actions[visu_actions == 0] = customer_numbers + 1
    visu_actions[visu_actions < 1 + customer_numbers] = 1
    visu_actions[visu_actions >= 1 + customer_numbers] = 0
    cus_count_per_traj = visu_actions.sum(axis=0)
    return float((cus_count_per_traj == customer_numbers).mean())


def _print_update_summary(update_step, args, rollout, ppo_info, aux_info, archive_info, regret_info, group_meta, lambda_fail):
    best_obj, fr = _best_objectives(rollout)
    finite_obj = best_obj[np.isfinite(best_obj)]
    obj_mean = float(finite_obj.mean()) if finite_obj.size else float("nan")
    prefix_len = rollout.get("prefix_len", torch.zeros(1))
    prefix_mean = float(prefix_len.float().mean().detach().cpu().item())
    group_counts = {}
    for meta in group_meta:
        key = str(meta.get("kind", "unknown"))
        group_counts[key] = group_counts.get(key, 0) + 1
    group_text = ",".join(f"{k}={v}" for k, v in sorted(group_counts.items()))
    aux_text = "none" if not aux_info else (
        f"loss={aux_info.get('loss', 0.0):.5f} bc={aux_info.get('bc_loss', 0.0):.5f} "
        f"pref={aux_info.get('pref_loss', 0.0):.5f} delta={aux_info.get('pref_delta', 0.0):.4f}"
    )
    regret_text = "none" if not regret_info else (
        f"probe={regret_info.get('probe', 0)} regret={regret_info.get('regret_mean', 0.0):.4f}"
    )
    adv_debug = rollout.get("adv_debug", {}) if bool(getattr(args, "adv_diag", True)) else {}
    adv_text = "none" if not adv_debug else (
        f"stage={adv_debug.get('stage', 'NA')} "
        f"w={adv_debug.get('obj_w', 0.0):.2f}/{adv_debug.get('prog_w', 0.0):.2f} "
        f"obj_std={adv_debug.get('obj_norm_std', 0.0):.3f} "
        f"prog_std={adv_debug.get('progress_norm_std', 0.0):.3f} "
        f"group={adv_debug.get('group_contrib_std', adv_debug.get('rank_contrib_std', 0.0)):.3f} "
        f"ref_raw={adv_debug.get('ref_raw_std', 0.0):.3f} "
        f"ref_used={adv_debug.get('ref_used_std', 0.0):.3f} "
        f"ref_gate={adv_debug.get('ref_gate_mean', 0.0):.3f}/{adv_debug.get('ref_gate_active', 0.0):.3f} "
        f"var_gate={adv_debug.get('var_gate_mean', 1.0):.3f}/{adv_debug.get('var_gate_active', 1.0):.3f} "
        f"fork_gate={adv_debug.get('fork_gate_mean', 1.0):.3f}/{adv_debug.get('fork_gate_active', 1.0):.3f} "
        f"ref={adv_debug.get('ref_contrib_std', 0.0):.3f} "
        f"cmp_raw={adv_debug.get('cmp_raw_std', 0.0):.3f} "
        f"cmp_used={adv_debug.get('cmp_used_std', 0.0):.3f} "
        f"cmp={adv_debug.get('cmp_contrib_std', 0.0):.3f} "
        f"final={adv_debug.get('final_std', 0.0):.3f}"
    )
    grad_cos = ppo_info.get("grad_cos", {}) or {}
    grad_text = "none" if not grad_cos else (
        f"obj_prog={grad_cos.get('cos_obj_progress', float('nan')):.3f} "
        f"base_group={grad_cos.get('cos_base_group', grad_cos.get('cos_base_rank', float('nan'))):.3f} "
        f"base_ref={grad_cos.get('cos_base_ref', float('nan')):.3f} "
        f"base_cmp={grad_cos.get('cos_base_cmp', float('nan')):.3f}"
    )
    sl_text = "none" if float(ppo_info.get("route_coef", 0.0)) <= 0.0 else (
        f"loss={ppo_info.get('route_loss', 0.0):.5f} "
        f"coef={ppo_info.get('route_coef', 0.0):.3f} "
        f"ratio={ppo_info.get('route_ratio_mean', 1.0):.3f}/{ppo_info.get('route_ratio_std', 0.0):.3f} "
        f"clip={ppo_info.get('route_clip_frac', 0.0):.3f} "
        f"adv={ppo_info.get('route_adv_mean', 0.0):.3f}/{ppo_info.get('route_adv_std', 0.0):.3f} "
        f"used={ppo_info.get('route_used', 0.0):.1f}"
    )
    print(
        f"[Train] update={update_step} "
        f"groups={group_text} "
        f"fr={float(fr.mean()):.4f} "
        f"obj_best_mean={obj_mean:.4f} "
        f"prefix_len_mean={prefix_mean:.2f} "
        f"lambda_fail={lambda_fail:.3f} "
        f"pg={ppo_info['pg_loss']:.5f} "
        f"vf={ppo_info['v_loss']:.5f} "
        f"kl={ppo_info['kl']:.5f} "
        f"sl=({sl_text}) "
        f"aux=({aux_text}) "
        f"archive_added={archive_info.get('added_online', 0)} "
        f"regret=({regret_text}) "
        f"adv=({adv_text}) "
        f"gradcos=({grad_text})"
    )


def train(args):
    print("---------------- Incumbent PPO Training Info ----------------")
    for key, value in vars(args).items():
        print(key, value)
    print("-------------------------------------------------------------")

    _set_global_seeds(args.seed, deterministic=args.torch_deterministic)

    try:
        gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    except Exception as exc:
        print(f"[GymRegister] skipped or already registered: {exc}")

    device = f"cuda:{args.cuda_id}" if args.cuda and torch.cuda.is_available() else "cpu"
    value_heads = len(_decomposed_component_names(args)) if bool(args.use_decomposed_reward_adv) else 1
    agent = CompetenceAgent(
        device=device,
        name=args.problem,
        tanh_clipping=args.tanh_clipping,
        n_encode_layers=args.n_encode_layers,
        value_heads=value_heads,
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

    records = _load_alns_buffer_from_dir(
        buffer_dir=args.alns_buffer_dir,
        progress_name=args.alns_progress_name,
        instance_pickle_name=args.alns_instance_pickle_name,
        obj_scale=args.alns_obj_scale,
    )
    if not records:
        raise ValueError("IncumbentBuffer initialization failed: no ALNS records were loaded.")
    if int(args.max_incumbent_records) > 0:
        records = records[: int(args.max_incumbent_records)]
    _initialize_incumbent_records(records, args)
    self_generated_buffer = []
    print(f"[IncumbentBuffer] initialized records={len(records)} self_generated=0 value_heads={value_heads}")

    with open(args.eval_data_path, "rb") as f:
        eval_data = pickle.load(f)
    if args.eval_batch_size > len(eval_data):
        raise ValueError(f"eval_batch_size={args.eval_batch_size} > len(eval_data)={len(eval_data)}")

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

    save_dir = os.path.join(args.save_dir, args.exp_name, f"Cus_{args.train_cus_num}_CS_{args.train_cs_num}")
    os.makedirs(save_dir, exist_ok=True)
    best_reward = float("-inf")
    rng = np.random.default_rng(args.seed + 9999)
    lambda_fail = float(args.lambda_fail_init)

    for update_step in tqdm(range(args.num_updates)):
        envs = None
        try:
            train_records = records + self_generated_buffer
            envs, records_meta, group_meta = _make_incumbent_train_envs(
                args=args,
                config=config,
                perturb_dict=perturb_dict,
                incumbent_records=train_records,
                lambda_fail=lambda_fail,
                update_step=update_step,
                rng=rng,
            )
            initial_obs, meta = _reset_envs_for_rollout(envs, records_meta, group_meta, args, rng)
            rollout = _student_rollout(
                agent=agent,
                envs=envs,
                args=args,
                device=device,
                num_steps=args.num_steps,
                initial_obs=initial_obs,
                rollout_meta=meta,
            )
            _cache_student_sequences(records_meta, rollout, args)

            _best, fr = _best_objectives(rollout)
            success_rate = float(np.mean(fr)) if len(fr) else 0.0
            args._current_update_step = int(update_step)
            if success_rate >= float(args.adv_fr_threshold):
                args._finish_rate_streak = int(getattr(args, "_finish_rate_streak", 0)) + 1
            else:
                args._finish_rate_streak = 0
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
            aux_info = _incumbent_aux_update(
                agent=agent,
                optim_backbone=optim_backbone,
                args=args,
                config=config,
                records_meta=records_meta,
                device=device,
                update_step=update_step,
            )
            archive_info = _update_incumbents_from_rollout(
                records_meta=records_meta,
                envs=envs,
                rollout=rollout,
                args=args,
                online_buffer=self_generated_buffer,
            )
            regret_info = _recompute_regret_probe(
                agent=agent,
                args=args,
                config=config,
                records=records + self_generated_buffer,
                device=device,
                update_step=update_step,
                rng=rng,
            )

            if args.debug or update_step % args.log_freq == 0:
                _print_update_summary(
                    update_step,
                    args,
                    rollout,
                    ppo_info,
                    aux_info,
                    archive_info,
                    regret_info,
                    group_meta,
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

            milestones = {int(x.strip()) for x in str(args.checkpoint_milestones).split(",") if x.strip()}
            if update_step + 1 in milestones:
                torch.save(agent.state_dict(), os.path.join(save_dir, f"model_update{update_step + 1:04d}.pth"))
        finally:
            if envs is not None:
                envs.close()

    torch.save(agent.state_dict(), os.path.join(save_dir, "cur_model.pth"))
    test_envs.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incumbent-buffer offline/online PPO for EVRPTW."
    )

    parser.add_argument("--exp-name", type=str, default="incumbent_strict")
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
    parser.add_argument("--use-decomposed-reward-adv", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--decomposed-reward-mode", type=str, default="objective_progress", choices=["objective_progress", "objective_progress_only"])
    parser.add_argument("--adv-objective-weight", type=float, default=0.5)
    parser.add_argument("--adv-progress-weight", type=float, default=0.5)
    parser.add_argument("--adv-weight-schedule", type=str, default="staged", choices=["fixed", "staged", "epoch_30_70", "updates_30_70", "finish_rate"])
    parser.add_argument("--adv-mid-update", type=int, default=30)
    parser.add_argument("--adv-late-update", type=int, default=70)
    parser.add_argument("--adv-early-obj-weight", type=float, default=0.5)
    parser.add_argument("--adv-early-progress-weight", type=float, default=0.5)
    parser.add_argument("--adv-mid-obj-weight", type=float, default=0.7)
    parser.add_argument("--adv-mid-progress-weight", type=float, default=0.3)
    parser.add_argument("--adv-late-obj-weight", type=float, default=0.85)
    parser.add_argument("--adv-late-progress-weight", type=float, default=0.15)
    parser.add_argument("--adv-fr-threshold", type=float, default=0.98)
    parser.add_argument("--adv-fr-patience", type=int, default=3)
    parser.add_argument("--success-gated-progress", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--success-obj-weight", type=float, default=0.85)
    parser.add_argument("--failure-obj-weight", type=float, default=0.30)
    parser.add_argument("--failure-progress-weight", type=float, default=0.70)
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

    parser.add_argument("--max-incumbent-records", type=int, default=0)
    parser.add_argument("--incumbent-ratio", type=float, default=0.2)
    parser.add_argument("--buffer-normal-frac", type=float, default=1.0)
    parser.add_argument("--buffer-temp-frac", type=float, default=0.0)
    parser.add_argument("--buffer-prefix-frac", type=float, default=0.0)
    parser.add_argument("--online-normal-frac", type=float, default=1.0)
    parser.add_argument("--online-temp-frac", type=float, default=0.0)
    parser.add_argument("--temperature-sampling", type=float, default=1.0)
    parser.add_argument("--use-regret-aware-sampler", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--buffer-regret-kappa", type=float, default=25.0)
    parser.add_argument("--buffer-regret-relative", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--buffer-regret-margin", type=float, default=0.0)
    parser.add_argument("--buffer-unknown-weight", type=float, default=1.0)
    parser.add_argument("--buffer-uncertain-weight", type=float, default=0.5)
    parser.add_argument("--buffer-ppo-win-weight", type=float, default=0.05)
    parser.add_argument("--buffer-lucky-ppo-weight", type=float, default=0.75)
    parser.add_argument("--buffer-alns-win-base-weight", type=float, default=1.0)
    parser.add_argument("--use-pomo50-regret-state", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--pomo50-stable-beat-rate", type=float, default=0.20)
    parser.add_argument("--pomo50-low-beat-rate", type=float, default=0.05)
    parser.add_argument("--pomo50-regret-margin-rel", type=float, default=0.01)
    parser.add_argument("--pomo50-regret-margin-abs", type=float, default=0.0)
    parser.add_argument("--incumbent-update-eps", type=float, default=1e-6)
    parser.add_argument("--use-self-generated-buffer", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--self-buffer-max-records", type=int, default=1024)
    parser.add_argument("--online-elite-quantile", type=float, default=0.20)
    parser.add_argument("--online-elite-max-obj", type=float, default=float("inf"))
    parser.add_argument("--online-elite-add-per-update", type=int, default=16)
    parser.add_argument("--regret-recompute-freq", type=int, default=20)
    parser.add_argument("--regret-probe-size", type=int, default=128)

    parser.add_argument("--use-route-preference", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--route-pref-coef", type=float, default=0.02)
    parser.add_argument("--route-pref-beta", type=float, default=2.0)
    parser.add_argument("--use-selective-bc", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--selective-bc-coef", type=float, default=0.02)
    parser.add_argument("--selective-bc-margin", type=float, default=0.0)
    parser.add_argument("--aux-batch-size", type=int, default=32)
    parser.add_argument("--aux-max-steps", type=int, default=75)

    parser.add_argument("--use-expert-prefix", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--expert-prefix-prob", type=float, default=0.0)
    parser.add_argument("--expert-prefix-remaining-min", type=float, default=0.10)
    parser.add_argument("--expert-prefix-remaining-max", type=float, default=0.30)

    parser.add_argument("--group-adv-coef", type=float, default=0.0)
    parser.add_argument("--group-adv-clip", type=float, default=3.0)
    parser.add_argument("--group-adv-mode", type=str, default="zscore", choices=["zscore", "std_floor", "rank", "top_bottom"])
    parser.add_argument("--group-adv-std-floor", type=float, default=0.0)
    parser.add_argument("--group-adv-top-frac", type=float, default=0.25)
    parser.add_argument("--use-variance-group-gate", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--variance-gate-vmin", type=float, default=0.005)
    parser.add_argument("--variance-gate-vmax", type=float, default=0.030)
    parser.add_argument("--use-decision-fork-gate", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--decision-fork-gate-mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--decision-fork-min-actions", type=int, default=4)
    parser.add_argument("--decision-fork-max-actions", type=int, default=12)
    parser.add_argument("--rank-adv-coef", type=float, default=0.0)
    parser.add_argument("--rank-adv-clip", type=float, default=3.0)
    parser.add_argument("--reference-adv-coef", type=float, default=0.0)
    parser.add_argument("--reference-adv-rho", type=float, default=0.05)
    parser.add_argument("--reference-adv-clip", type=float, default=2.0)
    parser.add_argument("--reference-adv-alns-win-only", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--reference-adv-gate-mode", type=str, default="fixed", choices=["fixed", "linear", "hard"])
    parser.add_argument("--reference-adv-gate-temp", type=float, default=0.05)
    parser.add_argument("--reference-adv-hard-threshold", type=float, default=0.03)
    parser.add_argument("--use-route-level-loss", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--route-loss-coef", type=float, default=0.0)
    parser.add_argument("--route-loss-warmup-updates", type=int, default=0)
    parser.add_argument("--route-clip-eps", type=float, default=0.2)
    parser.add_argument("--route-adv-source", type=str, default="group_ref", choices=["group", "ref", "group_ref", "ref_group"])
    parser.add_argument("--route-adv-std-floor", type=float, default=0.0)
    parser.add_argument("--only-success-route-loss", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--route-mask-mode", type=str, default="all", choices=["all", "positive", "elite", "positive_elite", "elite_positive"])
    parser.add_argument("--route-elite-frac", type=float, default=0.25)
    parser.add_argument("--route-positive-eps", type=float, default=0.0)
    parser.add_argument("--cmp-adv-coef", type=float, default=0.0)
    parser.add_argument("--cmp-adv-clip", type=float, default=3.0)
    parser.add_argument("--cmp-adv-mode", type=str, default="raw", choices=["raw", "gap_temp"])
    parser.add_argument("--cmp-gap-temp", type=float, default=0.05)
    parser.add_argument("--cmp-raw-clip", type=float, default=3.0)
    parser.add_argument("--adv-diag", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--grad-cos-diag", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--grad-cos-freq", type=int, default=10)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)

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
    if not (0.0 <= float(args.incumbent_ratio) <= 1.0):
        raise ValueError("incumbent_ratio must be in [0, 1].")
    if args.temperature_sampling <= 0:
        raise ValueError("temperature_sampling must be positive.")
    if args.buffer_regret_kappa < 0:
        raise ValueError("buffer_regret_kappa must be non-negative.")
    for name in ("buffer_unknown_weight", "buffer_uncertain_weight", "buffer_ppo_win_weight", "buffer_lucky_ppo_weight", "buffer_alns_win_base_weight"):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if not (0.0 <= float(args.pomo50_stable_beat_rate) <= 1.0) or not (0.0 <= float(args.pomo50_low_beat_rate) <= 1.0):
        raise ValueError("pomo50 beat-rate thresholds must be in [0, 1].")
    if float(args.pomo50_low_beat_rate) > float(args.pomo50_stable_beat_rate):
        raise ValueError("pomo50_low_beat_rate cannot exceed pomo50_stable_beat_rate.")
    if float(args.pomo50_regret_margin_rel) < 0.0 or float(args.pomo50_regret_margin_abs) < 0.0:
        raise ValueError("pomo50 regret margins must be non-negative.")
    if not (0.0 <= args.online_elite_quantile <= 1.0):
        raise ValueError("online_elite_quantile must be in [0, 1].")
    if args.aux_batch_size <= 0 or args.aux_max_steps <= 0:
        raise ValueError("aux_batch_size and aux_max_steps must be positive.")
    if args.adv_mid_update < 0 or args.adv_late_update <= args.adv_mid_update:
        raise ValueError("adv_late_update must be greater than adv_mid_update >= 0.")
    if args.adv_fr_patience <= 0:
        raise ValueError("adv_fr_patience must be positive.")
    if args.cmp_gap_temp <= 0 or args.cmp_adv_clip <= 0 or args.cmp_raw_clip <= 0:
        raise ValueError("cmp_gap_temp, cmp_adv_clip and cmp_raw_clip must be positive.")
    if args.group_adv_clip <= 0 or args.reference_adv_rho <= 0 or args.reference_adv_clip <= 0:
        raise ValueError("group_adv_clip, reference_adv_rho and reference_adv_clip must be positive.")
    if args.group_adv_std_floor < 0 or not (0.0 < args.group_adv_top_frac <= 0.5):
        raise ValueError("group_adv_std_floor must be non-negative and group_adv_top_frac must be in (0, 0.5].")
    if args.reference_adv_gate_temp <= 0 or args.reference_adv_hard_threshold < 0:
        raise ValueError("reference_adv_gate_temp must be positive and reference_adv_hard_threshold must be non-negative.")
    if args.route_loss_coef < 0 or args.route_loss_warmup_updates < 0:
        raise ValueError("route_loss_coef and route_loss_warmup_updates must be non-negative.")
    if args.route_clip_eps <= 0 or args.route_adv_std_floor < 0:
        raise ValueError("route_clip_eps must be positive and route_adv_std_floor must be non-negative.")
    if args.variance_gate_vmin < 0 or args.variance_gate_vmax <= args.variance_gate_vmin:
        raise ValueError("variance_gate_vmax must be greater than variance_gate_vmin >= 0.")
    if args.decision_fork_min_actions < 1 or args.decision_fork_max_actions <= args.decision_fork_min_actions:
        raise ValueError("decision_fork_max_actions must be greater than decision_fork_min_actions >= 1.")
    if not (0.0 < args.route_elite_frac <= 1.0) or args.route_positive_eps < 0:
        raise ValueError("route_elite_frac must be in (0, 1] and route_positive_eps must be non-negative.")
    if args.grad_cos_freq <= 0:
        raise ValueError("grad_cos_freq must be positive.")
    for name in (
        "adv_objective_weight", "adv_progress_weight",
        "adv_early_obj_weight", "adv_early_progress_weight",
        "adv_mid_obj_weight", "adv_mid_progress_weight",
        "adv_late_obj_weight", "adv_late_progress_weight",
        "success_obj_weight", "failure_obj_weight", "failure_progress_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if args.num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive.")
    if args.expert_prefix_remaining_min < 0 or args.expert_prefix_remaining_max > 1:
        raise ValueError("expert prefix remaining ratios must be within [0, 1].")
    if args.expert_prefix_remaining_min > args.expert_prefix_remaining_max:
        raise ValueError("expert_prefix_remaining_min cannot exceed max.")
    if args.rollout_temperature <= 0:
        raise ValueError("rollout_temperature must be positive.")

    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
