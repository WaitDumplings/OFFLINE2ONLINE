import os
import random
import time
import pickle
import warnings
from copy import deepcopy
from itertools import product

import numpy as np
from tqdm import tqdm

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from evrptw_gen.configs.load_config import Config
from evrptw_gen.benchmarks.DRL_Solver.models.graph_attention_model_wrapper import Agent
from evrptw_gen.benchmarks.DRL_Solver.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.benchmarks.DRL_Solver.utils.utils import (
    update_lambda_fail,
    _mean,
    _max,
    _min,
    _p90,
    _p10,
    make_env,
    grad_norm,
)

warnings.filterwarnings(
    "ignore",
    message="WARN: A Box observation space has an unconventional shape*",
    category=UserWarning,
)


# =========================================================
# Generic helpers
# =========================================================
def _bsrs_is_enabled(args, update_step=None, enabled=True):
    if not enabled:
        return False
    if not bool(getattr(args, "use_bsrs", False)):
        return False
    if bool(getattr(args, "use_split_bsrs", False)):
        customer_eta = getattr(args, "bsrs_customer_eta", None)
        if customer_eta is None:
            customer_eta = getattr(args, "bsrs_eta", 0.0)
        rs_eta = getattr(args, "bsrs_rs_eta", 0.0)
        if float(customer_eta) == 0.0 and float(rs_eta) == 0.0:
            return False
    elif float(getattr(args, "bsrs_eta", 0.0)) == 0.0:
        return False

    warmup = int(getattr(args, "bsrs_warmup_updates", 0))
    if update_step is not None and update_step < warmup:
        return False

    return True


def _decomposed_component_names(args):
    mode = str(
        getattr(args, "decomposed_reward_mode", "objective_progress_terminal_teacher")
    ).lower()
    if mode in ("legacy", "task_terminal_teacher"):
        return ("task", "terminal", "teacher")
    if mode in ("objective_progress", "objective_progress_only"):
        return ("objective", "progress")
    return ("objective", "progress", "terminal", "teacher")


def _optional_weight(args, new_name, fallback_name, default=1.0):
    value = getattr(args, new_name, None)
    if value is not None:
        return float(value)
    return float(getattr(args, fallback_name, default))


def _adaptive_adv_mix(args, success_rate):
    if success_rate is None:
        return None
    if not bool(getattr(args, "use_adaptive_adv_weights", False)):
        return None

    threshold = float(getattr(args, "adv_adapt_success_threshold", 0.85))
    temperature = max(float(getattr(args, "adv_adapt_success_temperature", 0.05)), 1e-8)
    x = np.clip((float(success_rate) - threshold) / temperature, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-x)))


def _adaptive_component_weight(args, name, stage):
    default_by_stage = {
        "early": {
            "objective": 0.20,
            "progress": 0.50,
            "task": 0.50,
            "terminal": 0.20,
            "teacher": 0.10,
        },
        "late": {
            "objective": 0.65,
            "progress": 0.20,
            "task": 0.20,
            "terminal": 0.10,
            "teacher": 0.05,
        },
    }
    attr_name = f"adv_{name}_{stage}_weight"
    value = getattr(args, attr_name, None)
    if value is not None:
        return float(value)
    return float(default_by_stage[stage].get(name, 1.0))


def _decomposed_component_weights(args, names, device, dtype=torch.float32, success_rate=None):
    mix = _adaptive_adv_mix(args, success_rate)
    weights = []
    for name in names:
        if mix is not None:
            early = _adaptive_component_weight(args, name, "early")
            late = _adaptive_component_weight(args, name, "late")
            weight = (1.0 - mix) * early + mix * late
        elif name == "objective":
            weight = _optional_weight(args, "adv_objective_weight", "adv_task_weight")
        elif name == "progress":
            weight = _optional_weight(args, "adv_progress_weight", "adv_task_weight")
        elif name == "task":
            weight = float(getattr(args, "adv_task_weight", 1.0))
        elif name == "terminal":
            weight = float(getattr(args, "adv_terminal_weight", 1.0))
        elif name == "teacher":
            weight = float(getattr(args, "adv_teacher_weight", 1.0))
        else:
            weight = 1.0
        weights.append(weight)

    return torch.tensor(weights, device=device, dtype=dtype)


def _bsrs_scalar_potential(value, args, mode=None):
    """
    Convert critic output to a scalar BSRS potential.

    For a single-head critic, this is just V(s).  For the decomposed critic,
    use a configurable scalarization so BSRS remains compatible with both
    training modes.
    """
    if value.dim() >= 1 and value.size(-1) == 1:
        return value.squeeze(-1)

    if value.dim() < 3:
        return value

    mode = str(mode if mode is not None else getattr(args, "bsrs_value_mode", "weighted")).lower()

    if mode == "task":
        return value[..., 0]
    if mode == "objective":
        return value[..., 0]
    if mode == "progress":
        return value[..., min(1, value.size(-1) - 1)]
    if mode == "terminal":
        idx = min(2, value.size(-1) - 1) if value.size(-1) > 3 else min(1, value.size(-1) - 1)
        return value[..., idx]
    if mode == "teacher":
        return value[..., min(value.size(-1) - 1, 3 if value.size(-1) > 3 else 2)]
    if mode == "sum":
        return value.sum(dim=-1)

    if bool(getattr(args, "use_decomposed_reward_adv", False)):
        names = _decomposed_component_names(args)
        weights = _decomposed_component_weights(args, names, value.device, value.dtype)
    else:
        weights = torch.ones(value.size(-1), device=value.device, dtype=value.dtype)

    if weights.numel() < value.size(-1):
        pad = torch.ones(
            value.size(-1) - weights.numel(),
            device=value.device,
            dtype=value.dtype,
        )
        weights = torch.cat([weights, pad], dim=0)
    weights = weights[: value.size(-1)].clamp(min=0.0)
    denom = weights.sum().clamp(min=1e-8)
    return (value * (weights / denom).view(*([1] * (value.dim() - 1)), -1)).sum(dim=-1)


def _compute_bsrs_delta(critic_value, next_value, done_tensor, args, value_mode, eta):
    if float(eta) == 0.0:
        return torch.zeros(done_tensor.shape, device=done_tensor.device, dtype=critic_value.dtype)

    phi_s = _bsrs_scalar_potential(critic_value, args, mode=value_mode)
    phi_sp = _bsrs_scalar_potential(next_value, args, mode=value_mode)
    phi_sp = torch.where(done_tensor, torch.zeros_like(phi_sp), phi_sp)

    shaped = float(eta) * (float(args.gamma) * phi_sp - phi_s)

    clip_value = float(getattr(args, "bsrs_clip", 0.0))
    if clip_value > 0.0:
        shaped = torch.clamp(shaped, -clip_value, clip_value)

    return shaped


def _compute_bsrs_reward(agent, obs, encoder_state, critic_value, done_tensor, args, action=None):
    next_value = agent.get_value_cached(obs, encoder_state)
    next_value = next_value.reshape(done_tensor.shape[0], done_tensor.shape[1], -1)
    critic_value = critic_value.reshape(done_tensor.shape[0], done_tensor.shape[1], -1)

    if bool(getattr(args, "use_split_bsrs", False)):
        customer_eta = getattr(args, "bsrs_customer_eta", None)
        if customer_eta is None:
            customer_eta = getattr(args, "bsrs_eta", 0.0)
        customer_reward = _compute_bsrs_delta(
            critic_value=critic_value,
            next_value=next_value,
            done_tensor=done_tensor,
            args=args,
            value_mode=getattr(args, "bsrs_customer_value_mode", "progress"),
            eta=customer_eta,
        )
        rs_reward = _compute_bsrs_delta(
            critic_value=critic_value,
            next_value=next_value,
            done_tensor=done_tensor,
            args=args,
            value_mode=getattr(args, "bsrs_rs_value_mode", "terminal"),
            eta=getattr(args, "bsrs_rs_eta", 0.0),
        )
        if bool(getattr(args, "bsrs_rs_negative_only", False)):
            rs_reward = torch.clamp(rs_reward, max=0.0)

        zeros = torch.zeros_like(customer_reward)
        if action is None:
            customer_mask = torch.ones_like(done_tensor, dtype=torch.bool)
            rs_mask = torch.zeros_like(done_tensor, dtype=torch.bool)
        elif isinstance(action, torch.Tensor):
            action_tensor = action.to(device=done_tensor.device, dtype=torch.long)
            customer_start = 1
            customer_end = customer_start + int(getattr(args, "train_cus_num", 0))
            rs_start = customer_end
            rs_end = rs_start + int(getattr(args, "train_cs_num", 0))
            customer_mask = (action_tensor >= customer_start) & (action_tensor < customer_end)
            rs_mask = (action_tensor >= rs_start) & (action_tensor < rs_end)
        else:
            action_tensor = torch.tensor(action, device=done_tensor.device, dtype=torch.long)
            customer_start = 1
            customer_end = customer_start + int(getattr(args, "train_cus_num", 0))
            rs_start = customer_end
            rs_end = rs_start + int(getattr(args, "train_cs_num", 0))
            customer_mask = (action_tensor >= customer_start) & (action_tensor < customer_end)
            rs_mask = (action_tensor >= rs_start) & (action_tensor < rs_end)

        customer_reward = torch.where(customer_mask, customer_reward, zeros)
        rs_reward = torch.where(rs_mask, rs_reward, zeros)
        total_reward = customer_reward + rs_reward
        return {
            "total": total_reward.detach(),
            "customer": customer_reward.detach(),
            "rs": rs_reward.detach(),
        }

    shaped = _compute_bsrs_delta(
        critic_value=critic_value,
        next_value=next_value,
        done_tensor=done_tensor,
        args=args,
        value_mode=getattr(args, "bsrs_value_mode", "weighted"),
        eta=getattr(args, "bsrs_eta", 0.0),
    )
    return {"total": shaped.detach()}


def _set_global_seeds(seed, deterministic=True):
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def _unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_state_dict_compatible(module, state_dict, strict=True):
    if strict:
        module.load_state_dict(state_dict)
        return

    current = module.state_dict()
    compatible = {
        k: v
        for k, v in state_dict.items()
        if k in current and tuple(v.shape) == tuple(current[k].shape)
    }
    skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
    current.update(compatible)
    module.load_state_dict(current)
    if len(skipped) > 0:
        print(
            f"[Init] Loaded compatible checkpoint tensors; skipped "
            f"{len(skipped)} shape-mismatched/missing tensors."
        )


def _set_attr_all(envs, name, value):
    if hasattr(envs, "update_attr"):
        try:
            envs.update_attr(name, value)
            return
        except Exception:
            pass

    for env in envs.envs:
        base_env = _unwrap_env(env)
        setattr(base_env, name, value)


def _stack_obs_list(obs_list):
    if len(obs_list) == 0:
        raise ValueError("obs_list is empty.")

    keys = obs_list[0].keys()
    stacked = {}

    for k in keys:
        stacked[k] = np.stack([obs[k] for obs in obs_list], axis=0)

    return stacked


def _reset_same_instance_all(envs, keep_teacher=True):
    obs_list = []

    for env in envs.envs:
        base_env = _unwrap_env(env)

        if not hasattr(base_env, "reset_same_instance"):
            raise AttributeError(
                "Base env does not implement reset_same_instance(...). "
                "Please update EVRPTWVectorEnv first."
            )

        obs_list.append(base_env.reset_same_instance(keep_teacher=keep_teacher))

    return _stack_obs_list(obs_list)


def _call_set_teacher_reference_each(
    envs,
    teacher_objs,
    teacher_sources,
    teacher_stages,
    enabled_flags,
    teacher_raw_objs=None,
):
    num_envs = len(envs.envs)

    if teacher_raw_objs is None:
        teacher_raw_objs = [None] * num_envs

    if not (
        len(teacher_objs)
        == len(teacher_sources)
        == len(teacher_stages)
        == len(enabled_flags)
        == len(teacher_raw_objs)
        == num_envs
    ):
        raise ValueError("Teacher reference lists must match num_envs.")

    for i, env in enumerate(envs.envs):
        base_env = _unwrap_env(env)

        if hasattr(base_env, "set_teacher_reference"):
            base_env.set_teacher_reference(
                teacher_obj=teacher_objs[i],
                source=teacher_sources[i],
                stage=teacher_stages[i],
                enabled=enabled_flags[i],
                raw_obj=teacher_raw_objs[i],
            )
        else:
            base_env.teacher_obj = teacher_objs[i]
            base_env.teacher_source = teacher_sources[i]
            base_env.teacher_stage = teacher_stages[i]
            base_env.teacher_raw_obj = teacher_raw_objs[i]
            base_env.use_teacher_reward = bool(
                enabled_flags[i] and teacher_objs[i] is not None
            )


def _extract_objectives_from_info(info, n_traj, device):
    if isinstance(info, (list, tuple)):
        obj_list = []

        for item in info:
            if isinstance(item, dict) and ("objective" in item):
                obj = np.asarray(item["objective"], dtype=np.float32).reshape(-1)

                if obj.shape[0] != n_traj:
                    obj = np.full((n_traj,), np.inf, dtype=np.float32)

                obj_list.append(obj)
            else:
                obj_list.append(np.full((n_traj,), np.inf, dtype=np.float32))

        return torch.tensor(
            np.stack(obj_list, axis=0),
            device=device,
            dtype=torch.float32,
        )

    if isinstance(info, dict) and ("objective" in info):
        obj = np.asarray(info["objective"], dtype=np.float32)
        return torch.tensor(obj, device=device, dtype=torch.float32)

    return None


def _extract_info_stack(info, key):
    if not isinstance(info, (list, tuple)) or len(info) == 0:
        return None

    if key not in info[0]:
        return None

    try:
        return np.stack([x[key] for x in info], axis=0)
    except Exception:
        return None


def _extract_reward_component_stack(info, key):
    if not isinstance(info, (list, tuple)) or len(info) == 0:
        return None

    try:
        return np.stack(
            [
                np.asarray(x["reward_components"][key], dtype=np.float32)
                for x in info
            ],
            axis=0,
        )
    except Exception:
        return None


# =========================================================
# ALNS buffer helpers
# =========================================================
def _stem(x):
    if x is None:
        return None

    x = str(x).strip().strip("/")
    if len(x) == 0:
        return None

    return os.path.splitext(os.path.basename(x))[0]


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_register_instance_alias(index, alias, instance):
    if alias is None:
        return

    alias_str = str(alias).strip().strip("/")
    if len(alias_str) == 0:
        return

    index[alias_str] = instance

    alias_stem = _stem(alias_str)
    if alias_stem is not None:
        index[alias_stem] = instance


def _build_instance_index(instance_data):
    index = {}

    def add_instance(key, inst):
        if not isinstance(inst, dict):
            return

        _safe_register_instance_alias(index, key, inst)

        for k in [
            "instance_id",
            "id",
            "name",
            "file",
            "filename",
            "path",
            "instance_path",
            "pickle_path",
            "pkl_path",
            "txt_path",
        ]:
            if k in inst:
                _safe_register_instance_alias(index, inst.get(k), inst)

        for outer_key in ["metadata", "meta", "info", "env"]:
            obj = inst.get(outer_key, None)

            if isinstance(obj, dict):
                for k in [
                    "instance_id",
                    "id",
                    "name",
                    "file",
                    "filename",
                    "path",
                    "instance_path",
                    "pickle_path",
                    "pkl_path",
                    "txt_path",
                ]:
                    if k in obj:
                        _safe_register_instance_alias(index, obj.get(k), inst)

    if isinstance(instance_data, list):
        for i, inst in enumerate(instance_data):
            add_instance(i, inst)

    elif isinstance(instance_data, dict):
        for wrapper_key in ["instances", "data", "dataset"]:
            if wrapper_key in instance_data:
                return _build_instance_index(instance_data[wrapper_key])

        for key, inst in instance_data.items():
            add_instance(key, inst)

    else:
        raise ValueError(f"Unsupported instance pickle format: {type(instance_data)}")

    return index


def _norm_alns_obj(obj, scale=100.0):
    scale = max(float(scale), 1e-8)
    return float(obj) / scale


def _load_alns_buffer_from_dir(
    buffer_dir,
    progress_name="buffer_progress.pkl",
    instance_pickle_name="evrptw_50C_12R.pkl",
    obj_scale=100.0,
):
    """
    Load ALNS teacher records and inject teacher information directly into instance.

    The env should automatically read:
        instance["teacher"]["obj"]
    """
    if buffer_dir is None:
        print("[ALNSBuffer] buffer_dir is None.")
        return []

    progress_path = os.path.join(buffer_dir, "progress", progress_name)
    instance_pickle_path = os.path.join(buffer_dir, "pickle", instance_pickle_name)

    if not os.path.exists(progress_path):
        print(f"[ALNSBuffer] progress file not found: {progress_path}")
        return []

    if not os.path.exists(instance_pickle_path):
        print(f"[ALNSBuffer] instance pickle not found: {instance_pickle_path}")
        return []

    progress_raw = _load_pickle(progress_path)
    instance_data = _load_pickle(instance_pickle_path)

    if not isinstance(progress_raw, dict):
        raise ValueError(f"ALNS progress must be dict, got {type(progress_raw)}")

    instance_index = _build_instance_index(instance_data)

    records = []
    invalid_count = 0
    miss_count = 0

    for key, value in progress_raw.items():
        if not isinstance(value, dict):
            invalid_count += 1
            continue

        obj = value.get("global_value", None)

        try:
            raw_obj = float(obj)
        except Exception:
            invalid_count += 1
            continue

        if not np.isfinite(raw_obj) or raw_obj <= 0:
            invalid_count += 1
            continue

        feasible = bool(value.get("visited_all", False))
        if not feasible:
            invalid_count += 1
            continue

        file_name = value.get("file", key)
        instance_id = value.get("instance_id", None)

        candidate_aliases = [
            instance_id,
            file_name,
            key,
            _stem(instance_id),
            _stem(file_name),
            _stem(key),
        ]

        matched_instance = None
        matched_key = None

        for alias in candidate_aliases:
            if alias is None:
                continue

            alias_str = str(alias).strip().strip("/")

            if alias_str in instance_index:
                matched_instance = instance_index[alias_str]
                matched_key = alias_str
                break

            alias_stem = _stem(alias_str)

            if alias_stem is not None and alias_stem in instance_index:
                matched_instance = instance_index[alias_stem]
                matched_key = alias_stem
                break

        if matched_instance is None:
            miss_count += 1
            continue

        teacher_obj = _norm_alns_obj(raw_obj, scale=obj_scale)
        instance = deepcopy(matched_instance)

        instance.setdefault(
            "instance_id",
            _stem(instance_id) or _stem(file_name) or _stem(key),
        )
        instance.setdefault("file", file_name)
        instance.setdefault("alns_buffer_key", key)
        instance.setdefault("alns_matched_key", matched_key)

        instance["teacher"] = {
            "obj": float(teacher_obj),
            "source": "alns",
            "stage": "offline",
            "raw_obj": float(raw_obj),
        }

        record = {
            "instance": instance,
            "teacher_obj": float(teacher_obj),
            "raw_teacher_obj": float(raw_obj),
            "best_routes": value.get("best_routes", None),
            "current_routes": value.get("current_routes", None),
            "instance_id": instance["instance_id"],
            "file": file_name,
            "key": key,
            "matched_key": matched_key,
            "cur_iter": value.get("cur_iter", None),
            "max_iters": value.get("max_iters", None),
            "runtime_sec": value.get("elapsed_time_s_last_run", None),
            "raw": value,
        }

        records.append(record)

    print(
        f"[ALNSBuffer] loaded | "
        f"progress={len(progress_raw)} | "
        f"instance_index={len(instance_index)} | "
        f"valid_records={len(records)} | "
        f"miss={miss_count} | "
        f"invalid={invalid_count}"
    )

    return records


def _sample_records(records, batch_size, rng):
    if records is None or len(records) == 0:
        return []

    n = len(records)

    if batch_size <= n:
        idx = rng.choice(n, size=batch_size, replace=False)
    else:
        idx = rng.choice(n, size=batch_size, replace=True)

    return [records[int(i)] for i in idx]


# =========================================================
# Env builders
# =========================================================
def _teacher_reward_cfg(args, use_teacher_reward):
    return {
        "reward_mode": str(getattr(args, "reward_mode", "legacy")),
        "max_route_events": int(getattr(args, "max_route_events", 16)),
        "max_completed_routes": int(getattr(args, "max_completed_routes", 4)),
        "use_teacher_reward": bool(use_teacher_reward),
        "use_time_energy_pbrs": bool(
            getattr(args, "use_time_energy_pbrs", False)
        ),
        "time_energy_pbrs_coef": float(
            getattr(args, "time_energy_pbrs_coef", 0.0)
        ),
        "time_energy_pbrs_clip": float(
            getattr(args, "time_energy_pbrs_clip", 0.05)
        ),
        "use_direct_progress_pbrs": bool(
            getattr(args, "use_direct_progress_pbrs", False)
        ),
        "progress_pbrs_coef": float(
            getattr(args, "progress_pbrs_coef", 1.0)
        ),
        "progress_pbrs_beta": float(
            getattr(args, "progress_pbrs_beta", 0.5)
        ),
        "use_repair_fail_reward": bool(
            getattr(args, "use_repair_fail_reward", False)
        ),
        "repair_fail_coef": float(getattr(args, "repair_fail_coef", 1.0)),
        "repair_progress_coef": float(
            getattr(args, "repair_progress_coef", 1.0)
        ),
        "repair_success_bonus": float(
            getattr(args, "repair_success_bonus", 1.0)
        ),

        # Reward mode interface.
        "teacher_reward_mode": args.teacher_reward_mode,
        "plain_success_bonus_coef": args.plain_success_bonus_coef,

        # Additive teacher reward params.
        "teacher_penalty_ratio": args.teacher_penalty_ratio,
        "teacher_bonus_ratio": args.teacher_bonus_ratio,
        "teacher_failure_ratio": args.teacher_failure_ratio,
        "teacher_reward_temp": args.teacher_reward_temp,
        "teacher_reward_margin": args.teacher_reward_margin,
        "teacher_closure_failure_min_ratio": args.teacher_closure_failure_min_ratio,

        # Scaled-success params.
        "teacher_success_coef_type": args.teacher_success_coef_type,
        "teacher_success_worse_floor": args.teacher_success_worse_floor,
        "teacher_success_equal_coef": args.teacher_success_equal_coef,
        "teacher_success_max_coef": args.teacher_success_max_coef,
        "teacher_success_improve_scale": args.teacher_success_improve_scale,
        "teacher_success_target_improve": args.teacher_success_target_improve,
        "teacher_success_power": args.teacher_success_power,
        "teacher_success_worse_temp": args.teacher_success_worse_temp,
        "teacher_scaled_failure": args.teacher_scaled_failure,
        "use_teacher_dense_reward": bool(
            getattr(args, "use_alns_dense_teacher", False)
        ),
        "teacher_dense_coef": float(
            getattr(args, "alns_dense_teacher_coef", 0.0)
        ),
        "teacher_dense_mode": str(
            getattr(args, "alns_dense_teacher_mode", "exact")
        ),
        "teacher_dense_lookahead": int(
            getattr(args, "alns_dense_teacher_lookahead", 1)
        ),
        "teacher_dense_rank_tau": float(
            getattr(args, "alns_dense_teacher_rank_tau", 2.0)
        ),
        "use_teacher_route_potential": bool(
            getattr(args, "use_alns_route_potential", False)
        ),
        "teacher_route_potential_coef": float(
            getattr(args, "alns_route_potential_coef", 0.0)
        ),
        "teacher_route_potential_clip": float(
            getattr(args, "alns_route_potential_clip", 0.0)
        ),
        "teacher_gap_worse_only": bool(
            getattr(args, "teacher_gap_worse_only", False)
        ),
    }


def _csv_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _parse_fixed_overrides(value):
    overrides = {}
    for item in _csv_list(value):
        if "=" not in item:
            raise ValueError(
                f"Invalid train config fixed override '{item}'. "
                "Expected key=value."
            )
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key or not val:
            raise ValueError(
                f"Invalid train config fixed override '{item}'. "
                "Expected non-empty key=value."
            )
        overrides[key] = val
    return overrides


def _set_config_choice(config, key, value):
    if key in ("instance_type", "customer_type"):
        config.data["test_instance_type"] = value
    else:
        config.data[key] = value


def _policy_options_from_config(config, policy_key):
    data = getattr(config, "data", None)
    if not isinstance(data, dict):
        raise ValueError("train config stratification expects a Config object.")

    if policy_key in ("instance_type", "customer_type"):
        dist = data.get("instance_type_distribution", None)
        if not isinstance(dist, dict) or len(dist) == 0:
            raise ValueError(
                "instance_type stratification expects instance_type_distribution in config."
            )
        options = [
            str(key)
            for key, score in dist.items()
            if float(score) > 0
        ]
        if not options:
            raise ValueError("No positive-probability instance_type options found.")
        return options

    list_key = f"{policy_key}_list"
    cfg_key = f"{policy_key}_config"
    if list_key not in data or cfg_key not in data:
        raise ValueError(
            f"Unknown train config stratify key '{policy_key}'. "
            f"Expected '{list_key}' and '{cfg_key}' in config."
        )

    policy_cfg = data.get(cfg_key) or {}
    options = []
    for option in data.get(list_key) or []:
        option_cfg = policy_cfg.get(option, {})
        score = float(option_cfg.get("score", 1.0))
        if score > 0:
            options.append(option)

    if not options:
        raise ValueError(
            f"No positive-score options found for train config key '{policy_key}'."
        )
    return options


def _train_config_combos(args, config):
    cached = getattr(args, "_train_config_combos", None)
    if cached is not None:
        return cached

    keys = _csv_list(getattr(args, "train_config_stratify_keys", ""))
    fixed_overrides = _parse_fixed_overrides(
        getattr(args, "train_config_fixed_overrides", "")
    )
    if not keys:
        combos = []
    else:
        option_groups = [_policy_options_from_config(config, key) for key in keys]
        combos = [dict(zip(keys, values)) for values in product(*option_groups)]
        if bool(getattr(args, "train_config_shuffle_combos", False)):
            rng_seed = int(getattr(args, "seed", 0)) + int(
                getattr(args, "train_config_shuffle_seed", 0)
            )
            order = np.random.default_rng(rng_seed).permutation(len(combos))
            combos = [combos[int(i)] for i in order]

    args._train_config_keys = keys
    args._train_config_combos = combos
    args._train_config_fixed_overrides = fixed_overrides

    if bool(getattr(args, "train_config_log", False)):
        if combos:
            print(
                f"[ConfigSchedule] keys={keys} | combos={len(combos)} | "
                f"mode={getattr(args, 'train_config_schedule', 'mixed')} | "
                f"fixed={fixed_overrides}"
            )
        else:
            print(
                "[ConfigSchedule] no stratify keys; "
                f"using mixed config sampling. fixed={fixed_overrides}"
            )

    return combos


def _scheduled_train_config(args, config, update_step):
    mode = str(getattr(args, "train_config_schedule", "mixed")).lower()
    fixed_overrides = getattr(args, "_train_config_fixed_overrides", None)
    if fixed_overrides is None:
        _train_config_combos(args, config)
        fixed_overrides = getattr(args, "_train_config_fixed_overrides", {})

    args._current_train_config_label = None
    args._current_train_config_idx = None
    args._current_train_config_total = None
    args._current_train_config_step = int(update_step or 0)

    if mode in ("mixed", "none", "off"):
        if not fixed_overrides:
            return config
        scheduled = deepcopy(config)
        for key, value in fixed_overrides.items():
            _set_config_choice(scheduled, key, value)
        args._current_train_config_label = ", ".join(
            f"{key}={value}" for key, value in fixed_overrides.items()
        )
        return scheduled

    combos = _train_config_combos(args, config)
    if not combos:
        if not fixed_overrides:
            return config
        scheduled = deepcopy(config)
        for key, value in fixed_overrides.items():
            _set_config_choice(scheduled, key, value)
        args._current_train_config_label = ", ".join(
            f"{key}={value}" for key, value in fixed_overrides.items()
        )
        return scheduled

    offset = int(getattr(args, "train_config_cycle_offset", 0))
    if mode == "cycle":
        combo_idx = (int(update_step or 0) + offset) % len(combos)
    elif mode == "random":
        rng_seed = int(getattr(args, "seed", 0)) + 1009 * int(update_step or 0) + offset
        combo_idx = int(np.random.default_rng(rng_seed).integers(0, len(combos)))
    else:
        raise ValueError(f"Unknown train_config_schedule: {mode}")

    combo = combos[combo_idx]
    scheduled = deepcopy(config)
    for key, value in fixed_overrides.items():
        _set_config_choice(scheduled, key, value)
    for key, value in combo.items():
        _set_config_choice(scheduled, key, value)

    label_parts = [f"{key}={value}" for key, value in combo.items()]
    label_parts.extend(f"{key}={value}" for key, value in fixed_overrides.items())
    args._current_train_config_label = ", ".join(label_parts)
    args._current_train_config_idx = combo_idx
    args._current_train_config_total = len(combos)

    if bool(getattr(args, "train_config_log", False)):
        freq = max(int(getattr(args, "train_config_log_freq", 1)), 1)
        if int(update_step or 0) % freq == 0:
            print(
                f"[ConfigSchedule] update={int(update_step or 0)} | "
                f"idx={combo_idx + 1}/{len(combos)} | "
                f"{args._current_train_config_label}"
            )

    return scheduled


def _scheduled_train_configs_for_batch(args, config, update_step, batch_size):
    mode = str(getattr(args, "train_config_schedule", "mixed")).lower()
    if mode not in ("batch_cycle", "batch_random"):
        train_config = _scheduled_train_config(args, config, update_step)
        return [train_config for _ in range(int(batch_size))]

    fixed_overrides = getattr(args, "_train_config_fixed_overrides", None)
    if fixed_overrides is None:
        _train_config_combos(args, config)
        fixed_overrides = getattr(args, "_train_config_fixed_overrides", {})

    args._current_train_config_label = None
    args._current_train_config_idx = None
    args._current_train_config_total = None
    args._current_train_config_step = int(update_step or 0)

    combos = _train_config_combos(args, config)
    if not combos:
        train_config = _scheduled_train_config(args, config, update_step)
        return [train_config for _ in range(int(batch_size))]

    batch_size = int(batch_size)
    offset = int(getattr(args, "train_config_cycle_offset", 0))
    step = int(update_step or 0)

    if mode == "batch_cycle":
        combo_indices = [
            (step + offset + env_idx) % len(combos)
            for env_idx in range(batch_size)
        ]
    else:
        rng_seed = int(getattr(args, "seed", 0)) + 1009 * step + offset
        rng = np.random.default_rng(rng_seed)
        base_order = np.arange(len(combos), dtype=np.int64)
        combo_indices = []
        while len(combo_indices) < batch_size:
            combo_indices.extend(rng.permutation(base_order).tolist())
        combo_indices = combo_indices[:batch_size]

    counts = np.bincount(np.asarray(combo_indices, dtype=np.int64), minlength=len(combos))
    train_configs = []
    for combo_idx in combo_indices:
        combo = combos[int(combo_idx)]
        scheduled = deepcopy(config)
        for key, value in fixed_overrides.items():
            _set_config_choice(scheduled, key, value)
        for key, value in combo.items():
            _set_config_choice(scheduled, key, value)
        train_configs.append(scheduled)

    combo_labels = []
    for combo_idx, combo in enumerate(combos):
        label_parts = [f"{key}={value}" for key, value in combo.items()]
        if int(counts[combo_idx]) > 0:
            combo_labels.append(f"({';'.join(label_parts)}):{int(counts[combo_idx])}")

    fixed_label = ", ".join(
        f"{key}={value}" for key, value in fixed_overrides.items()
    )
    args._current_train_config_label = (
        f"{mode}[{len(combos)}] "
        + " | ".join(combo_labels)
        + (f" | fixed={fixed_label}" if fixed_label else "")
    )
    args._current_train_config_total = len(combos)

    if bool(getattr(args, "train_config_log", False)):
        freq = max(int(getattr(args, "train_config_log_freq", 1)), 1)
        if step % freq == 0:
            print(
                f"[ConfigSchedule] update={step} | "
                f"mode={mode} | {args._current_train_config_label}"
            )

    return train_configs


def _make_generated_train_envs(
    args,
    config,
    perturb_dict,
    customer_numbers,
    charging_stations_numbers,
    lambda_fail,
    update_step=None,
):
    if bool(getattr(args, "train_config_use_online_counter", False)):
        schedule_step = int(getattr(args, "_train_config_online_step", 0))
    else:
        schedule_step = int(update_step or 0)

    train_configs = _scheduled_train_configs_for_batch(
        args,
        config,
        schedule_step,
        batch_size=args.num_envs,
    )
    if bool(getattr(args, "train_config_use_online_counter", False)):
        args._train_config_online_step = schedule_step + 1

    seed_offset = 0
    if bool(getattr(args, "train_env_seed_by_update", False)) and update_step is not None:
        seed_offset = int(update_step) * int(args.num_envs)

    return SyncVectorEnv(
        [
            make_env(
                args.env_id,
                int(args.seed + seed_offset + i),
                cfg={
                    "env_mode": "train",
                    "config": train_configs[i],
                    "n_traj": args.n_traj,
                    "num_customers": customer_numbers,
                    "num_charging_stations": charging_stations_numbers,
                    "gamma": args.gamma,
                    "lambda_fail": lambda_fail,
                    "perturb_dict": perturb_dict["perturb"],
                    **_teacher_reward_cfg(args, use_teacher_reward=False),
                },
            )
            for i in range(args.num_envs)
        ]
    )


def _make_alns_train_envs(
    args,
    config,
    update_step,
    records=None,
    alns_rng=None,
    sampled_records=None,
    use_teacher_reward=True,
    seed_offset=200000,
):
    if sampled_records is None:
        batch_size = int(getattr(args, "alns_teacher_batch_size", args.num_envs))
        sampled = _sample_records(records, batch_size=batch_size, rng=alns_rng)
    else:
        sampled = list(sampled_records)

    if len(sampled) == 0:
        return None, []

    if use_teacher_reward:
        for record in sampled:
            instance = record.get("instance", None)
            if not isinstance(instance, dict):
                continue

            teacher = instance.setdefault("teacher", {})
            teacher["gate"] = float(record.get("teacher_gate", 1.0))

            if (
                bool(getattr(args, "use_alns_dense_teacher", False))
                or bool(getattr(args, "use_alns_route_potential", False))
            ):
                action_sequence = record.get("teacher_action_sequence", None)
                if action_sequence is None:
                    action_sequence = _alns_record_to_action_sequence(
                        record,
                        num_customers=int(args.train_cus_num),
                        num_nodes=int(args.train_cus_num + args.train_cs_num + 1),
                    )
                    record["teacher_action_sequence"] = action_sequence
                teacher["action_sequence"] = list(action_sequence)

    envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                int(args.seed + seed_offset + update_step * 1000 + i),
                cfg={
                    "env_mode": "train",
                    "config": config,
                    "n_traj": args.n_traj,
                    "eval_data": deepcopy(sampled[i]["instance"]),
                    "num_customers": args.train_cus_num,
                    "num_charging_stations": args.train_cs_num,
                    "gamma": args.gamma,
                    "lambda_fail": args.lambda_fail_init,
                    **_teacher_reward_cfg(
                        args,
                        use_teacher_reward=use_teacher_reward,
                    ),
                },
            )
            for i in range(len(sampled))
        ]
    )

    return envs, sampled


def _student_rollout(
    agent,
    envs,
    args,
    device,
    num_steps,
    initial_obs=None,
    update_step=None,
    enable_bsrs=True,
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
    value_heads = int(getattr(agent, "value_heads", 1))
    if value_heads > 1:
        values = torch.zeros((num_steps, num_envs, args.n_traj, value_heads), device=device)
    else:
        values = torch.zeros((num_steps, num_envs, args.n_traj), device=device)
    valid_masks = torch.zeros(
        (num_steps, num_envs, args.n_traj),
        dtype=torch.bool,
        device=device,
    )
    reward_components = None
    if bool(getattr(args, "use_decomposed_reward_adv", False)):
        reward_components = {
            name: torch.zeros((num_steps, num_envs, args.n_traj), device=device)
            for name in _decomposed_component_names(args)
        }

    r_teacher_all = []
    r_teacher_dense_all = []
    r_teacher_route_all = []
    teacher_gap_all = []
    success_bonus_coef_all = []
    r_bsrs_all = []
    r_bsrs_customer_all = []
    r_bsrs_rs_all = []

    agent.train()
    next_obs = initial_obs if initial_obs is not None else envs.reset()

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
            action, logprob, _, value, _ = agent.get_action_and_value_cached(
                next_obs,
                state=encoder_state,
            )

            action = action.view(num_envs, args.n_traj)
            if value_heads > 1:
                values[step] = value.view(num_envs, args.n_traj, value_heads)
            else:
                values[step] = value.view(num_envs, args.n_traj)

        actions[step] = action
        logprobs[step] = logprob.view(num_envs, args.n_traj)

        if step == num_steps - 1:
            _set_attr_all(envs, "terminate", True)

        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        last_info = info

        done_tensor = torch.tensor(done, device=device, dtype=torch.bool)

        bsrs_reward = None
        bsrs_customer_reward = None
        bsrs_rs_reward = None
        if _bsrs_is_enabled(args, update_step=update_step, enabled=enable_bsrs):
            with torch.no_grad():
                bsrs_parts = _compute_bsrs_reward(
                    agent=agent,
                    obs=next_obs,
                    encoder_state=encoder_state,
                    critic_value=value,
                    done_tensor=done_tensor,
                    args=args,
                    action=action,
                )
                bsrs_reward = bsrs_parts["total"]
                bsrs_customer_reward = bsrs_parts.get("customer")
                bsrs_rs_reward = bsrs_parts.get("rs")
            r_bsrs_all.append(bsrs_reward.detach().cpu().numpy())
            if bsrs_customer_reward is not None:
                r_bsrs_customer_all.append(bsrs_customer_reward.detach().cpu().numpy())
            if bsrs_rs_reward is not None:
                r_bsrs_rs_all.append(bsrs_rs_reward.detach().cpu().numpy())

        reward_tensor = torch.tensor(
            reward,
            device=device,
            dtype=torch.float32,
        )
        if bsrs_reward is not None:
            reward_tensor = reward_tensor + bsrs_reward

        rewards[step] = reward_tensor

        if reward_components is not None:
            for component_name in reward_components:
                component = _extract_reward_component_stack(info, component_name)
                if component is None:
                    component = np.zeros((num_envs, args.n_traj), dtype=np.float32)
                component_tensor = torch.tensor(
                    component,
                    device=device,
                    dtype=torch.float32,
                )
                if bsrs_reward is not None:
                    if bool(getattr(args, "use_split_bsrs", False)):
                        if component_name in ("task", "progress") and bsrs_customer_reward is not None:
                            component_tensor = component_tensor + bsrs_customer_reward
                        elif component_name == "terminal" and bsrs_rs_reward is not None:
                            component_tensor = component_tensor + bsrs_rs_reward
                    elif component_name in ("task", "progress"):
                        component_tensor = component_tensor + bsrs_reward
                reward_components[component_name][step] = component_tensor

        maybe_obj = _extract_objectives_from_info(
            info=info,
            n_traj=args.n_traj,
            device=device,
        )

        if maybe_obj is not None:
            objectives = maybe_obj

        r_teacher = _extract_info_stack(info, "r_teacher")
        if r_teacher is not None:
            r_teacher_all.append(r_teacher)

        r_teacher_dense = _extract_info_stack(info, "r_teacher_dense")
        if r_teacher_dense is not None:
            r_teacher_dense_all.append(r_teacher_dense)

        r_teacher_route = _extract_info_stack(info, "r_teacher_route")
        if r_teacher_route is not None:
            r_teacher_route_all.append(r_teacher_route)

        teacher_gap = _extract_info_stack(info, "teacher_gap")
        if teacher_gap is not None:
            teacher_gap_all.append(teacher_gap)

        teacher_success_coef = _extract_info_stack(info, "teacher_success_coef")
        if teacher_success_coef is not None:
            success_bonus_coef_all.append(teacher_success_coef)

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
        "r_teacher_all": r_teacher_all,
        "r_teacher_dense_all": r_teacher_dense_all,
        "r_teacher_route_all": r_teacher_route_all,
        "teacher_gap_all": teacher_gap_all,
        "success_bonus_coef_all": success_bonus_coef_all,
        "r_bsrs_all": r_bsrs_all,
        "r_bsrs_customer_all": r_bsrs_customer_all,
        "r_bsrs_rs_all": r_bsrs_rs_all,
    }
    if reward_components is not None:
        result["reward_components"] = {
            k: v[:valid_step] for k, v in reward_components.items()
        }
    return result


def _student_best_objectives_from_rollout(rollout):
    objectives = rollout["objectives"]

    with torch.no_grad():
        student_success = torch.isfinite(objectives)
        student_fr_by_env = student_success.float().mean(dim=1)
        student_obj_by_env = objectives.min(dim=1).values

    return (
        student_obj_by_env.detach().cpu().numpy().astype(np.float32),
        student_fr_by_env.detach().cpu().numpy().astype(np.float32),
    )


def _student_best_action_sequences_from_rollout(rollout):
    objectives = rollout["objectives"].detach().cpu()
    actions = rollout["actions"].detach().cpu().numpy()
    valid_masks = rollout["valid_masks"].detach().cpu().numpy().astype(np.bool_)

    if objectives.numel() == 0 or actions.size == 0:
        return []

    num_envs = objectives.shape[0]
    best_traj = objectives.argmin(dim=1).numpy().astype(np.int64)
    sequences = []

    for env_idx in range(num_envs):
        traj_idx = int(best_traj[env_idx])
        valid = valid_masks[:, env_idx, traj_idx]
        seq = actions[valid, env_idx, traj_idx].astype(np.int64).tolist()
        sequences.append([int(a) for a in seq])

    return sequences


def _filter_alns_records_by_student_probe(
    agent,
    args,
    config,
    sampled_records,
    device,
    num_steps,
    update_step,
    DEBUG,
):
    if len(sampled_records) == 0:
        return [], None

    probe_envs = None

    try:
        probe_envs, _ = _make_alns_train_envs(
            args=args,
            config=config,
            update_step=update_step,
            sampled_records=sampled_records,
            use_teacher_reward=False,
            seed_offset=300000,
        )

        if probe_envs is None:
            return [], None

        probe_rollout = _student_rollout(
            agent=agent,
            envs=probe_envs,
            args=args,
            device=device,
            num_steps=num_steps,
            initial_obs=probe_envs.reset(),
            update_step=update_step,
            enable_bsrs=False,
        )

        student_obj, student_fr = _student_best_objectives_from_rollout(
            probe_rollout
        )
        student_sequences = _student_best_action_sequences_from_rollout(
            probe_rollout
        )

    finally:
        if probe_envs is not None:
            probe_envs.close()

    teacher_obj = np.asarray(
        [r["teacher_obj"] for r in sampled_records],
        dtype=np.float32,
    )

    margin = float(getattr(args, "alns_teacher_filter_margin", 0.0))
    teacher_valid = np.isfinite(teacher_obj) & (teacher_obj > 0.0)
    student_valid = np.isfinite(student_obj) & (student_obj > 0.0)

    keep_mask = (
        teacher_valid
        & (
            (~student_valid)
            | ((teacher_obj + margin) < student_obj)
        )
    )

    gap_abs = student_obj - teacher_obj
    gate_temp = max(float(getattr(args, "alns_gap_gate_temp", 0.10)), 1e-8)
    gate_margin = float(getattr(args, "alns_gap_gate_margin", margin))
    gate_raw = 1.0 / (1.0 + np.exp(-((gap_abs - gate_margin) / gate_temp)))
    gate_raw = np.where(np.isfinite(gate_raw), gate_raw, 1.0).astype(np.float32)

    if not bool(getattr(args, "use_alns_gap_gate", False)):
        gate_raw = np.ones_like(gate_raw, dtype=np.float32)

    filtered_records = [
        record
        for record, keep, gate, gap in zip(sampled_records, keep_mask, gate_raw, gap_abs)
        if bool(keep)
    ]
    filtered_gates = [
        float(gate)
        for keep, gate in zip(keep_mask, gate_raw)
        if bool(keep)
    ]
    filtered_gaps = [
        float(gap)
        for keep, gap in zip(keep_mask, gap_abs)
        if bool(keep)
    ]

    for record, gate, gap in zip(filtered_records, filtered_gates, filtered_gaps):
        record["teacher_gate"] = float(np.clip(gate, 0.0, 1.0))
        record["teacher_gap_abs"] = float(gap)

    if len(student_sequences) == len(sampled_records):
        filtered_student_sequences = [
            seq for seq, keep in zip(student_sequences, keep_mask) if bool(keep)
        ]
        for record, seq in zip(filtered_records, filtered_student_sequences):
            record["student_action_sequence"] = list(seq)

    raw_keep = len(filtered_records)
    num_minibatches = max(1, int(args.num_minibatches))
    usable_keep = (raw_keep // num_minibatches) * num_minibatches

    if usable_keep != raw_keep:
        filtered_records = filtered_records[:usable_keep]
        filtered_gates = filtered_gates[:usable_keep]
        filtered_gaps = filtered_gaps[:usable_keep]

    if DEBUG:
        kept_teacher_obj = teacher_obj[keep_mask]
        skipped_teacher_obj = teacher_obj[~keep_mask]
        kept_student_obj = student_obj[keep_mask]
        skipped_student_obj = student_obj[~keep_mask]

        def finite_mean(x):
            x = np.asarray(x, dtype=np.float32)
            x = x[np.isfinite(x)]
            return float(x.mean()) if x.size > 0 else float("nan")

        print(
            f"[ALNSFilter] update={update_step} | "
            f"sampled={len(sampled_records)} | "
            f"raw_keep={raw_keep} | "
            f"usable_keep={usable_keep} | "
            f"margin={margin:.6f} | "
            f"gate_mean={finite_mean(filtered_gates):.4f} | "
            f"gap_abs_mean={finite_mean(filtered_gaps):.4f} | "
            f"rl_fr_mean={float(np.mean(student_fr)):.4f} | "
            f"kept_alns_obj_mean={finite_mean(kept_teacher_obj):.4f} | "
            f"kept_rl_obj_mean={finite_mean(kept_student_obj):.4f} | "
            f"skipped_alns_obj_mean={finite_mean(skipped_teacher_obj):.4f} | "
            f"skipped_rl_obj_mean={finite_mean(skipped_student_obj):.4f}"
        )

    filter_info = {
        "sampled": len(sampled_records),
        "raw_keep": raw_keep,
        "usable_keep": usable_keep,
        "student_obj": student_obj,
        "student_fr": student_fr,
        "teacher_obj": teacher_obj,
    }

    return filtered_records, filter_info


def _alns_record_to_action_sequence(record, num_customers, num_nodes):
    routes = record.get("best_routes", None)
    if routes is None:
        routes = record.get("raw", {}).get("best_routes", None)

    if not isinstance(routes, list):
        return []

    actions = []

    for route in routes:
        if not isinstance(route, (list, tuple)) or len(route) == 0:
            continue

        clean_route = [int(x) for x in route]

        if clean_route[0] == 0:
            clean_route = clean_route[1:]

        if len(clean_route) == 0 or clean_route[-1] != 0:
            clean_route.append(0)

        for action in clean_route:
            if action < 0 or action >= num_nodes:
                return []
            actions.append(action)

    return actions


def _record_normalized_dist_matrix(record):
    instance = record.get("instance", None)
    if not isinstance(instance, dict):
        return None

    cached = record.get("_normalized_dist_matrix", None)
    if cached is not None:
        return cached

    try:
        nodes_raw = np.concatenate(
            (
                np.asarray(instance["depot"], dtype=np.float32),
                np.asarray(instance["customers"], dtype=np.float32),
                np.asarray(instance["charging_stations"], dtype=np.float32),
            ),
            axis=0,
        )

        env_cfg = instance["env"]
        area_size = np.asarray(env_cfg["area_size"], dtype=np.float32)
        x_scale = max(float(area_size[0][1] - area_size[0][0]), 1e-8)
        y_scale = max(float(area_size[1][1] - area_size[1][0]), 1e-8)

        nodes = np.zeros_like(nodes_raw, dtype=np.float32)
        nodes[:, 0] = (nodes_raw[:, 0] - float(area_size[0][0])) / x_scale
        nodes[:, 1] = (nodes_raw[:, 1] - float(area_size[1][0])) / y_scale

        diff = nodes[:, None, :] - nodes[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)

    except Exception:
        return None

    record["_normalized_dist_matrix"] = dist_matrix
    return dist_matrix


def _route_customer_events(
    action_sequence,
    dist_matrix,
    num_customers,
    normalize_route_distance=True,
):
    if action_sequence is None or dist_matrix is None:
        return []

    n_nodes = int(dist_matrix.shape[0])
    events = []
    route_events = []
    last = 0
    route_dist = 0.0

    def flush_route(total_dist):
        if len(route_events) == 0:
            return

        denom = max(float(total_dist), 1e-8)
        for step_idx, customer, arrival_dist in route_events:
            coord = (
                float(arrival_dist) / denom
                if normalize_route_distance
                else float(arrival_dist)
            )
            events.append((int(step_idx), int(customer), float(coord)))

    for step_idx, raw_action in enumerate(action_sequence):
        try:
            action = int(raw_action)
        except Exception:
            continue

        if action < 0 or action >= n_nodes:
            continue

        route_dist += float(dist_matrix[last, action])

        if 1 <= action <= int(num_customers):
            route_events.append((int(step_idx), int(action), float(route_dist)))

        if action == 0:
            flush_route(route_dist)
            route_events = []
            route_dist = 0.0
            last = 0
        else:
            last = action

    if len(route_events) > 0:
        total_dist = route_dist + float(dist_matrix[last, 0])
        flush_route(total_dist)

    return events


def _expert_route_coordinate_map(record, dist_matrix, args):
    cache_key = (
        "_teacher_route_coord_norm"
        if bool(getattr(args, "alns_rr_normalize_route_distance", True))
        else "_teacher_route_coord_abs"
    )

    cached = record.get(cache_key, None)
    if cached is not None:
        return cached

    num_customers = int(args.train_cus_num)
    num_nodes = int(args.train_cus_num + args.train_cs_num + 1)

    action_sequence = record.get("teacher_action_sequence", None)
    if action_sequence is None:
        action_sequence = _alns_record_to_action_sequence(
            record,
            num_customers=num_customers,
            num_nodes=num_nodes,
        )
        record["teacher_action_sequence"] = action_sequence

    events = _route_customer_events(
        action_sequence=action_sequence,
        dist_matrix=dist_matrix,
        num_customers=num_customers,
        normalize_route_distance=bool(
            getattr(args, "alns_rr_normalize_route_distance", True)
        ),
    )

    coord_map = {}
    for _step_idx, customer, coord in events:
        coord_map.setdefault(int(customer), float(coord))

    record[cache_key] = coord_map
    return coord_map


def _rollout_teacher_component_stack(rollout, device):
    valid_step = int(rollout["valid_step"])
    rewards = rollout["rewards"]

    reward_components = rollout.get("reward_components", None)
    if reward_components is not None and "teacher" in reward_components:
        return reward_components["teacher"].detach().clone()

    teacher_info = rollout.get("r_teacher_all", [])
    if len(teacher_info) == valid_step:
        teacher_np = np.stack(teacher_info, axis=0).astype(np.float32)
        return torch.tensor(teacher_np, device=device, dtype=torch.float32)

    return torch.zeros_like(rewards, device=device)


def _build_route_credit_vector(
    action_sequence,
    expert_coord_map,
    dist_matrix,
    args,
):
    valid_len = len(action_sequence)
    credit = np.zeros((valid_len,), dtype=np.float32)

    if valid_len == 0:
        return credit, 0, 0.0

    events = _route_customer_events(
        action_sequence=action_sequence,
        dist_matrix=dist_matrix,
        num_customers=int(args.train_cus_num),
        normalize_route_distance=bool(
            getattr(args, "alns_rr_normalize_route_distance", True)
        ),
    )

    if len(events) == 0:
        return credit, 0, 0.0

    if str(getattr(args, "alns_rr_credit_mode", "route_coord")) == "uniform":
        scores = np.ones((len(events),), dtype=np.float32)
    else:
        tau = max(float(getattr(args, "alns_rr_temperature", 0.20)), 1e-8)
        score_list = []

        for _step_idx, customer, coord in events:
            expert_coord = expert_coord_map.get(int(customer), None)
            if expert_coord is None:
                mismatch = 1.0
            else:
                mismatch = abs(float(coord) - float(expert_coord))

            mismatch = float(np.clip(mismatch, 0.0, 1.0))
            # Higher mismatch receives larger credit for negative expert gaps;
            # the small floor keeps exact route matches trainable.
            score_list.append(0.05 + (1.0 - np.exp(-mismatch / tau)))

        scores = np.asarray(score_list, dtype=np.float32)

    score_sum = float(scores.sum())
    if not np.isfinite(score_sum) or score_sum <= 1e-8:
        scores = np.ones((len(events),), dtype=np.float32)
        score_sum = float(scores.sum())

    scores = scores / max(score_sum, 1e-8)

    mismatches = []
    for score, (step_idx, customer, coord) in zip(scores, events):
        if 0 <= int(step_idx) < valid_len:
            credit[int(step_idx)] += float(score)

        expert_coord = expert_coord_map.get(int(customer), None)
        if expert_coord is not None:
            mismatches.append(abs(float(coord) - float(expert_coord)))

    avg_mismatch = float(np.mean(mismatches)) if len(mismatches) > 0 else 0.0
    return credit, len(events), avg_mismatch


def _apply_alns_return_redistribution(
    rollout,
    records,
    args,
    device,
    update_step,
    DEBUG,
):
    if not bool(getattr(args, "use_alns_return_redistribution", False)):
        return None

    if records is None or len(records) == 0:
        return None

    valid_step = int(rollout["valid_step"])
    if valid_step <= 0:
        return None

    source = str(getattr(args, "alns_rr_signal_source", "hybrid"))
    use_existing_teacher = source in ("env_teacher", "hybrid")
    use_gap_signal = source in ("gap", "hybrid")

    rewards = rollout["rewards"]
    actions = rollout["actions"].detach().cpu().numpy()
    valid_masks = rollout["valid_masks"].detach().cpu().numpy().astype(np.bool_)
    objectives = rollout["objectives"].detach().cpu().numpy()

    num_envs = rewards.shape[1]
    n_traj = rewards.shape[2]

    teacher_component = _rollout_teacher_component_stack(rollout, device)
    redistributed = torch.zeros_like(rewards, device=device)

    coef = float(getattr(args, "alns_rr_coef", 0.50))
    clip_value = float(getattr(args, "alns_rr_clip", 0.25))
    fail_value = float(getattr(args, "alns_rr_fail_value", 0.25))

    total_signal = []
    gap_signal = []
    existing_signal = []
    avg_mismatch = []
    nonzero_traj = 0
    credit_events = 0

    for env_idx in range(min(num_envs, len(records))):
        record = records[env_idx]
        dist_matrix = _record_normalized_dist_matrix(record)
        if dist_matrix is None:
            continue

        expert_coord_map = _expert_route_coordinate_map(
            record=record,
            dist_matrix=dist_matrix,
            args=args,
        )
        if len(expert_coord_map) == 0:
            continue

        teacher_obj = float(record.get("teacher_obj", np.nan))
        teacher_gate = float(np.clip(record.get("teacher_gate", 1.0), 0.0, 1.0))

        for traj_idx in range(n_traj):
            valid = valid_masks[:valid_step, env_idx, traj_idx]
            if not np.any(valid):
                continue

            seq = actions[:valid_step, env_idx, traj_idx][valid].astype(np.int64).tolist()
            credit, event_count, mismatch = _build_route_credit_vector(
                action_sequence=seq,
                expert_coord_map=expert_coord_map,
                dist_matrix=dist_matrix,
                args=args,
            )

            if event_count <= 0 or credit.sum() <= 1e-8:
                continue

            existing = 0.0
            if use_existing_teacher:
                existing = float(
                    teacher_component[:valid_step, env_idx, traj_idx].sum().item()
                )

            gap_part = 0.0
            if use_gap_signal and coef > 0.0 and teacher_gate > 0.0:
                student_obj = float(objectives[env_idx, traj_idx])
                served_customers = {
                    int(a)
                    for a in seq
                    if 1 <= int(a) <= int(args.train_cus_num)
                }

                if (
                    np.isfinite(student_obj)
                    and student_obj > 0.0
                    and np.isfinite(teacher_obj)
                    and teacher_obj > 0.0
                ):
                    log_gap = np.log(teacher_obj / (student_obj + 1e-8))
                else:
                    served_ratio = len(served_customers) / max(float(args.train_cus_num), 1.0)
                    log_gap = -fail_value * max(0.25, 1.0 - served_ratio)

                log_gap = float(np.clip(log_gap, -clip_value, clip_value))
                gap_part = coef * teacher_gate * log_gap

            signal = existing + gap_part
            if not np.isfinite(signal) or abs(signal) <= 1e-10:
                continue

            valid_indices = np.flatnonzero(valid)
            credit_t = torch.tensor(
                credit,
                device=device,
                dtype=torch.float32,
            )
            add_indices = torch.tensor(valid_indices, device=device, dtype=torch.long)
            redistributed[add_indices, env_idx, traj_idx] += credit_t * float(signal)

            total_signal.append(float(signal))
            gap_signal.append(float(gap_part))
            existing_signal.append(float(existing))
            avg_mismatch.append(float(mismatch))
            nonzero_traj += 1
            credit_events += int(event_count)

    if use_existing_teacher:
        rollout["rewards"].sub_(teacher_component)
        reward_components = rollout.get("reward_components", None)
        if reward_components is not None and "teacher" in reward_components:
            reward_components["teacher"].sub_(teacher_component)

    rollout["rewards"].add_(redistributed)
    reward_components = rollout.get("reward_components", None)
    if reward_components is not None and "teacher" in reward_components:
        reward_components["teacher"].add_(redistributed)

    redist_abs = redistributed.abs()
    nonzero_steps = int((redist_abs > 1e-10).sum().item())
    redist_sum = float(redistributed.sum().item())

    info = {
        "nonzero_traj": int(nonzero_traj),
        "nonzero_steps": int(nonzero_steps),
        "credit_events": int(credit_events),
        "signal_mean": float(np.mean(total_signal)) if len(total_signal) > 0 else 0.0,
        "signal_abs_mean": float(np.mean(np.abs(total_signal))) if len(total_signal) > 0 else 0.0,
        "gap_signal_mean": float(np.mean(gap_signal)) if len(gap_signal) > 0 else 0.0,
        "existing_signal_mean": float(np.mean(existing_signal)) if len(existing_signal) > 0 else 0.0,
        "route_mismatch_mean": float(np.mean(avg_mismatch)) if len(avg_mismatch) > 0 else 0.0,
        "redistributed_sum": redist_sum,
        "source": source,
    }
    rollout["alns_rr_info"] = info

    if DEBUG:
        print(
            f"[ALNSReturnRedistribution] update={update_step} | "
            f"source={source} | "
            f"traj={info['nonzero_traj']}/{num_envs * n_traj} | "
            f"steps={info['nonzero_steps']} | "
            f"events={info['credit_events']} | "
            f"signal_mean={info['signal_mean']:.6f} | "
            f"abs_signal_mean={info['signal_abs_mean']:.6f} | "
            f"gap_signal_mean={info['gap_signal_mean']:.6f} | "
            f"existing_signal_mean={info['existing_signal_mean']:.6f} | "
            f"route_mismatch_mean={info['route_mismatch_mean']:.6f} | "
            f"redistributed_sum={info['redistributed_sum']:.6f}"
        )

    return info


def _make_alns_bc_envs(
    args,
    config,
    sampled_records,
    update_step,
):
    if len(sampled_records) == 0:
        return None

    return SyncVectorEnv(
        [
            make_env(
                args.env_id,
                int(args.seed + 400000 + update_step * 1000 + i),
                cfg={
                    "env_mode": "train",
                    "config": config,
                    "n_traj": 1,
                    "eval_data": deepcopy(record["instance"]),
                    "num_customers": args.train_cus_num,
                    "num_charging_stations": args.train_cs_num,
                    "gamma": args.gamma,
                    "lambda_fail": args.lambda_fail_init,
                    **_teacher_reward_cfg(args, use_teacher_reward=False),
                },
            )
            for i, record in enumerate(sampled_records)
        ]
    )


def _alns_behavior_cloning_update(
    agent,
    optim_backbone,
    args,
    config,
    records,
    device,
    update_step,
    DEBUG,
):
    if not bool(getattr(args, "use_alns_bc", True)):
        return None

    if records is None or len(records) == 0:
        return None

    bc_batch_size = int(getattr(args, "alns_bc_batch_size", 64))
    bc_batch_size = max(1, min(bc_batch_size, len(records)))

    # Keep deterministic selection within this already-sampled ALNS batch.
    bc_records = list(records[:bc_batch_size])
    num_nodes = int(args.train_cus_num + args.train_cs_num + 1)
    max_steps = int(getattr(args, "alns_bc_max_steps", args.num_steps))
    max_steps = max(1, max_steps)

    sequences = [
        _alns_record_to_action_sequence(
            record,
            num_customers=int(args.train_cus_num),
            num_nodes=num_nodes,
        )[:max_steps]
        for record in bc_records
    ]
    bc_gates = np.asarray(
        [
            float(record.get("teacher_gate", 1.0))
            if bool(getattr(args, "use_alns_gap_gate", False))
            else 1.0
            for record in bc_records
        ],
        dtype=np.float32,
    )

    valid_seq_mask = np.asarray([len(seq) > 0 for seq in sequences], dtype=np.bool_)
    if not np.any(valid_seq_mask):
        return None

    bc_records = [record for record, keep in zip(bc_records, valid_seq_mask) if keep]
    sequences = [seq for seq, keep in zip(sequences, valid_seq_mask) if keep]
    bc_gates = bc_gates[valid_seq_mask]

    total_expert_actions = sum(len(seq) for seq in sequences)
    if total_expert_actions <= 0:
        return None

    bc_envs = None
    agent.train()

    try:
        bc_envs = _make_alns_bc_envs(
            args=args,
            config=config,
            sampled_records=bc_records,
            update_step=update_step,
        )

        if bc_envs is None:
            return None

        obs = bc_envs.reset()
        cached_embeddings = agent.backbone.encode(obs)
        active = np.ones(len(sequences), dtype=np.bool_)

        optim_backbone.zero_grad(set_to_none=True)

        used_actions = 0
        infeasible_actions = 0
        total_loss_value = 0.0
        bc_coef = float(getattr(args, "alns_bc_coef", 0.2))
        loss_terms = []

        for step in range(max_steps):
            target = np.zeros((len(sequences), 1), dtype=np.int64)
            has_target = np.zeros(len(sequences), dtype=np.bool_)

            for i, seq in enumerate(sequences):
                if active[i] and step < len(seq):
                    target[i, 0] = int(seq[step])
                    has_target[i] = True

            if not np.any(has_target):
                break

            action_mask = np.asarray(obs["action_mask"], dtype=np.bool_)
            feasible = action_mask[
                np.arange(len(sequences)),
                0,
                target[:, 0],
            ]
            valid = has_target & active & feasible
            infeasible_actions += int(np.sum(has_target & active & (~feasible)))

            if np.any(valid):
                target_t = torch.tensor(target, device=device, dtype=torch.long)
                gate_t = torch.tensor(
                    bc_gates[:, None],
                    device=device,
                    dtype=torch.float32,
                )

                _, logprob, _, _, _ = agent.get_action_and_value_cached(
                    obs,
                    action=target_t,
                    cached_embeddings=cached_embeddings,
                )

                valid_t = torch.tensor(valid[:, None], device=device, dtype=torch.bool)
                weighted_logprob = logprob * gate_t
                loss = -weighted_logprob[valid_t].sum() / float(total_expert_actions)
                loss_terms.append(loss)

                used_actions += int(np.sum(valid))
                total_loss_value += float(loss.detach().cpu().item())

            forced_action = target.copy()
            forced_action[~valid, 0] = 0
            obs, _, done, _ = bc_envs.step(forced_action)
            active = active & valid & (~done.reshape(-1))

            if not np.any(active):
                break

        if used_actions == 0:
            optim_backbone.zero_grad(set_to_none=True)
            return None

        total_loss = torch.stack(loss_terms).sum()
        scaled_loss = bc_coef * total_loss
        scaled_loss.backward()

        grad_norm_value = nn.utils.clip_grad_norm_(
            agent.backbone.parameters(),
            args.max_grad_norm_backbone,
        )
        optim_backbone.step()
        optim_backbone.zero_grad(set_to_none=True)

        info = {
            "batch": len(bc_records),
            "used_actions": used_actions,
            "expert_actions": total_expert_actions,
            "infeasible_actions": infeasible_actions,
            "loss": total_loss_value,
            "coef": bc_coef,
            "gate_mean": float(np.mean(bc_gates)) if bc_gates.size > 0 else 1.0,
            "grad_norm": float(grad_norm_value),
        }

        if DEBUG:
            print(
                f"[ALNSBC] update={update_step} | "
                f"batch={info['batch']} | "
                f"used_actions={used_actions}/{total_expert_actions} | "
                f"infeasible={infeasible_actions} | "
                f"loss={total_loss_value:.6f} | "
                f"coef={bc_coef:.4f} | "
                f"gate_mean={info['gate_mean']:.4f} | "
                f"grad_norm={float(grad_norm_value):.6f}"
            )

        return info

    finally:
        if bc_envs is not None:
            bc_envs.close()


def _forced_sequence_logprobs(
    agent,
    envs,
    sequences,
    device,
    max_steps,
    length_norm=True,
):
    if envs is None or len(sequences) == 0:
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
        feasible = action_mask[
            np.arange(batch_size),
            0,
            target[:, 0],
        ]
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

    if length_norm:
        route_logprob = logprob_sums / lengths.clamp(min=1.0)
    else:
        route_logprob = logprob_sums

    valid_routes = lengths > 0
    return route_logprob, valid_routes, lengths, infeasible_actions


def _alns_preference_update(
    agent,
    optim_backbone,
    args,
    config,
    records,
    device,
    update_step,
    DEBUG,
):
    if not bool(getattr(args, "use_alns_preference", False)):
        return None

    if records is None or len(records) == 0:
        return None

    pref_batch_size = int(getattr(args, "alns_pref_batch_size", 64))
    pref_batch_size = max(1, min(pref_batch_size, len(records)))
    pref_records = list(records[:pref_batch_size])

    num_nodes = int(args.train_cus_num + args.train_cs_num + 1)
    max_steps = int(getattr(args, "alns_pref_max_steps", args.num_steps))
    max_steps = max(1, max_steps)

    alns_sequences = []
    student_sequences = []
    pref_gates = []
    valid_records = []

    for record in pref_records:
        alns_seq = record.get("teacher_action_sequence", None)
        if alns_seq is None:
            alns_seq = _alns_record_to_action_sequence(
                record,
                num_customers=int(args.train_cus_num),
                num_nodes=num_nodes,
            )
            record["teacher_action_sequence"] = alns_seq

        student_seq = record.get("student_action_sequence", None)
        if student_seq is None:
            continue

        alns_seq = [int(a) for a in list(alns_seq)[:max_steps]]
        student_seq = [int(a) for a in list(student_seq)[:max_steps]]

        if len(alns_seq) == 0 or len(student_seq) == 0:
            continue

        alns_sequences.append(alns_seq)
        student_sequences.append(student_seq)
        pref_gates.append(
            float(record.get("teacher_gate", 1.0))
            if bool(getattr(args, "use_alns_gap_gate", False))
            else 1.0
        )
        valid_records.append(record)

    if len(alns_sequences) == 0:
        return None

    pos_envs = None
    neg_envs = None
    agent.train()

    try:
        pos_envs = _make_alns_bc_envs(
            args=args,
            config=config,
            sampled_records=valid_records,
            update_step=update_step,
        )
        neg_envs = _make_alns_bc_envs(
            args=args,
            config=config,
            sampled_records=valid_records,
            update_step=update_step + 777,
        )

        if pos_envs is None or neg_envs is None:
            return None

        length_norm = bool(getattr(args, "alns_pref_length_norm", True))
        pos_result = _forced_sequence_logprobs(
            agent=agent,
            envs=pos_envs,
            sequences=alns_sequences,
            device=device,
            max_steps=max_steps,
            length_norm=length_norm,
        )
        neg_result = _forced_sequence_logprobs(
            agent=agent,
            envs=neg_envs,
            sequences=student_sequences,
            device=device,
            max_steps=max_steps,
            length_norm=length_norm,
        )

        if pos_result is None or neg_result is None:
            return None

        pos_logprob, pos_valid, pos_lengths, pos_infeasible = pos_result
        neg_logprob, neg_valid, neg_lengths, neg_infeasible = neg_result
        valid = pos_valid & neg_valid

        if not torch.any(valid):
            return None

        weights = torch.tensor(pref_gates, device=device, dtype=torch.float32)
        weights = torch.clamp(weights, min=0.0)
        weights = weights * valid.float()

        if weights.sum() <= 1e-8:
            return None

        beta = float(getattr(args, "alns_pref_beta", 2.0))
        pref_coef = float(getattr(args, "alns_pref_coef", 0.05))

        route_delta = pos_logprob - neg_logprob
        loss_terms = -F.logsigmoid(beta * route_delta)
        total_loss = (loss_terms * weights).sum() / weights.sum().clamp(min=1e-8)
        scaled_loss = pref_coef * total_loss

        optim_backbone.zero_grad(set_to_none=True)
        scaled_loss.backward()
        grad_norm_value = nn.utils.clip_grad_norm_(
            agent.backbone.parameters(),
            args.max_grad_norm_backbone,
        )
        optim_backbone.step()
        optim_backbone.zero_grad(set_to_none=True)

        info = {
            "batch": int(torch.sum(valid).detach().cpu().item()),
            "loss": float(total_loss.detach().cpu().item()),
            "coef": pref_coef,
            "beta": beta,
            "delta_mean": float(route_delta[valid].detach().mean().cpu().item()),
            "pos_logprob": float(pos_logprob[valid].detach().mean().cpu().item()),
            "neg_logprob": float(neg_logprob[valid].detach().mean().cpu().item()),
            "pos_len": float(pos_lengths[valid].detach().mean().cpu().item()),
            "neg_len": float(neg_lengths[valid].detach().mean().cpu().item()),
            "gate_mean": float(weights[valid].detach().mean().cpu().item()),
            "infeasible_actions": int(pos_infeasible + neg_infeasible),
            "grad_norm": float(grad_norm_value),
        }

        if DEBUG:
            print(
                f"[ALNSPref] update={update_step} | "
                f"batch={info['batch']} | "
                f"loss={info['loss']:.6f} | "
                f"coef={info['coef']:.4f} | "
                f"beta={info['beta']:.3f} | "
                f"delta={info['delta_mean']:.4f} | "
                f"pos_lp={info['pos_logprob']:.4f} | "
                f"neg_lp={info['neg_logprob']:.4f} | "
                f"len={info['pos_len']:.1f}/{info['neg_len']:.1f} | "
                f"gate={info['gate_mean']:.4f} | "
                f"infeasible={info['infeasible_actions']} | "
                f"grad_norm={info['grad_norm']:.6f}"
            )

        return info

    finally:
        if pos_envs is not None:
            pos_envs.close()
        if neg_envs is not None:
            neg_envs.close()


def _build_alns_soft_targets(
    alns_sequences,
    teacher_progress,
    action_mask,
    lookahead,
    tau,
    num_customers,
):
    batch_size = len(alns_sequences)
    num_nodes = action_mask.shape[-1]
    targets = np.zeros((batch_size, num_nodes), dtype=np.float32)
    valid_rows = np.zeros(batch_size, dtype=np.bool_)
    target_counts = np.zeros(batch_size, dtype=np.int32)

    lookahead = max(1, int(lookahead))
    tau = max(float(tau), 1e-6)
    cus_start = 1
    rs_start = 1 + int(num_customers)

    for batch_idx, seq in enumerate(alns_sequences):
        if len(seq) == 0:
            continue

        start = int(np.clip(teacher_progress[batch_idx], 0, max(0, len(seq) - 1)))
        end = min(len(seq), start + lookahead)
        if start >= end:
            continue

        for rank, node in enumerate(seq[start:end]):
            node = int(node)
            if node < cus_start or node >= rs_start:
                continue
            if node >= num_nodes:
                continue
            if not bool(action_mask[batch_idx, 0, node]):
                continue

            targets[batch_idx, node] += float(np.exp(-float(rank) / tau))
            target_counts[batch_idx] += 1

        row_sum = float(targets[batch_idx].sum())
        if row_sum > 1e-8:
            targets[batch_idx] /= row_sum
            valid_rows[batch_idx] = True

    return targets, valid_rows, target_counts


def _advance_teacher_progress_from_actions(alns_sequences, teacher_progress, actions, valid):
    for batch_idx, is_valid in enumerate(valid):
        if not bool(is_valid):
            continue

        seq = alns_sequences[batch_idx]
        if len(seq) == 0:
            continue

        start = int(np.clip(teacher_progress[batch_idx], 0, len(seq)))
        if start >= len(seq):
            continue

        action = int(actions[batch_idx])
        suffix = np.asarray(seq[start:], dtype=np.int64)
        hits = np.where(suffix == action)[0]
        if hits.size > 0:
            teacher_progress[batch_idx] = min(len(seq), start + int(hits[0]) + 1)


def _alns_soft_target_update(
    agent,
    optim_backbone,
    args,
    config,
    records,
    device,
    update_step,
    DEBUG,
):
    if not bool(getattr(args, "use_alns_soft_kl", False)):
        return None

    if records is None or len(records) == 0:
        return None

    soft_batch_size = int(getattr(args, "alns_soft_kl_batch_size", 64))
    soft_batch_size = max(1, min(soft_batch_size, len(records)))
    soft_records = list(records[:soft_batch_size])

    num_nodes = int(args.train_cus_num + args.train_cs_num + 1)
    max_steps = int(getattr(args, "alns_soft_kl_max_steps", args.num_steps))
    max_steps = max(1, max_steps)
    lookahead = int(getattr(args, "alns_soft_kl_lookahead", 8))
    tau = float(getattr(args, "alns_soft_kl_tau", 2.0))

    alns_sequences = []
    student_sequences = []
    soft_gates = []
    valid_records = []

    for record in soft_records:
        alns_seq = record.get("teacher_action_sequence", None)
        if alns_seq is None:
            alns_seq = _alns_record_to_action_sequence(
                record,
                num_customers=int(args.train_cus_num),
                num_nodes=num_nodes,
            )
            record["teacher_action_sequence"] = alns_seq

        student_seq = record.get("student_action_sequence", None)
        if student_seq is None:
            continue

        alns_seq = [int(a) for a in list(alns_seq)]
        student_seq = [int(a) for a in list(student_seq)[:max_steps]]

        if len(alns_seq) == 0 or len(student_seq) == 0:
            continue

        alns_sequences.append(alns_seq)
        student_sequences.append(student_seq)
        soft_gates.append(
            float(record.get("teacher_gate", 1.0))
            if bool(getattr(args, "use_alns_gap_gate", False))
            else 1.0
        )
        valid_records.append(record)

    if len(alns_sequences) == 0:
        return None

    soft_envs = None
    agent.train()

    try:
        soft_envs = _make_alns_bc_envs(
            args=args,
            config=config,
            sampled_records=valid_records,
            update_step=update_step + 333,
        )

        if soft_envs is None:
            return None

        obs = soft_envs.reset()
        cached_embeddings = agent.backbone.encode(obs)
        batch_size = len(alns_sequences)
        active = np.ones(batch_size, dtype=np.bool_)
        teacher_progress = np.zeros(batch_size, dtype=np.int32)
        gates = np.asarray(soft_gates, dtype=np.float32)

        optim_backbone.zero_grad(set_to_none=True)

        numerator_terms = []
        denominator_terms = []
        used_states = 0
        total_target_choices = 0
        infeasible_student_actions = 0

        for step in range(max_steps):
            target_action = np.zeros((batch_size, 1), dtype=np.int64)
            has_action = np.zeros(batch_size, dtype=np.bool_)

            for batch_idx, seq in enumerate(student_sequences):
                if active[batch_idx] and step < len(seq):
                    target_action[batch_idx, 0] = int(seq[step])
                    has_action[batch_idx] = True

            if not np.any(has_action):
                break

            action_mask = np.asarray(obs["action_mask"], dtype=np.bool_)

            soft_targets, target_valid, target_counts = _build_alns_soft_targets(
                alns_sequences=alns_sequences,
                teacher_progress=teacher_progress,
                action_mask=action_mask,
                lookahead=lookahead,
                tau=tau,
                num_customers=int(args.train_cus_num),
            )

            row_valid = active & has_action & target_valid

            if np.any(row_valid):
                backbone_output = agent.backbone.decode(obs, cached_embeddings)
                logits = agent.actor(backbone_output).squeeze(1)
                log_probs = F.log_softmax(logits, dim=-1)

                target_t = torch.tensor(
                    soft_targets,
                    device=device,
                    dtype=torch.float32,
                )
                weight_t = torch.tensor(
                    gates,
                    device=device,
                    dtype=torch.float32,
                )
                valid_t = torch.tensor(row_valid, device=device, dtype=torch.bool)

                safe_log_probs = torch.nan_to_num(
                    log_probs,
                    nan=0.0,
                    neginf=-1e9,
                    posinf=0.0,
                )
                target_log_probs = torch.where(
                    target_t > 0.0,
                    safe_log_probs,
                    torch.zeros_like(safe_log_probs),
                )
                cross_entropy = -(target_t * target_log_probs).sum(dim=-1)
                weights = torch.clamp(weight_t, min=0.0) * valid_t.float()
                weight_sum = weights.sum()

                if weight_sum > 1e-8:
                    numerator_terms.append((cross_entropy * weights).sum())
                    denominator_terms.append(weight_sum)
                    used_states += int(np.sum(row_valid))
                    total_target_choices += int(target_counts[row_valid].sum())

            feasible_student = action_mask[
                np.arange(batch_size),
                0,
                target_action[:, 0],
            ]
            valid_action = active & has_action & feasible_student
            infeasible_student_actions += int(np.sum(active & has_action & (~feasible_student)))

            forced_action = target_action.copy()
            forced_action[~valid_action, 0] = 0
            obs, _, done, _ = soft_envs.step(forced_action)

            _advance_teacher_progress_from_actions(
                alns_sequences=alns_sequences,
                teacher_progress=teacher_progress,
                actions=forced_action[:, 0],
                valid=valid_action,
            )

            active = active & valid_action & (~done.reshape(-1))

            if not np.any(active):
                break

        if used_states == 0 or len(numerator_terms) == 0:
            optim_backbone.zero_grad(set_to_none=True)
            return None

        numerator = torch.stack(numerator_terms).sum()
        denominator = torch.stack(denominator_terms).sum().clamp(min=1e-8)
        total_loss = numerator / denominator

        soft_coef = float(getattr(args, "alns_soft_kl_coef", 0.05))
        scaled_loss = soft_coef * total_loss
        scaled_loss.backward()

        grad_norm_value = nn.utils.clip_grad_norm_(
            agent.backbone.parameters(),
            args.max_grad_norm_backbone,
        )
        optim_backbone.step()
        optim_backbone.zero_grad(set_to_none=True)

        info = {
            "batch": len(valid_records),
            "used_states": used_states,
            "avg_target_choices": (
                float(total_target_choices) / float(max(used_states, 1))
            ),
            "loss": float(total_loss.detach().cpu().item()),
            "coef": soft_coef,
            "lookahead": lookahead,
            "tau": tau,
            "gate_mean": float(np.mean(gates)) if gates.size > 0 else 1.0,
            "infeasible_student_actions": infeasible_student_actions,
            "grad_norm": float(grad_norm_value),
        }

        if DEBUG:
            print(
                f"[ALNSSoftKL] update={update_step} | "
                f"batch={info['batch']} | "
                f"states={info['used_states']} | "
                f"choices={info['avg_target_choices']:.2f} | "
                f"loss={info['loss']:.6f} | "
                f"coef={info['coef']:.4f} | "
                f"lookahead={info['lookahead']} | "
                f"tau={info['tau']:.3f} | "
                f"gate={info['gate_mean']:.4f} | "
                f"infeasible_student={info['infeasible_student_actions']} | "
                f"grad_norm={info['grad_norm']:.6f}"
            )

        return info

    finally:
        if soft_envs is not None:
            soft_envs.close()


# =========================================================
# PPO update
# =========================================================
def _compute_gae_and_returns(agent, rollout, args, device):
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    next_obs = rollout["next_obs"]
    next_done = rollout["next_done"]
    encoder_state = rollout["encoder_state"]
    valid_step = rollout["valid_step"]

    num_envs = rewards.shape[1]

    def standardize_component_advantages(component_advantages):
        valid_masks = rollout["valid_masks"][:valid_step].bool()
        standardized = torch.zeros_like(component_advantages, device=device)
        weights = _decomposed_component_weights(
            args,
            _decomposed_component_names(args),
            device=device,
            dtype=torch.float32,
            success_rate=getattr(args, "current_rollout_success_rate", None),
        )

        for head_idx in range(component_advantages.shape[-1]):
            adv_i = component_advantages[..., head_idx]
            valid_adv = adv_i[valid_masks]

            if valid_adv.numel() <= 1:
                weights[head_idx] = 0.0
                continue

            adv_std = valid_adv.std()
            if not torch.isfinite(adv_std) or adv_std <= 1e-8:
                weights[head_idx] = 0.0
                continue

            adv_mean = valid_adv.mean()
            standardized[..., head_idx] = (adv_i - adv_mean) / (adv_std + 1e-8)
            standardized[..., head_idx] = standardized[..., head_idx] * valid_masks

        weights = torch.clamp(weights, min=0.0)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum

        return standardized, weights

    def gae_for(reward_tensor, value_tensor, next_value_tensor):
        advantages = torch.zeros_like(reward_tensor, device=device)
        lastgaelam = torch.zeros(num_envs, args.n_traj, device=device)

        for t in reversed(range(valid_step)):
            if t == valid_step - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value_tensor
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = value_tensor[t + 1]

            delta = reward_tensor[t] + args.gamma * nextvalues * nextnonterminal - value_tensor[t]
            advantages[t] = lastgaelam = (
                delta
                + args.gamma
                * args.gae_lambda
                * nextnonterminal
                * lastgaelam
            )

        returns = advantages + value_tensor
        return advantages, returns

    with torch.no_grad():
        next_value = agent.get_value_cached(next_obs, encoder_state).squeeze(-1)

        if bool(getattr(args, "use_decomposed_reward_adv", False)):
            if values.dim() != 4 or next_value.dim() != 3:
                raise ValueError(
                    "use_decomposed_reward_adv=True requires a multi-head critic."
                )

            component_rewards = rollout.get("reward_components", None)
            if component_rewards is None:
                raise ValueError(
                    "use_decomposed_reward_adv=True requires reward_components in rollout."
                )

            names = _decomposed_component_names(args)
            adv_list = []
            ret_list = []

            for head_idx, name in enumerate(names):
                adv_i, ret_i = gae_for(
                    component_rewards[name],
                    values[..., head_idx],
                    next_value[..., head_idx],
                )
                adv_list.append(adv_i)
                ret_list.append(ret_i)

            component_advantages = torch.stack(adv_list, dim=-1)
            returns = torch.stack(ret_list, dim=-1)
            standardized_advantages, weights = standardize_component_advantages(
                component_advantages
            )
            args.last_adv_component_weights = {
                name: float(weights[head_idx].detach().cpu().item())
                for head_idx, name in enumerate(names)
            }
            args.last_adv_adaptive_mix = _adaptive_adv_mix(
                args,
                getattr(args, "current_rollout_success_rate", None),
            )
            advantages = (
                standardized_advantages * weights.view(1, 1, 1, -1)
            ).sum(dim=-1)
            return advantages, returns

        advantages, returns = gae_for(rewards, values, next_value)

    return advantages, returns


def _ppo_update(
    agent,
    optim_backbone,
    optim_critic,
    envs,
    rollout,
    advantages,
    returns,
    args,
    DEBUG,
):
    obs = rollout["obs"]
    actions = rollout["actions"]
    logprobs = rollout["logprobs"]
    values = rollout["values"]
    valid_masks = rollout["valid_masks"]
    valid_step = rollout["valid_step"]

    num_envs = len(envs.envs)

    args.batch_size = int(num_envs * valid_step)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    b_obs = {
        k: np.concatenate([obs_[k] for obs_ in obs])
        for k in envs.single_observation_space
    }

    b_logprobs = logprobs.reshape(-1, args.n_traj)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1, args.n_traj).detach()
    if values.dim() == 4:
        value_heads = values.shape[-1]
        b_returns = returns.reshape(-1, args.n_traj, value_heads)
        b_values = values.reshape(-1, args.n_traj, value_heads)
    else:
        value_heads = 1
        b_returns = returns.reshape(-1, args.n_traj)
        b_values = values.reshape(-1, args.n_traj)
    b_valid_masks = valid_masks.reshape(-1, args.n_traj).bool()

    assert num_envs % args.num_minibatches == 0

    envsperbatch = num_envs // args.num_minibatches
    envinds = np.arange(num_envs)
    flatinds = np.arange(args.batch_size).reshape(valid_step, num_envs)

    mb_kls = []
    mb_pg_losses = []
    mb_v_losses = []
    mb_entropies = []
    mb_grad_norms_backbone = []
    mb_grad_norms_critic = []

    if args.norm_adv and not bool(getattr(args, "use_decomposed_reward_adv", False)):
        valid_adv = b_advantages[b_valid_masks]

        if valid_adv.numel() > 1:
            adv_mean_norm = valid_adv.mean()
            adv_std_norm = valid_adv.std() + 1e-8
            b_advantages = (b_advantages - adv_mean_norm) / adv_std_norm
            b_advantages = b_advantages * b_valid_masks

    stop_early = False
    accum_steps = max(1, int(getattr(args, "accum_steps", 1)))

    for epoch in range(args.update_epochs):
        if stop_early:
            break

        np.random.shuffle(envinds)
        epoch_kls = []

        optim_backbone.zero_grad(set_to_none=True)
        optim_critic.zero_grad(set_to_none=True)

        accum_counter = 0
        step_grad_norms_backbone = []
        step_grad_norms_critic = []

        for start_mb in range(0, num_envs, envsperbatch):
            end_mb = start_mb + envsperbatch
            mbenvinds = envinds[start_mb:end_mb]

            mb_inds = flatinds[:, mbenvinds].ravel()
            r_inds = np.tile(np.arange(envsperbatch), valid_step)

            cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}
            encoder_state_mb = agent.backbone.encode(cur_obs)

            mb_valid = b_valid_masks[mb_inds]

            _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value_cached(
                {k: v[mb_inds] for k, v in b_obs.items()},
                b_actions.long()[mb_inds],
                (embedding[r_inds, :] for embedding in encoder_state_mb),
            )

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                if mb_valid.any():
                    lr = logratio[mb_valid]
                    rr = ratio[mb_valid]
                    approx_kl = ((rr - 1.0) - lr).mean().detach()
                else:
                    approx_kl = logratio.new_tensor(0.0).detach()

                epoch_kls.append(approx_kl)

            mb_advantages = b_advantages[mb_inds]
            mb_returns = b_returns[mb_inds]
            mb_values = b_values[mb_inds]

            valid_count = mb_valid.sum()

            if valid_count == 0:
                continue

            valid_count = valid_count.float()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio,
                1 - args.clip_coef,
                1 + args.clip_coef,
            )

            pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_valid).sum() / valid_count

            if value_heads > 1:
                newvalue = newvalue.view(-1, args.n_traj, value_heads)
                mb_valid_for_value = mb_valid.unsqueeze(-1).expand_as(newvalue)
                value_valid_count = mb_valid_for_value.sum().float()
            else:
                newvalue = newvalue.view(-1, args.n_traj)
                mb_valid_for_value = mb_valid
                value_valid_count = valid_count

            huber_unclipped = F.smooth_l1_loss(
                newvalue,
                mb_returns,
                reduction="none",
                beta=1.0,
            )

            if args.clip_vloss:
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values,
                    -args.clip_coef,
                    args.clip_coef,
                )

                huber_clipped = F.smooth_l1_loss(
                    v_clipped,
                    mb_returns,
                    reduction="none",
                    beta=1.0,
                )

                v_loss = 0.5 * (
                    torch.max(huber_unclipped, huber_clipped) * mb_valid_for_value
                ).sum() / value_valid_count
            else:
                v_loss = 0.5 * (
                    huber_unclipped * mb_valid_for_value
                ).sum() / value_valid_count

            entropy_loss = (entropy * mb_valid).sum() / valid_count

            ppo_loss = (
                pg_loss
                - args.ent_coef * entropy_loss
                + args.vf_coef * v_loss
            )

            mb_pg_losses.append(pg_loss.item())
            mb_v_losses.append(v_loss.item())
            mb_entropies.append(entropy_loss.item())

            (ppo_loss / accum_steps).backward()
            accum_counter += 1

            if epoch == 0 and accum_counter == 1 and DEBUG:
                gn_backbone = grad_norm(agent.backbone.parameters())
                gn_critic = grad_norm(agent.critic.parameters())

                print(
                    f"[GradSplit pre-clip] "
                    f"backbone={gn_backbone:.6f}, critic={gn_critic:.6f}"
                )
                print(
                    f"[Loss] "
                    f"ppo={ppo_loss:.6f}, "
                    f"pg={pg_loss:.6f}, "
                    f"ent={args.ent_coef * entropy_loss:.6f}, "
                    f"v={v_loss:.6f}"
                )

            is_last_minibatch = (start_mb + envsperbatch) >= num_envs
            do_step = (accum_counter % accum_steps == 0) or is_last_minibatch

            if do_step:
                pre_backbone = nn.utils.clip_grad_norm_(
                    agent.backbone.parameters(),
                    args.max_grad_norm_backbone,
                )

                pre_critic = nn.utils.clip_grad_norm_(
                    agent.critic.parameters(),
                    args.max_grad_norm_critic,
                )

                step_grad_norms_backbone.append(float(pre_backbone))
                step_grad_norms_critic.append(float(pre_critic))

                optim_backbone.step()
                optim_critic.step()

                optim_backbone.zero_grad(set_to_none=True)
                optim_critic.zero_grad(set_to_none=True)

        mb_grad_norms_backbone.extend(step_grad_norms_backbone)
        mb_grad_norms_critic.extend(step_grad_norms_critic)

        if len(epoch_kls) > 0:
            epoch_kl_tensor = torch.stack(epoch_kls)
            mean_kl = float(epoch_kl_tensor.mean().item())
            max_kl = float(epoch_kl_tensor.max().item())
            mb_kls.append(mean_kl)
        else:
            mean_kl = 0.0
            max_kl = 0.0

        if args.target_kl is not None and len(epoch_kls) > 0:

            if mean_kl > args.target_kl:
                stop_early = True
                print(
                    f"[EarlyStop] epoch={epoch} "
                    f"mean_kl={mean_kl:.5f} > target_kl={args.target_kl:.5f}"
                )

        if len(epoch_kls) > 0:
            print(
                f"[EpochKL] epoch={epoch} "
                f"mean_kl={mean_kl:.5f} "
                f"max_kl={max_kl:.5f}"
            )

    return {
        "kl_mean": _mean(mb_kls),
        "kl_p90": _p90(mb_kls),
        "kl_max": _max(mb_kls),
        "pg_loss_mean": _mean(mb_pg_losses),
        "v_loss_mean": _mean(mb_v_losses),
        "entropy_mean": _mean(mb_entropies),
        "gn_bb_mean": _mean(mb_grad_norms_backbone),
        "gn_bb_max": _max(mb_grad_norms_backbone),
        "gn_v_mean": _mean(mb_grad_norms_critic),
        "gn_v_max": _max(mb_grad_norms_critic),
    }


# =========================================================
# Diagnostics
# =========================================================
def _print_rollout_diagnostics(
    rollout,
    args,
    num_envs,
    customer_numbers,
    charging_stations_numbers,
    update_step,
    lambda_fail,
    DEBUG,
):
    if not DEBUG:
        return

    rewards = rollout["rewards"]
    valid_masks = rollout["valid_masks"]
    objectives = rollout["objectives"]
    actions = rollout["actions"]
    valid_step = rollout["valid_step"]

    visu_actions = actions.reshape((valid_step, -1)).cpu().numpy().copy()
    visu_actions[visu_actions == 0] = customer_numbers + 1
    visu_actions[visu_actions < 1 + customer_numbers] = 1
    visu_actions[visu_actions >= 1 + customer_numbers] = 0

    cus_count_per_traj = visu_actions.sum(axis=0)
    finish_flags = cus_count_per_traj == customer_numbers
    success_rate = float(finish_flags.mean())

    print("\n------------------ Training Record ------------------")
    print(f"Epoch: {update_step}/{args.num_updates}")
    print(
        "Customer Numbers:",
        customer_numbers,
        "Charging Stations Numbers:",
        charging_stations_numbers,
    )
    print(f"Avg Customer Visits: {cus_count_per_traj.mean():.2f}")
    print(
        f"Finish Rate: {finish_flags.sum()}/{finish_flags.size} = "
        f"{success_rate:.3f}"
    )
    group_label = getattr(args, "_current_train_config_label", None)
    if group_label:
        group_idx = getattr(args, "_current_train_config_idx", None)
        group_total = getattr(args, "_current_train_config_total", None)
        group_step = getattr(args, "_current_train_config_step", update_step)
        if group_idx is not None and group_total is not None:
            print(
                f"Train Config Group: step={group_step} "
                f"idx={int(group_idx) + 1}/{group_total} | {group_label}"
            )
        else:
            print(f"Train Config Group: step={group_step} | {group_label}")
    print(f"Current lambda_fail before update: {lambda_fail:.3f}")
    print("----------------------------------------------------")
    quiet_teacher_diag = bool(getattr(args, "quiet_teacher_diagnostics", False))

    if not quiet_teacher_diag:
        with torch.no_grad():
            student_success = torch.isfinite(objectives)
            student_fr_by_env = student_success.float().mean(dim=1)
            student_obj_by_env = objectives.min(dim=1).values

            valid_student_mask = torch.isfinite(student_obj_by_env)
            valid_student_obj = student_obj_by_env[valid_student_mask]
            valid_student_fr = student_fr_by_env[valid_student_mask]

            if valid_student_obj.numel() > 0:
                print(
                    f"[StudentObjDiag] "
                    f"valid={valid_student_obj.numel()}/{num_envs} | "
                    f"FR_mean={valid_student_fr.mean().item():.4f} | "
                    f"FR_min={valid_student_fr.min().item():.4f} | "
                    f"obj_mean={valid_student_obj.mean().item():.4f} | "
                    f"obj_min={valid_student_obj.min().item():.4f} | "
                    f"obj_max={valid_student_obj.max().item():.4f}"
                )
            else:
                print(f"[StudentObjDiag] valid=0/{num_envs} | no finite student objective")

    if (not quiet_teacher_diag) and len(rollout.get("success_bonus_coef_all", [])) > 0:
        coef_stack = np.stack(rollout["success_bonus_coef_all"], axis=0)
        finite_coef = coef_stack[np.isfinite(coef_stack)]

        if finite_coef.size > 0:
            print(
                f"[SuccessCoefDiag] "
                f"mean={finite_coef.mean():.6f}, "
                f"min={finite_coef.min():.6f}, "
                f"max={finite_coef.max():.6f}, "
                f"finite_ratio={finite_coef.size / coef_stack.size:.4f}"
            )

    if (not quiet_teacher_diag) and len(rollout.get("teacher_gap_all", [])) > 0:
        gap_stack = np.stack(rollout["teacher_gap_all"], axis=0)
        finite_gap = gap_stack[np.isfinite(gap_stack)]

        if finite_gap.size > 0:
            print(
                f"[TeacherGapDiag] "
                f"mean={finite_gap.mean():.6f}, "
                f"min={finite_gap.min():.6f}, "
                f"max={finite_gap.max():.6f}, "
                f"finite_ratio={finite_gap.size / gap_stack.size:.4f}"
            )

    if (not quiet_teacher_diag) and len(rollout.get("r_teacher_dense_all", [])) > 0:
        dense_stack = np.stack(rollout["r_teacher_dense_all"], axis=0)
        finite_dense = dense_stack[np.isfinite(dense_stack)]

        if finite_dense.size > 0:
            nonzero = finite_dense[np.abs(finite_dense) > 1e-8]
            print(
                f"[TeacherDenseDiag] "
                f"nonzero={nonzero.size}/{finite_dense.size} "
                f"= {nonzero.size / max(1, finite_dense.size):.4f} | "
                f"mean_nonzero={nonzero.mean():.8f} "
                if nonzero.size > 0
                else (
                    f"[TeacherDenseDiag] "
                    f"nonzero=0/{finite_dense.size} = 0.0000 | "
                    f"mean_nonzero=0.00000000 "
                )
            )

    if (not quiet_teacher_diag) and len(rollout.get("r_teacher_route_all", [])) > 0:
        route_stack = np.stack(rollout["r_teacher_route_all"], axis=0)
        finite_route = route_stack[np.isfinite(route_stack)]

        if finite_route.size > 0:
            nonzero = finite_route[np.abs(finite_route) > 1e-8]
            print(
                f"[TeacherRouteDiag] "
                f"nonzero={nonzero.size}/{finite_route.size} "
                f"= {nonzero.size / max(1, finite_route.size):.4f} | "
                f"mean={finite_route.mean():.8f} | "
                f"abs_mean={np.abs(finite_route).mean():.8f} | "
                f"min={finite_route.min():.8f} | "
                f"max={finite_route.max():.8f}"
            )

    if len(rollout.get("r_bsrs_all", [])) > 0:
        bsrs_stack = np.stack(rollout["r_bsrs_all"], axis=0)
        finite_bsrs = bsrs_stack[np.isfinite(bsrs_stack)]

        if finite_bsrs.size > 0:
            nonzero = finite_bsrs[np.abs(finite_bsrs) > 1e-8]
            if bool(getattr(args, "use_split_bsrs", False)):
                eta_text = (
                    f"split(customer={float(getattr(args, 'bsrs_customer_eta', 0.0)):.6f}, "
                    f"rs={float(getattr(args, 'bsrs_rs_eta', 0.0)):.6f})"
                )
            else:
                eta_text = f"{float(getattr(args, 'bsrs_eta', 0.0)):.6f}"
            print(
                f"[BSRSDiag] "
                f"eta={eta_text} | "
                f"nonzero={nonzero.size}/{finite_bsrs.size} "
                f"= {nonzero.size / max(1, finite_bsrs.size):.4f} | "
                f"mean={finite_bsrs.mean():.8f} | "
                f"abs_mean={np.abs(finite_bsrs).mean():.8f} | "
                f"min={finite_bsrs.min():.8f} | "
                f"max={finite_bsrs.max():.8f}"
            )

    for key, label in (
        ("r_bsrs_customer_all", "BSRSCustomerDiag"),
        ("r_bsrs_rs_all", "BSRSRSDiag"),
    ):
        if len(rollout.get(key, [])) == 0:
            continue
        bsrs_stack = np.stack(rollout[key], axis=0)
        finite_bsrs = bsrs_stack[np.isfinite(bsrs_stack)]

        if finite_bsrs.size > 0:
            nonzero = finite_bsrs[np.abs(finite_bsrs) > 1e-8]
            print(
                f"[{label}] "
                f"nonzero={nonzero.size}/{finite_bsrs.size} "
                f"= {nonzero.size / max(1, finite_bsrs.size):.4f} | "
                f"mean={finite_bsrs.mean():.8f} | "
                f"abs_mean={np.abs(finite_bsrs).mean():.8f} | "
                f"min={finite_bsrs.min():.8f} | "
                f"max={finite_bsrs.max():.8f}"
            )


def _compute_success_rate_from_rollout(rollout, customer_numbers):
    actions = rollout["actions"]
    valid_step = rollout["valid_step"]

    visu_actions = actions.reshape((valid_step, -1)).cpu().numpy().copy()
    visu_actions[visu_actions == 0] = customer_numbers + 1
    visu_actions[visu_actions < 1 + customer_numbers] = 1
    visu_actions[visu_actions >= 1 + customer_numbers] = 0

    cus_count_per_traj = visu_actions.sum(axis=0)
    finish_flags = cus_count_per_traj == customer_numbers

    return float(finish_flags.mean())


# =========================================================
# Evaluation
# =========================================================
def _run_evaluation(
    agent,
    test_envs,
    args,
    test_num_cus,
    test_num_cs,
    batch_size,
    test_max_step,
    save_dir,
    best_reward,
    decode_mode=None,
    eval_label=None,
    num_agents=None,
    update_best=True,
):
    t_eval_start = time.time()

    agent.eval()
    record_info = []
    decode_mode = str(decode_mode or getattr(args, "eval_decode_mode", "greedy")).lower()
    eval_label = str(eval_label or decode_mode)
    num_agents = int(num_agents or getattr(args, "test_agent", 1))

    record_done = np.zeros((batch_size, num_agents), dtype=np.float32)
    record_cs = np.zeros((batch_size, num_agents), dtype=np.float32)
    record_cus = np.zeros((batch_size, num_agents), dtype=np.float32)
    episode_returns = np.zeros((batch_size, num_agents), dtype=np.float32)

    action_history = [
        [["D"] for _ in range(num_agents)]
        for _ in range(batch_size)
    ]

    test_obs = test_envs.reset()

    for step in range(test_max_step):
        with torch.no_grad():
            if decode_mode == "sampling":
                action, _, _, _ = agent.get_action_and_value(test_obs)
            else:
                action, _ = agent(test_obs)

        action = action.cpu().numpy()
        active_before = record_done == 0
        test_obs, test_reward, test_done, test_info = test_envs.step(action)
        test_reward = np.asarray(test_reward, dtype=np.float32).reshape(
            episode_returns.shape
        )
        test_done = np.asarray(test_done, dtype=bool).reshape(record_done.shape)
        episode_returns[active_before] += test_reward[active_before]

        finish_idx = active_before & test_done
        record_done[finish_idx] = step + 1

        record_cs[(action > test_num_cus) & active_before] += 1
        record_cus[(action <= test_num_cus) & (action > 0) & active_before] += 1

        for b in range(batch_size):
            for t in range(num_agents):
                if not active_before[b, t]:
                    continue

                a = int(action[b, t])

                if a == 0:
                    if action_history[b][t][-1] != "D":
                        action_history[b][t].append("D")
                elif a > test_num_cus:
                    action_history[b][t].append(f"R{a - test_num_cus}")
                else:
                    action_history[b][t].append(f"C{a}")

        for item in test_info:
            if "episode" in item.keys():
                record_info.append(item)

        if test_done.all():
            break

    traj_finished = record_done > 0
    completed_traj_rewards = episode_returns[traj_finished]
    traj_avg_reward = (
        float(np.mean(completed_traj_rewards))
        if completed_traj_rewards.size > 0
        else float("-inf")
    )
    traj_episode_count = int(traj_finished.sum())

    solved_instances = traj_finished.any(axis=1)
    solved_count = int(solved_instances.sum())
    best_instance_rewards = np.full(batch_size, np.nan, dtype=np.float32)
    best_instance_steps = np.full(batch_size, np.nan, dtype=np.float32)
    best_instance_cs = np.full(batch_size, np.nan, dtype=np.float32)

    for b in np.where(solved_instances)[0]:
        masked_rewards = np.where(traj_finished[b], episode_returns[b], -np.inf)
        best_t = int(np.argmax(masked_rewards))
        best_instance_rewards[b] = float(masked_rewards[best_t])
        best_instance_steps[b] = float(record_done[b, best_t])
        best_instance_cs[b] = float(record_cs[b, best_t])

    use_pomo_best = decode_mode == "sampling" and num_agents > 1
    if use_pomo_best:
        avg_reward = (
            float(np.nanmean(best_instance_rewards))
            if solved_count > 0
            else float("-inf")
        )
        effective_episodes = solved_count
        avg_done_step = (
            float(np.nanmean(best_instance_steps))
            if solved_count > 0
            else 0.0
        )
        avg_cs = (
            float(np.nanmean(best_instance_cs))
            if solved_count > 0
            else 0.0
        )
        aggregation = f"best_of_{num_agents}"
    else:
        avg_reward = traj_avg_reward
        effective_episodes = traj_episode_count
        avg_done_step = (
            float(record_done[traj_finished].mean())
            if traj_episode_count > 0
            else 0.0
        )
        avg_cs = (
            float(record_cs[traj_finished].mean())
            if traj_episode_count > 0
            else 0.0
        )
        aggregation = "avg"

    print(f"----- Evaluation Result ({eval_label}) -----")
    print(
        "Number of Customers:",
        test_num_cus,
        "Number of Charging Stations:",
        test_num_cs,
    )
    print(
        f"Evaluation over {effective_episodes} episodes: {avg_reward:.3f}, "
        f"Step: {step}, "
        f"Avg Done Step: {avg_done_step:.2f}, "
        f"#CS visited: {avg_cs:.2f}"
    )
    print(
        f"[EvalSummary] mode={eval_label} episodes={effective_episodes} "
        f"avg_reward={avg_reward:.6f} step={step} "
        f"avg_done_step={avg_done_step:.6f} "
        f"avg_cs={avg_cs:.6f} solved_rate={solved_count / batch_size:.6f} "
        f"aggregation={aggregation}"
    )

    if use_pomo_best:
        print(
            f"[EvalSummary] mode={eval_label}_avg episodes={traj_episode_count} "
            f"avg_reward={traj_avg_reward:.6f} step={step} "
            f"avg_done_step="
            f"{(float(record_done[traj_finished].mean()) if traj_episode_count > 0 else 0.0):.6f} "
            f"avg_cs="
            f"{(float(record_cs[traj_finished].mean()) if traj_episode_count > 0 else 0.0):.6f} "
            f"solved_rate={traj_episode_count / float(batch_size * num_agents):.6f} "
            f"aggregation=trajectory_avg"
        )

    if args.debug_test:
        print("Sample finished trajectory:")
        print("->".join(action_history[0][0]))

    unfinished_mask = record_done == 0

    if unfinished_mask.any():
        b_idx, t_idx = np.argwhere(unfinished_mask)[0]

        print("[Eval] Found unfinished trajectory:")
        print(f"  instance_idx={b_idx}, traj_idx={t_idx}")
        print(f"  visited_customers={record_cus[b_idx, t_idx]:.0f}/{test_num_cus}")
        print(f"  visited_charging_stations={record_cs[b_idx, t_idx]:.0f}")
        print(f"  done_step=NOT_FINISHED (max_step={test_max_step})")
        print("  trajectory:")
        print("  " + "->".join(action_history[b_idx][t_idx]))
    else:
        print("[Eval] All trajectories finished.")

    print("Eval cost : {:.4f}s".format(time.time() - t_eval_start))

    os.makedirs(save_dir, exist_ok=True)

    if update_best and avg_reward > best_reward and effective_episodes == batch_size:
        best_reward = avg_reward

        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(agent.state_dict(), best_path)

        print(
            f"[Checkpoint] Saved current run best model. "
            f"best_reward={best_reward:.3f}"
        )

    torch.save(agent.state_dict(), os.path.join(save_dir, "cur_model.pth"))

    return best_reward


# =========================================================
# Main training loop
# =========================================================
def train(args):
    print("---------------- Training Info ---------------------")
    for key, value in vars(args).items():
        print(key, value)
    print("----------------------------------------------------")

    _set_global_seeds(args.seed, deterministic=True)
    print(f"[Seed] global_seed={int(args.seed)} deterministic_cudnn=True")

    try:
        gym.envs.register(
            id=args.env_id,
            entry_point=args.env_entry_point,
        )
    except Exception as e:
        print(f"[GymRegister] skipped or already registered: {e}")

    lambda_fail = args.lambda_fail_init

    if getattr(args, "cuda", True) and torch.cuda.is_available():
        device = f"cuda:{args.cuda_id}"
    else:
        device = "cpu"

    # =========================================================
    # Student model
    # =========================================================
    value_heads = (
        len(_decomposed_component_names(args))
        if bool(getattr(args, "use_decomposed_reward_adv", False))
        else 1
    )
    agent = Agent(
        device=device,
        name=args.problem,
        tanh_clipping=args.tanh_clipping,
        n_encode_layers=args.n_encode_layers,
        value_heads=value_heads,
        use_candidate_dynamic_embedding=bool(
            getattr(args, "use_candidate_dynamic_embedding", True)
        ),
    ).to(device)

    init_ckpt_path = getattr(args, "init_ckpt_path", None)
    if init_ckpt_path is not None and os.path.exists(init_ckpt_path):
        ckpt = _safe_torch_load(init_ckpt_path, device)
        _load_state_dict_compatible(
            agent,
            ckpt,
            strict=(
                not bool(getattr(args, "use_decomposed_reward_adv", False))
                and not bool(getattr(args, "use_candidate_dynamic_embedding", True))
            ),
        )
        print(f"[Init] Loaded student checkpoint from: {init_ckpt_path}")

    # =========================================================
    # Optimizer
    # =========================================================
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

    # =========================================================
    # Config
    # =========================================================
    test_num_cus = args.train_cus_num
    test_num_cs = args.train_cs_num
    customer_numbers = args.train_cus_num
    charging_stations_numbers = args.train_cs_num

    num_updates = args.num_updates
    num_steps = args.num_steps

    model_size_name = f"Cus_{test_num_cus}_CS_{test_num_cs}"
    save_dir = os.path.join(args.save_dir, model_size_name)
    os.makedirs(save_dir, exist_ok=True)

    best_reward = float("-inf")

    perturb_dict = Config(args.perturb_dict_path).setup_env_parameters()
    config = Config(args.config_path)

    test_max_step = num_steps

    # =========================================================
    # ALNS buffer
    # =========================================================
    alns_records = []
    alns_rng = np.random.default_rng(int(args.seed) + 9999)

    if bool(getattr(args, "use_alns_teacher", False)):
        alns_records = _load_alns_buffer_from_dir(
            buffer_dir=getattr(args, "alns_buffer_dir", None),
            progress_name=getattr(args, "alns_progress_name", "buffer_progress.pkl"),
            instance_pickle_name=getattr(args, "alns_instance_pickle_name", "evrptw_50C_12R.pkl"),
            obj_scale=getattr(args, "alns_obj_scale", 100.0),
        )

        if len(alns_records) == 0:
            print("[ALNSTeacher] use_alns_teacher=True but no valid ALNS records were loaded.")

    # =========================================================
    # Eval data / envs
    # =========================================================
    eval_data = None
    if getattr(args, "eval_data_path", None) is not None:
        with open(args.eval_data_path, "rb") as f:
            eval_data = pickle.load(f)

    if eval_data is None:
        raise ValueError("Evaluation requires eval_data, but eval_data is None.")

    num_test_envs = len(eval_data)
    eval_batch_size = args.eval_batch_size

    if eval_batch_size > num_test_envs:
        raise ValueError(
            f"eval_batch_size={eval_batch_size} > len(eval_data)={num_test_envs}"
        )

    eval_env_ids = np.random.choice(
        num_test_envs,
        size=eval_batch_size,
        replace=False,
    )

    batch_size = len(eval_env_ids)

    test_envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                int(args.seed + i),
                cfg={
                    "env_mode": "eval",
                    "config": config,
                    "n_traj": args.test_agent,
                    "eval_data": eval_data[env_id],
                    "max_route_events": int(getattr(args, "max_route_events", 16)),
                    "max_completed_routes": int(
                        getattr(args, "max_completed_routes", 4)
                    ),
                },
            )
            for i, env_id in enumerate(eval_env_ids)
        ]
    )

    greedy_test_envs = None
    if bool(getattr(args, "eval_greedy_too", False)):
        greedy_test_envs = SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    int(args.seed + 700000 + i),
                    cfg={
                        "env_mode": "eval",
                        "config": config,
                        "n_traj": 1,
                        "eval_data": eval_data[env_id],
                        "max_route_events": int(getattr(args, "max_route_events", 16)),
                        "max_completed_routes": int(
                            getattr(args, "max_completed_routes", 4)
                        ),
                    },
                )
                for i, env_id in enumerate(eval_env_ids)
            ]
        )

    # =========================================================
    # Main training loop
    # =========================================================
    for update_step in tqdm(range(num_updates)):
        DEBUG = args.debug
        t0 = time.time()

        use_alns_this_update = (
            bool(getattr(args, "use_alns_teacher", False))
            and len(alns_records) > 0
            and int(getattr(args, "alns_teacher_freq", 0)) > 0
            and update_step > 0
            and update_step % int(args.alns_teacher_freq) == 0
        )

        envs = None

        try:
            # -----------------------------------------------------
            # Build envs
            # -----------------------------------------------------
            args._current_train_config_label = None
            args._current_train_config_idx = None
            args._current_train_config_total = None
            args._current_train_config_step = update_step

            if use_alns_this_update:
                sampled_records = _sample_records(
                    alns_records,
                    batch_size=int(
                        getattr(args, "alns_teacher_batch_size", args.num_envs)
                    ),
                    rng=alns_rng,
                )

                if len(sampled_records) == 0:
                    print("[ALNSTeacher] sampled empty batch; fallback to generated envs.")
                    use_alns_this_update = False

                if (
                    use_alns_this_update
                    and bool(getattr(args, "alns_teacher_filter_better", True))
                ):
                    sampled_records, _filter_info = (
                        _filter_alns_records_by_student_probe(
                            agent=agent,
                            args=args,
                            config=config,
                            sampled_records=sampled_records,
                            device=device,
                            num_steps=num_steps,
                            update_step=update_step,
                            DEBUG=DEBUG,
                        )
                    )

                    if len(sampled_records) == 0:
                        print(
                            "[ALNSFilter] no ALNS-better records remain; "
                            "fallback to generated envs."
                        )
                        use_alns_this_update = False

                if use_alns_this_update:
                    envs, sampled_records = _make_alns_train_envs(
                        args=args,
                        config=config,
                        update_step=update_step,
                        sampled_records=sampled_records,
                        use_teacher_reward=True,
                    )

                    if envs is None:
                        print("[ALNSTeacher] empty filtered batch; fallback to generated envs.")
                        use_alns_this_update = False

            if not use_alns_this_update:
                envs = _make_generated_train_envs(
                    args=args,
                    config=config,
                    perturb_dict=perturb_dict,
                    customer_numbers=customer_numbers,
                    charging_stations_numbers=charging_stations_numbers,
                    lambda_fail=lambda_fail,
                    update_step=update_step,
                )

            num_envs_cur = len(envs.envs)

            # -----------------------------------------------------
            # Teacher setup
            # -----------------------------------------------------
            if use_alns_this_update:
                next_obs = envs.reset()

                if DEBUG:
                    teacher_objs = np.asarray(
                        [r["teacher_obj"] for r in sampled_records],
                        dtype=np.float32,
                    )
                    print(
                        f"[UpdateMode] ALNS teacher | "
                        f"update={update_step} | "
                        f"batch={num_envs_cur} | "
                        f"teacher_obj_mean={teacher_objs.mean():.4f} | "
                        f"teacher_obj_min={teacher_objs.min():.4f} | "
                        f"teacher_obj_max={teacher_objs.max():.4f}"
                    )

            else:
                next_obs = envs.reset()
                _set_attr_all(envs, "use_teacher_reward", False)

                if DEBUG:
                    print(f"[UpdateMode] online PPO only | update={update_step}")

            t1 = time.time()

            # -----------------------------------------------------
            # Student rollout
            # -----------------------------------------------------
            rollout = _student_rollout(
                agent=agent,
                envs=envs,
                args=args,
                device=device,
                num_steps=num_steps,
                initial_obs=next_obs,
                update_step=update_step,
                enable_bsrs=True,
            )

            t2 = time.time()

            _print_rollout_diagnostics(
                rollout=rollout,
                args=args,
                num_envs=num_envs_cur,
                customer_numbers=customer_numbers,
                charging_stations_numbers=charging_stations_numbers,
                update_step=update_step,
                lambda_fail=lambda_fail,
                DEBUG=DEBUG,
            )

            success_rate = _compute_success_rate_from_rollout(
                rollout=rollout,
                customer_numbers=customer_numbers,
            )
            args.current_rollout_success_rate = success_rate

            lambda_fail = update_lambda_fail(
                lambda_fail=lambda_fail,
                success_rate=success_rate,
                target_success=args.target_success,
                lambda_max=args.lambda_max,
                lr_up=args.lambda_lr_up,
                lr_down=args.lambda_lr_down,
                tolerance=args.lambda_tolerance,
            )

            if DEBUG:
                print(f"[LambdaFail] updated={lambda_fail:.3f}")

            # -----------------------------------------------------
            # GAE / PPO update
            # -----------------------------------------------------
            t_gae_start = time.time()
            advantages, returns = _compute_gae_and_returns(
                agent=agent,
                rollout=rollout,
                args=args,
                device=device,
            )
            t_gae_end = time.time()

            if DEBUG and bool(getattr(args, "use_decomposed_reward_adv", False)):
                weights = getattr(args, "last_adv_component_weights", None)
                if weights:
                    mix = getattr(args, "last_adv_adaptive_mix", None)
                    mix_text = "static" if mix is None else f"mix={mix:.4f}"
                    weight_text = " ".join(
                        f"{name}={value:.4f}"
                        for name, value in weights.items()
                    )
                    print(
                        f"[AdvWeights] success_rate={success_rate:.4f} "
                        f"{mix_text} {weight_text}"
                    )

            ppo_info = _ppo_update(
                agent=agent,
                optim_backbone=optim_backbone,
                optim_critic=optim_critic,
                envs=envs,
                rollout=rollout,
                advantages=advantages,
                returns=returns,
                args=args,
                DEBUG=DEBUG,
            )
            t_ppo_end = time.time()

            # -----------------------------------------------------
            # ALNS route preference / imitation updates
            # -----------------------------------------------------
            if use_alns_this_update:
                _alns_preference_update(
                    agent=agent,
                    optim_backbone=optim_backbone,
                    args=args,
                    config=config,
                    records=sampled_records,
                    device=device,
                    update_step=update_step,
                    DEBUG=DEBUG,
                )
                _alns_behavior_cloning_update(
                    agent=agent,
                    optim_backbone=optim_backbone,
                    args=args,
                    config=config,
                    records=sampled_records,
                    device=device,
                    update_step=update_step,
                    DEBUG=DEBUG,
                )
            t_alns_update_end = time.time()

            t3 = time.time()

            if DEBUG:
                print(
                    "[Time] "
                    f"Env Setup: {t1 - t0:.4f}s | "
                    f"Rollout: {t2 - t1:.4f}s | "
                    f"GAE: {t_gae_end - t_gae_start:.4f}s | "
                    f"PPO: {t_ppo_end - t_gae_end:.4f}s | "
                    f"ALNS Update: {t_alns_update_end - t_ppo_end:.4f}s | "
                    f"Post Update: {t3 - t_alns_update_end:.4f}s | "
                    f"Update Total: {t3 - t2:.4f}s"
                )

            # -----------------------------------------------------
            # Evaluation
            # -----------------------------------------------------
            if (update_step + 1) % args.eval_freq == 0:
                best_reward = _run_evaluation(
                    agent=agent,
                    test_envs=test_envs,
                    args=args,
                    test_num_cus=test_num_cus,
                    test_num_cs=test_num_cs,
                    batch_size=batch_size,
                    test_max_step=test_max_step,
                    save_dir=save_dir,
                    best_reward=best_reward,
                    decode_mode=getattr(args, "eval_decode_mode", "greedy"),
                    eval_label=(
                        "pomo"
                        if str(getattr(args, "eval_decode_mode", "greedy")).lower() == "sampling"
                        else "greedy"
                    ),
                    num_agents=int(getattr(args, "test_agent", 1)),
                    update_best=True,
                )
                if greedy_test_envs is not None:
                    _run_evaluation(
                        agent=agent,
                        test_envs=greedy_test_envs,
                        args=args,
                        test_num_cus=test_num_cus,
                        test_num_cs=test_num_cs,
                        batch_size=batch_size,
                        test_max_step=test_max_step,
                        save_dir=save_dir,
                        best_reward=best_reward,
                        decode_mode="greedy",
                        eval_label="greedy",
                        num_agents=1,
                        update_best=False,
                    )

        finally:
            if envs is not None:
                envs.close()

    test_envs.close()
    if greedy_test_envs is not None:
        greedy_test_envs.close()
