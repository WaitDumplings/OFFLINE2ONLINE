# evrptw_gen/policies/timewindows.py
from __future__ import annotations
from typing import Dict, Protocol
from collections import Counter
import numpy as np
from scipy.stats import truncnorm

testset_name = "time_window_policy"
config_list_name = "time_window_policy_list"
config_type_name = "time_window_policy_config"

class TimeWindowPolicy(Protocol):
    def build(
        self,
        env: Dict,
        depot_pos: np.ndarray,
        cs_pos: np.ndarray,
        cus_pos: np.ndarray,
        time_depot_to_css: np.ndarray,
        time_depot_to_cuss: np.ndarray,
        time_cus_to_depot: np.ndarray,
        rng: np.random.Generator,
        tw_format: str
    ) -> np.ndarray:
        ...


# ---- Two concrete strategies: Narrow / Wide ----
class NarrowTWPolicy:
    NAME = "narrow"

    def build(
        self,
        env,
        t_earliest,          # (N,)
        t_latest,            # (N,)
        service_time,         # (N,) or scalar
        rng,
        tw_format = "hours"
    ):
        """
        Generate (N, 2) time windows under 'Narrow' policy.

        Assumptions:
        - All time quantities are in the same unit as working/instance times in env (typically minutes).
        - time_depot_to_cuss[i] is the earliest-arrival travel time from depot to customer i.
        - service_time can be scalar or length-N array (minutes).
        - time_route_instance optionally provides a per-customer margin to finish the route before instance_endTime.
        """
        cfg = env[config_type_name][self.NAME]
        alpha = float(cfg.get("alpha", 0.3))  # mean width factor of feasible span
        beta  = float(cfg.get("beta",  0.05)) # std factor of feasible span
        round_ndigits = int(cfg.get("round_ndigits", 2))

        # N = int(env.get("num_customers", service_time.shape[0]))
        N = env["num_customers"]
        lb = float(env["working_startTime"])
        ub = float(env["working_endTime"])
 
        if np.any(t_earliest > t_latest + 1e-6):
            raise ValueError("t_earliest cannot be greater than t_latest for any customer.")

        centers = t_earliest + (t_latest - t_earliest) * rng.random(len(t_earliest))
        span = t_latest - t_earliest

        # target width ~ Normal(alpha*span, beta*span)
        mean_w = alpha * span
        std_w  = beta  * span

        a, b = (0 - mean_w) / std_w, np.inf

        widths = truncnorm.rvs(a, b, loc=mean_w, scale=std_w, size=N, random_state=rng)
        # widths = rng.normal(loc=mean_w, scale=std_w)
        # enforce width >= tw_min_width but also not exceed span (otherwise clamp to span)
        # widths = np.clip(widths, a_min=tw_min_width, a_max=np.maximum(span, tw_min_width))

        # compose windows, then intersect with [lb, ub]
        starts = centers - 0.5 * widths
        ends   = centers + 0.5 * widths

        # intersect with feasible [lb, ub]
        starts = np.maximum(starts, lb)
        ends   = np.minimum(ends, ub)

        if tw_format == "hours":
            starts /= 60
            ends /= 60
            
        tw = np.stack([np.round(starts, round_ndigits), np.round(ends, round_ndigits)], axis=1)
        return tw



class WideTWPolicy:
    NAME = "wide"

    def build(
        self,
        env,
        t_earliest,          # (N,)
        t_latest,            # (N,)
        service_time,         # (N,) or scalar
        rng,
        tw_format = "hours"
    ):
        """
        Generate (N, 2) time windows under 'Narrow' policy.

        Assumptions:
        - All time quantities are in the same unit as working/instance times in env (typically minutes).
        - time_depot_to_cuss[i] is the earliest-arrival travel time from depot to customer i.
        - service_time can be scalar or length-N array (minutes).
        - time_route_instance optionally provides a per-customer margin to finish the route before instance_endTime.
        """
        cfg = env.get("time_window_wide_config", {})
        alpha = float(cfg.get("alpha", 0.3))  # mean width factor of feasible span
        beta  = float(cfg.get("beta",  0.05)) # std factor of feasible span
        tw_min_width = float(cfg.get("min_width", 1.0))  # absolute minimum width (minutes)
        round_ndigits = int(cfg.get("round_ndigits", 2))

        N = int(env.get("num_customers", service_time.shape[0]))

        if env['num_customers'] is not None and env['num_customers'] != service_time.shape[0]:
            breakpoint()
        lb = float(env["working_startTime"])
        ub = float(env["working_endTime"])
 
        if np.any(t_earliest > t_latest + 1e-6):
            raise ValueError("t_earliest cannot be greater than t_latest for any customer.")

        centers = t_earliest + (t_latest - t_earliest) * rng.random(len(t_earliest))
        span = t_latest - t_earliest

        # target width ~ Normal(alpha*span, beta*span)
        mean_w = alpha * span
        std_w  = beta  * span
        a, b = (0 - mean_w) / std_w, np.inf
        widths = truncnorm.rvs(a, b, loc=mean_w, scale=std_w, size=N, random_state=rng)

        # widths = rng.normal(loc=mean_w, scale=std_w)
        # enforce width >= tw_min_width but also not exceed span (otherwise clamp to span)
        # widths = np.clip(widths, a_min=tw_min_width, a_max=np.maximum(span, tw_min_width))

        # compose windows, then intersect with [lb, ub]
        starts = centers - 0.5 * widths
        ends   = centers + 0.5 * widths

        # intersect with feasible [lb, ub]
        starts = np.maximum(starts, lb)
        ends   = np.minimum(ends, ub)

        if tw_format == "hours":
            starts /= 60
            ends /= 60

        tw = np.stack([np.round(starts, round_ndigits), np.round(ends, round_ndigits)], axis=1)
        return tw

# ---- Factory: only Narrow / Wide are currently supported ----
class TimeWindowPolicies:
    REGISTRY = {
        "narrow": NarrowTWPolicy,
        "wide": WideTWPolicy,
    }

    @classmethod
    def _sample_choice(cls, env: Dict, rng: np.random.Generator = None) -> str:
        """
        Select the time window type with the following priority:
        1) If 'time_window_type' is present in env, use it (for deterministic testing).
        2) Otherwise, sample from 'time_window_type_distribution' according to probabilities.
        3) If both missing or invalid, raise ValueError.
        """
        if testset_name in env and env[testset_name] is not None:
            return env[testset_name]

        type_list = env.get(config_list_name, None)
        if Counter(type_list) != Counter(cls.REGISTRY.keys()):
            raise ValueError(f"Unknown policy: {type_list}")
        score_list = []
        for key in type_list:
            cfg = env[config_type_name][key]
            score = cfg.get("score", 0)
            score_list.append(score)

        if rng is None:
            rng = np.random.default_rng()

        choice = rng.choice(
            type_list,
            p=np.array(score_list) / np.sum(score_list)
        )
        return choice

    @classmethod
    def from_env(cls, env: Dict, rng: np.random.Generator = None) -> TimeWindowPolicy:
        choice = cls._sample_choice(env, rng=rng)
        return cls.REGISTRY[choice]()
