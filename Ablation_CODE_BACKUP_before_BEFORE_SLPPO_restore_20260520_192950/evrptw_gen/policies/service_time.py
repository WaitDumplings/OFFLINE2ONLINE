# evrptw_gen/policies/servicetime.py
from __future__ import annotations
from typing import Dict, Protocol
from collections import Counter
import numpy as np

testset_name = "service_time_policy"
config_list_name = "service_time_policy_list"
config_type_name = "service_time_policy_config"

class ServiceTimePolicies(Protocol):
    def build(
        self,
        env: Dict,
        num_customers: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (num_customers,), dtype=float or int (minutes).
        """
        ...
class CargoWeightBased_ServiceTimePolicy:
    """
    Service time based on cargo weight.
    Config layout in env:
        env['service_time_cargo_weight'] = {
            'base_service_time': <float>,      # base service time (minutes)
            'weight_coefficient': <float>,     # coefficient for weight impact
        }
    """
    NAME = "cargoweight"

    def build(self, env: Dict, num_customers: int, rng = np.random.Generator, demand = None, unit = "hours") -> np.ndarray:
        cfg = env[config_type_name][self.NAME]
        if demand is None:
            raise ValueError("Demand must be specified in env for CargoWeightBased_ServiceTimePolicy.")
        
        base_service_time = float(cfg.get("base_service_time", 3.0))
        weight_coefficient = float(cfg.get("weight_coefficient", 30.0))

        # Calculate service time based on demand (weight)
        service_times = base_service_time + weight_coefficient * demand

        if unit == "hours":
            time_multiplier = 60.0  # convert hours to minutes
        elif unit == "minutes":
            time_multiplier = 1.0   # already in minutes
        else:
            raise ValueError(f"unit issue, cannot recognize unit {unit}!")

        return service_times.astype(np.float32) / time_multiplier

class Random_ServiceTimePolicy:
    """
    Uniform random service time in [min, max].
    Config layout in env:
        env['service_time_random'] = {
            'min': <float|int>,      # inclusive lower bound (minutes)
            'max': <float|int>,      # inclusive upper bound (minutes)
            'integer': True|False,   # if True -> integers; else floats
            # optional:
            # 'round_ndigits': 2      # when integer=False, round to ndigits
        }
    """
    NAME = "random"

    def build(self, env: Dict, num_customers: int, rng: np.random.Generator, unit = "hours", **kwargs) -> np.ndarray:
        cfg = env[config_type_name][self.NAME]

        if unit == "hours":
            time_multiplier = 60.0  # convert hours to minutes
        elif unit == "minutes":
            time_multiplier = 1.0   # already in minutes
        else:
            raise ValueError(f"unit issue, cannot recognize unit {unit}!")

        service_time_range = cfg.get('st_range')
        lo, hi = service_time_range
        integer = bool(cfg.get("integer", True))

        if integer:
            # inclusive upper bound for integers
            lo_i = int(np.floor(lo))
            hi_i = int(np.ceil(hi))
            if hi_i < lo_i:
                hi_i = lo_i
            st = rng.integers(lo_i, hi_i + 1, size=num_customers, dtype=np.int32)
            return st.astype(np.int32) / time_multiplier

        # float minutes
        st = rng.uniform(lo, hi, size=num_customers).astype(np.float32)
        nd = int(cfg.get("round_ndigits", 2))

        return np.round(st, nd).astype(np.float32) / time_multiplier


class ServiceTimePolicies:
    """
    Factory for service time policy selection.

    Selection priority:
        1) env['test_servicetime_type'] (deterministic testing)
        2) sample from env['servicetime_type_distribution'] (e.g., {'Random': 1.0})
        3) raise if neither is provided/valid
    """
    REGISTRY = {
        "random": Random_ServiceTimePolicy,
        "cargoweight": CargoWeightBased_ServiceTimePolicy
    }

    @classmethod
    def _sample_choice(cls, env: Dict, rng: np.random.Generator = None) -> str:
        """
        Select the time window type with the following priority:
        1) If 'test_timewindow_type' is present in env, use it (for deterministic testing).
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
    def from_env(cls, env: Dict, rng: np.random.Generator = None) -> ServiceTimePolicies:
        choice = cls._sample_choice(env, rng=rng)
        return cls.REGISTRY[choice]()
