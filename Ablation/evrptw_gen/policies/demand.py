# evrptw_gen/policies/demand.py
from __future__ import annotations
from typing import Dict, Protocol
from collections import Counter
import numpy as np

testset_name = "demands_policy"
config_list_name = "demands_policy_list"
config_type_name = "demands_policy_config"

class DemandPolicy(Protocol):
    def build(self, env: Dict, num_customers: int, rng: np.random.Generator) -> np.ndarray:
        """Return demand array of shape (num_customers,)."""
        ...

class Uniform_DemandPolicy:
    NAME = "uniform"
    def build(self, env: Dict, num_customers: int, rng: np.random.Generator) -> np.ndarray:
        cfg = env.get("demand_uniform_config", {})
        dmin = float(cfg.get("min", 0.001))
        dmax = float(cfg.get("max", 0.1))

        demands = np.full(num_customers, (dmin + dmax) / 2)
        print("Uniform")
        return demands.astype(np.float32)

class _20_80_DemandPolicy:
    NAME = "20_80"

    def build(
        self,
        env: Dict,
        num_customers: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        cfg = env.get("demands_policy_config")[self.NAME]

        small_weight_range = cfg.get("small_weight_range")
        large_weight_range = cfg.get("large_weight_range")


        # 80%: low demand
        demands = rng.uniform(small_weight_range[0], small_weight_range[1], size=num_customers)

        # 20%: high demand
        num_high = int(0.2 * num_customers)
        high_idx = rng.choice(num_customers, size=num_high, replace=False)
        high_values = rng.uniform(large_weight_range[0], large_weight_range[1], size=num_high)

        demands[high_idx] = high_values
        return np.round(demands.astype(np.float32), 3)

class RandomDemandPolicy:
    """
    Uniformly sample customer demands in [min, max].
    If unspecified, defaults to [0.1, env['loading_capacity']].
    """
    NAME = "random"

    def build(self, env: Dict, num_customers: int, rng: np.random.Generator) -> np.ndarray:
        cfg = env.get("demands_policy_config")[self.NAME]
        weight_range = cfg.get("weight_range")
        dmin, dmax = weight_range

        if dmax <= dmin:
            dmax = dmin + 0.01
        demands = rng.uniform(dmin, dmax, size=num_customers)
        return np.round(demands, 3).astype(np.float32)


class DemandPolicies:
    REGISTRY = {"random": RandomDemandPolicy,
                "20_80": _20_80_DemandPolicy,
                "uniform": Uniform_DemandPolicy
                }

    @classmethod
    def from_env(cls, env: Dict, rng: np.random.Generator = None) -> DemandPolicy:
        """
        Currently only supports 'Random'.
        Future: can extend with 20_80 or clustered demand distributions.
        """
        choice = env.get("demand_policy", None)
        choice_list = env.get("demand_policy_list", None)
        if Counter(choice_list) != Counter(cls.REGISTRY.keys()):
            return ValueError("Environment demand_policy_list does not match registry.")

        if not choice:
            choice_number = len(cls.REGISTRY)
            if rng is None:
                rng = np.random.default_rng()
            choice = list(cls.REGISTRY.keys())[int(rng.integers(choice_number))]

        if choice not in cls.REGISTRY:
            raise ValueError(f"Unknown demand policy: {choice}")
        env["demand_policy"] = choice
        return cls.REGISTRY[choice]()


    @classmethod
    def _sample_choice(cls, env: Dict, rng: np.random.Generator = None) -> str:
        """
        Select the demands type with the following priority:
        1) If 'test_demands_type' is present in env, use it (for deterministic testing).
        2) Otherwise, sample from 'demands_type_distribution' according to probabilities.
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
    def from_env(cls, env: Dict, rng: np.random.Generator = None) -> DemandPolicy:
        choice = cls._sample_choice(env, rng=rng)
        return cls.REGISTRY[choice]()
