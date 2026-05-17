# evrptw_gen/policies/demand.py
from __future__ import annotations
from typing import Dict, Protocol
from collections import Counter
import numpy as np

testset_name = "cluster_number_policy"
config_list_name = "cluster_number_policy_list"
config_type_name = "cluster_number_policy_config"

class ClusterNumberPolicy(Protocol):

    def build(self, env: Dict, rng: np.random.Generator) -> np.ndarray:
        """Return demand array of shape (num_customers,)."""
        ...

class RandomClusterNumberPolicy:
    """
 
    """
    NAME = "random"

    def build(self, env: Dict, rng: np.random.Generator) -> np.ndarray:
        cfg = env[config_type_name][self.NAME]
        lo = int(cfg.get("min_cluster_num", 1))
        hi = int(cfg.get("max_cluster_num", env['num_customers'] // 2))
        cluster_number = rng.integers(lo, hi + 1)
        return cluster_number

class FixedClusterNumberPolicy:
    """
 
    """
    NAME = "fixed"

    def build(self, env: Dict, rng: np.random.Generator) -> np.ndarray:
        cfg = env[config_type_name][self.NAME]
        cluster_number = int(cfg.get("fixed_cluster_num", 5))
        return cluster_number

class ClusterNumberPolicies:
    REGISTRY = {"fixed": FixedClusterNumberPolicy,
                "random": RandomClusterNumberPolicy}


    @classmethod
    def _sample_choice(cls, env: Dict, rng: np.random.Generator = None) -> str:
        """
        Select the cluster number type with the following priority:
        1) If 'test_cluster_number_type' is present in env, use it (for deterministic testing).
        2) Otherwise, sample from 'cluster_number_type_distribution' according to probabilities.
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
    def from_env(cls, env: Dict, rng: np.random.Generator = None) -> ClusterNumberPolicy:
        choice = cls._sample_choice(env, rng=rng)
        return cls.REGISTRY[choice]()
