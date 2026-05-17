# evrptw_gen/policies/demand.py
from __future__ import annotations
from typing import Dict, Protocol, Tuple
from collections import Counter
import numpy as np

testset_name = "rc_customers_assign_policy"
config_list_name = "rc_customers_assign_policy_list"
config_type_name = "rc_customers_assign_policy_config"

class RCSplitPolicy(Protocol):

    def build(self, env: Dict, rng: np.random.Generator) -> Tuple[int, int]:
        """Return demand array of shape (num_customers,)."""
        ...


class FixedRCSplit:
    """
 
    """
    NAME = "fixed"

    def build(self, env: Dict, rng: np.random.Generator) -> Tuple[int, int]:
        cfg = env[config_type_name][self.NAME]
        ratios = cfg.get('fixed_rc_customer_ratio', None)
        random_ratio, cluster_ratio = ratios

        if random_ratio + cluster_ratio != 1:
            raise ValueError(f"random type ratio plus cluster type ratio should be 1, but we get {random_ratio + cluster_ratio} instead.")
        customer_num = env['num_customers']
        random_customer_number = int(customer_num * random_ratio)
        cluster_customer_number = customer_num - random_customer_number
        if (cluster_customer_number + random_customer_number) != customer_num:
            breakpoint()
            raise ValueError("Inconsistent customer number after rc customer assignment.")
        return random_customer_number, cluster_customer_number

class RCPolicies:
    REGISTRY = {"fixed": FixedRCSplit}

    
    @classmethod
    def _sample_choice(cls, env: Dict, rng: np.random.Generator = None) -> str:
        """
        Select the rc customers type with the following priority:
        1) If 'rc_customers_type' is present in env, use it (for deterministic testing).
        2) Otherwise, sample from 'rc_customers_type_distribution' according to probabilities.
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
    def from_env(cls, env: Dict, rng: np.random.Generator = None) -> RCSplitPolicy:
        choice = cls._sample_choice(env, rng=rng)
        return cls.REGISTRY[choice]()
