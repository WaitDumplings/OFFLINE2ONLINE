"""
Clean model contract for competence-guided PPO.

The network architecture is intentionally the same as the current route-event
agent so existing backbone checkpoints can be used as warm starts. The clean
trainer imports from this file instead of the legacy training module.
"""

from evrptw_gen.benchmarks.DRL_Solver.models.graph_attention_model_wrapper import (
    Actor,
    Agent as CompetenceAgent,
    Backbone,
    Critic,
    StateWrapper,
)

__all__ = [
    "Actor",
    "Backbone",
    "CompetenceAgent",
    "Critic",
    "StateWrapper",
]
