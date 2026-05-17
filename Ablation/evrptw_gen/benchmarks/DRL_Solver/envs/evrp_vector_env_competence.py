import numpy as np

from evrptw_gen.benchmarks.DRL_Solver.envs.evrp_vector_env import EVRPTWVectorEnv


class CompetenceEVRPTWVectorEnv(EVRPTWVectorEnv):
    """
    Clean training env for competence-guided offline-to-online PPO.

    This class deliberately disables legacy teacher rewards. ALNS is used only
    as fixed-instance data, optional prefix reset, and objective reference.
    """

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("reward_mode", "vanilla")
        kwargs["use_teacher_reward"] = False
        kwargs["teacher_reward_mode"] = "none"
        super().__init__(*args, **kwargs)
        self.prefix_len = 0
        self.prefix_objective = np.zeros(self.n_traj, dtype=np.float32)
        self.teacher_suffix_obj = np.full(self.n_traj, np.nan, dtype=np.float32)

    def set_teacher_reference(self, *args, **kwargs):
        """
        Keep teacher metadata loadable for diagnostics, but never enable reward.
        """
        kwargs = dict(kwargs)
        kwargs["enabled"] = False
        super().set_teacher_reference(*args, **kwargs)
        self.use_teacher_reward = False

    def reset(self):
        obs = super().reset()
        self.use_teacher_reward = False
        self.prefix_len = 0
        self.prefix_objective = np.zeros(self.n_traj, dtype=np.float32)
        self.teacher_suffix_obj = self._current_teacher_suffix_obj()
        return obs

    def reset_with_teacher_prefix(self, action_sequence=None, prefix_len=0):
        """
        Reset to the same fixed/generated instance and replay a shared ALNS
        prefix before PPO starts. The replay is not part of the PPO rollout.
        """
        obs = self.reset()

        seq = [] if action_sequence is None else [int(x) for x in action_sequence]
        max_prefix = min(max(int(prefix_len), 0), len(seq))

        actual_prefix = 0
        for action in seq[:max_prefix]:
            if action < 0 or action >= self.n_nodes:
                break
            if not np.all(self.mask[:, action]):
                break

            action_vec = np.full(self.n_traj, action, dtype=np.int64)
            obs, _, done, _ = self.step(action_vec)
            actual_prefix += 1

            if np.any(done):
                break

        self.use_teacher_reward = False
        self.prefix_len = int(actual_prefix)
        self.prefix_objective = self.objective.astype(np.float32).copy()
        self.teacher_suffix_obj = self._current_teacher_suffix_obj()
        self.state = self._update_state()
        return self.state

    def _current_teacher_suffix_obj(self):
        teacher_obj = self.teacher_obj
        if teacher_obj is None:
            teacher = None
            if isinstance(self.current_instance, dict):
                teacher = self.current_instance.get("teacher", None)
            if isinstance(teacher, dict):
                teacher_obj = teacher.get("obj", None)

        try:
            teacher_value = float(np.asarray(teacher_obj, dtype=np.float32).reshape(-1)[0])
        except Exception:
            teacher_value = np.nan

        suffix = teacher_value - self.prefix_objective
        return np.asarray(suffix, dtype=np.float32).reshape(self.n_traj)
