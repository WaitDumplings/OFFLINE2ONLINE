from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
from gym.vector.utils import concatenate, create_empty_array, iterate
from gym.vector.vector_env import VectorEnv

__all__ = ["SyncVectorEnv"]


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments."""

    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True, dataset=None):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata
        self.n_traj = self._unwrap_env(self.envs[0]).n_traj
        self.dataset = dataset

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space

        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()

        self.observations = create_empty_array(
            self.single_observation_space,
            n=self.num_envs,
            fn=np.zeros,
        )

        self._rewards = np.zeros((self.num_envs, self.n_traj), dtype=np.float64)
        self._dones = np.zeros((self.num_envs, self.n_traj), dtype=np.bool_)
        self._actions = None

    def _unwrap_env(self, env):
        """
        Recursively unwrap Gym wrappers to reach the base environment.

        This avoids hard-coding wrapper depth such as env.env.env.env.
        """
        while hasattr(env, "env"):
            env = env.env
        return env

    def _normalize_attr_values(self, values, *, allow_broadcast=True):
        """
        Normalize attribute values for vectorized env assignment.

        Parameters
        ----------
        values:
            Scalar, list, tuple, or np.ndarray.

        allow_broadcast:
            If True, scalar values are broadcast to all environments.
            If False, values must be list/tuple/ndarray with length num_envs.

        Returns
        -------
        list
            A list with length self.num_envs.
        """
        if isinstance(values, np.ndarray):
            if values.ndim == 0:
                if not allow_broadcast:
                    raise TypeError(
                        "Scalar ndarray is not allowed when allow_broadcast=False."
                    )
                return [values.item() for _ in range(self.num_envs)]

            if len(values) != self.num_envs:
                raise ValueError(
                    "Numpy array values must have length equal to num_envs. "
                    f"Got shape {values.shape} for {self.num_envs} environments."
                )

            return values.tolist()

        if isinstance(values, (list, tuple)):
            if len(values) != self.num_envs:
                raise ValueError(
                    "Values must have length equal to the number of environments. "
                    f"Got {len(values)} values for {self.num_envs} environments."
                )
            return list(values)

        if allow_broadcast:
            return [values for _ in range(self.num_envs)]

        raise TypeError(
            "`values` must be a list, tuple, or np.ndarray with length equal "
            f"to num_envs. Got type: {type(values)}."
        )

    def seed(self, seed=None):
        super().seed(seed=seed)

        if seed is None:
            seed = [None for _ in range(self.num_envs)]

        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]

        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            base_env = self._unwrap_env(env)
            base_env.seed(single_seed)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]

        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]

        assert len(seed) == self.num_envs

        self._dones[:] = False
        observations = []
        data_list = []

        for env, single_seed in zip(self.envs, seed):
            kwargs = {}

            if single_seed is not None:
                kwargs["seed"] = single_seed

            if options is not None:
                kwargs["options"] = options

            if return_info:
                kwargs["return_info"] = return_info

            if not return_info:
                observation = env.reset(**kwargs)
                observations.append(observation)
            else:
                observation, data = env.reset(**kwargs)
                observations.append(observation)
                data_list.append(data)

        self.observations = concatenate(
            self.single_observation_space,
            observations,
            self.observations,
        )

        if not return_info:
            return deepcopy(self.observations) if self.copy else self.observations

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            data_list,
        )

    def step_async(self, actions):
        self._actions = iterate(self.action_space, actions)

    def step_wait(self):
        observations, infos = [], []

        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            observations.append(observation)
            infos.append(info)

        self.observations = concatenate(
            self.single_observation_space,
            observations,
            self.observations,
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def call(self, name, *args, **kwargs):
        results = []

        for env in self.envs:
            base_env = self._unwrap_env(env)
            attr = getattr(base_env, name)

            if callable(attr):
                results.append(attr(*args, **kwargs))
            else:
                results.append(attr)

        return tuple(results)

    def set_attr(self, name, values):
        """
        Set an attribute on each underlying base environment.

        Behavior:
        - scalar value: broadcast to all envs
        - list/tuple/np.ndarray with length num_envs: per-env values
        """
        values = self._normalize_attr_values(values, allow_broadcast=True)

        for env, value in zip(self.envs, values):
            base_env = self._unwrap_env(env)
            setattr(base_env, name, value)

    def update_attr(self, name, values):
        """
        Update an attribute on each underlying base environment.

        Behavior:
        - scalar value: broadcast to all envs
        - list/tuple/np.ndarray with length num_envs: per-env values

        This is kept for compatibility with existing training code.
        """
        values = self._normalize_attr_values(values, allow_broadcast=True)

        for env, value in zip(self.envs, values):
            base_env = self._unwrap_env(env)
            setattr(base_env, name, value)

    def update_attr_each(self, name, values):
        """
        Explicit per-env attribute update.

        Use this when each environment needs a different value, e.g.,
        teacher_obj_by_env with shape (num_envs,).
        """
        values = self._normalize_attr_values(values, allow_broadcast=False)

        for env, value in zip(self.envs, values):
            base_env = self._unwrap_env(env)
            setattr(base_env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        for env in self.envs:
            env.close()

    def _check_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, "
                    "the action spaces from all environments must be equal."
                )

        return True