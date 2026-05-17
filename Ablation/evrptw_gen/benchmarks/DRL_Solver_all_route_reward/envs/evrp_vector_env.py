import gym
import numpy as np
import random
from copy import deepcopy
from gym import spaces

from evrptw_gen.generator import InstanceGenerator
from numba import njit


@njit(cache=True)
def _ffp_customer_numba(
    base_cus_mask,
    battery,
    current_time,
    time_to_customer,
    energy_to_customer,
    tw_open_cus,
    service_time_cus,
    time_cus_to_rs,
    energy_cus_to_rs,
    charging_beta,
    rs_time_to_depot,
    battery_capacity,
    instance_max_time,
):
    """
    Future-feasibility pruning for customer actions.

    For candidate customer c, check whether:
        current -> c -> some RS/depot -> depot
    is feasible under full-recharge semantics.
    """
    n_traj, n_cus = base_cus_mask.shape
    n_rs1 = rs_time_to_depot.shape[0]

    out = np.zeros((n_traj, n_cus), dtype=np.bool_)

    for t in range(n_traj):
        bt = battery[t]
        ct = current_time[t]

        for c in range(n_cus):
            if not base_cus_mask[t, c]:
                continue

            battery_at_customer = bt + energy_to_customer[t, c]
            arrival_time = ct + time_to_customer[t, c]

            if arrival_time > tw_open_cus[c]:
                start_service_time = arrival_time
            else:
                start_service_time = tw_open_cus[c]

            time_after_service = start_service_time + service_time_cus[c]

            feasible = False

            for r in range(n_rs1):
                battery_at_rs = battery_at_customer + energy_cus_to_rs[c, r]

                if battery_at_rs > battery_capacity:
                    continue

                finish_time = time_after_service + time_cus_to_rs[c, r]

                # r == 0 means depot. No charging time at depot in this look-ahead.
                if r != 0:
                    finish_time += battery_at_rs * charging_beta

                finish_time += rs_time_to_depot[r]

                if finish_time <= instance_max_time:
                    feasible = True
                    break

            out[t, c] = feasible

    return out


@njit(cache=True)
def _update_mask_numba(
    visited,
    last,
    load,
    battery,
    current_time,
    served_cus,
    route_served_cus_count,
    demands,
    travel_time,
    edge_energy,
    tw_close_all,
    tw_open_cus,
    service_time_cus,
    time_cus_to_rs,
    energy_cus_to_rs,
    rs_cols,
    rs_time_to_depot,
    charging_beta,
    loading_capacity,
    battery_capacity,
    instance_max_time,
    cus_num,
    cus_start,
    rs_start,
):
    n_traj, n_nodes = visited.shape
    n_cus = cus_num
    n_rs1 = rs_cols.shape[0]
    action_mask = np.empty((n_traj, n_nodes), dtype=np.bool_)

    for t in range(n_traj):
        last_t = last[t]
        all_visited = served_cus[t] == cus_num

        for j in range(n_nodes):
            allowed = not visited[t, j]

            if j == 0 or j >= rs_start:
                allowed = True

            if j == last_t:
                allowed = False

            if allowed:
                if load[t] + demands[j] > loading_capacity:
                    allowed = False
                elif battery[t] + edge_energy[last_t, j] > battery_capacity:
                    allowed = False
                elif current_time[t] + travel_time[last_t, j] > tw_close_all[j]:
                    allowed = False

            if allowed and j >= cus_start and j < rs_start:
                c = j - cus_start

                battery_at_customer = battery[t] + edge_energy[last_t, j]
                arrival_time = current_time[t] + travel_time[last_t, j]

                if arrival_time > tw_open_cus[c]:
                    start_service_time = arrival_time
                else:
                    start_service_time = tw_open_cus[c]

                time_after_service = start_service_time + service_time_cus[c]
                feasible = False

                for r in range(n_rs1):
                    battery_at_rs = battery_at_customer + energy_cus_to_rs[c, r]
                    if battery_at_rs > battery_capacity:
                        continue

                    finish_time = time_after_service + time_cus_to_rs[c, r]
                    if r != 0:
                        finish_time += battery_at_rs * charging_beta
                    finish_time += rs_time_to_depot[r]

                    if finish_time <= instance_max_time:
                        feasible = True
                        break

                allowed = feasible

            action_mask[t, j] = allowed

        for r in range(n_rs1):
            j = rs_cols[r]
            if action_mask[t, j]:
                battery_at_rs = battery[t] + edge_energy[last_t, j]
                time_at_rs = current_time[t] + travel_time[last_t, j]
                finish_time = time_at_rs + rs_time_to_depot[r]

                if r != 0:
                    finish_time += battery_at_rs * charging_beta

                if battery_at_rs > battery_capacity or finish_time > instance_max_time:
                    action_mask[t, j] = False

        if all_visited:
            action_mask[t, 0] = True

            if last_t == 0:
                for j in range(1, n_nodes):
                    action_mask[t, j] = False
        elif route_served_cus_count[t] == 0 and action_mask[t, 0]:
            has_non_depot_choice = False
            for j in range(1, n_nodes):
                if action_mask[t, j]:
                    has_non_depot_choice = True
                    break

            if has_non_depot_choice:
                action_mask[t, 0] = False

    return action_mask


def gen_dist_matrix(nodes1: np.ndarray, nodes2: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(nodes1, dtype=np.float32)
    y = np.ascontiguousarray(nodes2, dtype=np.float32)

    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    d2 = np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)

    return np.sqrt(d2, out=d2).astype(np.float32, copy=False)


class EVRPTWVectorEnv(gym.Env):
    """
    Vectorized EVRPTW environment with normalized internal dynamics.

    Key design:
        1. env_mode="train" can use either generated instances or fixed instances.
        2. env_mode="eval" always disables teacher reward.
        3. A fixed instance may carry teacher information:
               instance["teacher"] = {
                   "obj": float,
                   "source": "alns" / "self_teacher" / ...,
                   "stage": "offline" / "online" / ...,
               }
        4. Self-teacher flow:
               reset() -> teacher rollout -> set_teacher_reference(...)
               -> reset_same_instance(keep_teacher=True) -> student rollout.
        5. Teacher reward modes:
               "scaled_success":
                    teacher controls success_bonus coefficient.
               "additive":
                    old behavior, teacher gives extra additive reward/penalty.
               "none":
                    teacher is ignored.
    """

    metadata = {"render.modes": []}

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # =====================================================
        # Basic mode/config
        # =====================================================
        self.env_mode = kwargs.get("env_mode", "train")
        self.config_path = kwargs.get("config_path", None)
        self.config = kwargs.get("config", None)
        self.initial_seed = kwargs.get("seed", None)

        self.n_traj = int(kwargs.get("n_traj", 100))
        self.Q_ratio = float(kwargs.get("Q_ratio", 1.0))
        self.r_ratio = float(kwargs.get("r_ratio", 1.0))

        self.terminate = False

        # =====================================================
        # Reward switches
        # =====================================================
        self.reward_mode = str(kwargs.get("reward_mode", "legacy")).lower()
        if self.reward_mode not in {"legacy", "vanilla"}:
            raise ValueError(
                f"Unknown reward_mode={self.reward_mode!r}; expected legacy or vanilla."
            )
        self.phi_reward = bool(kwargs.get("phi_reward", True))
        self.lag_reward = bool(kwargs.get("lag_reward", True))

        self.gamma = float(kwargs.get("gamma", 0.99))
        self.alpha = float(kwargs.get("alpha", 2.0))
        self.beta = float(kwargs.get("beta", 0.5))
        self.lambda_fail = float(kwargs.get("lambda_fail", 5.0))
        self.success_bonus = float(kwargs.get("success_bonus", 10.0))
        self.pbrs_mode = str(kwargs.get("pbrs_mode", "served")).lower()
        if self.pbrs_mode == "serve":
            self.pbrs_mode = "served"
        self.pbrs_coef = float(kwargs.get("pbrs_coef", 1.0))
        self.pbrs_feasible_coef = float(kwargs.get("pbrs_feasible_coef", 0.0))
        self.rs_progress_eps = float(kwargs.get("rs_progress_eps", 1e-6))
        self.use_direct_progress_pbrs = bool(
            kwargs.get("use_direct_progress_pbrs", False)
        )
        self.progress_pbrs_coef = float(
            kwargs.get("progress_pbrs_coef", self.pbrs_coef)
        )
        self.progress_pbrs_beta = float(
            kwargs.get("progress_pbrs_beta", self.beta)
        )
        self.use_repair_fail_reward = bool(
            kwargs.get("use_repair_fail_reward", False)
        )
        self.repair_fail_coef = float(kwargs.get("repair_fail_coef", 1.0))
        self.repair_progress_coef = float(kwargs.get("repair_progress_coef", 1.0))
        self.repair_success_bonus = float(kwargs.get("repair_success_bonus", 1.0))

        self.empty_route_penalty = float(kwargs.get("empty_route_penalty", 0.1))
        self.empty_route_length_coef = float(kwargs.get("empty_route_length_coef", 0.5))
        self.empty_route_count_penalty = float(
            kwargs.get("empty_route_count_penalty", 0.0)
        )
        self.empty_route_count_cap = max(
            1,
            int(kwargs.get("empty_route_count_cap", 10)),
        )
        self.no_progress_rs_penalty = float(kwargs.get("no_progress_rs_penalty", 0.0))
        self.empty_no_progress_route_penalty = float(
            kwargs.get("empty_no_progress_route_penalty", 0.0)
        )
        self.rs_streak_penalty = float(kwargs.get("rs_streak_penalty", 0.0))
        self.rs_streak_free = max(0, int(kwargs.get("rs_streak_free", 2)))

        # =====================================================
        # Teacher/reference
        # =====================================================
        self.use_teacher_reward = bool(kwargs.get("use_teacher_reward", False))
        self.teacher_obj = kwargs.get("teacher_obj", None)
        self.teacher_source = kwargs.get("teacher_source", None)
        self.teacher_stage = kwargs.get("teacher_stage", None)
        self.teacher_raw_obj = kwargs.get("teacher_raw_obj", None)
        self.teacher_reward_gate = float(kwargs.get("teacher_reward_gate", 1.0))
        self.use_teacher_dense_reward = bool(kwargs.get("use_teacher_dense_reward", False))
        self.teacher_dense_coef = float(kwargs.get("teacher_dense_coef", 0.0))
        self.teacher_dense_mode = str(kwargs.get("teacher_dense_mode", "exact")).lower()
        self.teacher_dense_lookahead = max(
            1,
            int(kwargs.get("teacher_dense_lookahead", 1)),
        )
        self.teacher_dense_rank_tau = max(
            1e-6,
            float(kwargs.get("teacher_dense_rank_tau", 2.0)),
        )
        self.teacher_action_sequence = kwargs.get("teacher_action_sequence", None)
        self.use_teacher_route_potential = bool(
            kwargs.get("use_teacher_route_potential", False)
        )
        self.teacher_route_potential_coef = float(
            kwargs.get("teacher_route_potential_coef", 0.5)
        )
        self.teacher_route_potential_clip = float(
            kwargs.get("teacher_route_potential_clip", 0.05)
        )
        self.teacher_gap_worse_only = bool(
            kwargs.get("teacher_gap_worse_only", False)
        )
        self.teacher_edge_from = np.empty(0, dtype=np.int32)
        self.teacher_edge_to = np.empty(0, dtype=np.int32)
        self.teacher_edge_matched = np.zeros((self.n_traj, 0), dtype=np.bool_)
        self.teacher_route_rank_by_node = np.empty(0, dtype=np.float32)
        self.teacher_route_rank_denom = 0.0
        self.teacher_route_rank_progress = np.zeros(self.n_traj, dtype=np.float32)

        # =====================================================
        # Teacher-calibrated success bonus
        # =====================================================
        # Main switch:
        #   "scaled_success": teacher controls success_bonus coefficient
        #   "additive": old additive teacher reward
        #   "none": ignore teacher
        self.teacher_reward_mode = str(
            kwargs.get("teacher_reward_mode", "scaled_success")
        )
        if self.teacher_reward_mode == "success_coef":
            self.teacher_reward_mode = "scaled_success"

        # When no teacher is available, successful routes receive:
        #   success_bonus * plain_success_bonus_coef
        self.plain_success_bonus_coef = float(
            kwargs.get("plain_success_bonus_coef", 1.0)
        )

        # Coefficient type for teacher-calibrated success bonus:
        #   "saturation", "convex", "linear"
        self.teacher_success_coef_type = str(
            kwargs.get("teacher_success_coef_type", "saturation")
        )
        if self.teacher_success_coef_type == "nonlinear":
            self.teacher_success_coef_type = "saturation"

        self.teacher_reward_margin = float(
            kwargs.get("teacher_reward_margin", 0.01)
        )

        # Old additive teacher reward parameters.
        # Kept for backward compatibility.
        self.teacher_penalty_ratio = float(kwargs.get("teacher_penalty_ratio", 0.50))
        self.teacher_bonus_ratio = float(kwargs.get("teacher_bonus_ratio", 0.20))
        self.teacher_failure_ratio = float(kwargs.get("teacher_failure_ratio", 0.50))
        self.teacher_reward_temp = float(kwargs.get("teacher_reward_temp", 0.20))
        self.teacher_closure_failure_min_ratio = float(
            kwargs.get("teacher_closure_failure_min_ratio", 0.10)
        )

        # New scaled-success parameters.
        self.teacher_success_worse_floor = float(
            kwargs.get("teacher_success_worse_floor", 0.20)
        )
        self.teacher_success_equal_coef = float(
            kwargs.get("teacher_success_equal_coef", 1.00)
        )
        self.teacher_success_max_coef = float(
            kwargs.get("teacher_success_max_coef", 2.00)
        )

        # Saturation:
        # coef = equal + (max - equal) * (1 - exp(-improve / scale))
        self.teacher_success_improve_scale = float(
            kwargs.get("teacher_success_improve_scale", 0.10)
        )

        # Convex / linear:
        # coef = equal + (max - equal) * (improve / target) ** power
        self.teacher_success_target_improve = float(
            kwargs.get("teacher_success_target_improve", 0.20)
        )
        self.teacher_success_power = float(
            kwargs.get("teacher_success_power", 3.0)
        )

        # Worse-than-teacher decay.
        self.teacher_success_worse_temp = float(
            kwargs.get("teacher_success_worse_temp", 0.20)
        )

        # Optional teacher-aware failure penalty.
        # Recommended default: False.
        self.teacher_scaled_failure = bool(
            kwargs.get("teacher_scaled_failure", False)
        )

        if self.reward_mode == "vanilla":
            self.phi_reward = True
            self.lag_reward = True
            if self.use_direct_progress_pbrs:
                self.pbrs_mode = "direct_progress"
            else:
                self.pbrs_mode = "served"
            self.pbrs_feasible_coef = 0.0
            self.empty_route_penalty = 0.0
            self.empty_route_length_coef = 0.0
            self.empty_route_count_penalty = 0.0
            self.no_progress_rs_penalty = 0.0
            self.empty_no_progress_route_penalty = 0.0
            self.rs_streak_penalty = 0.0
            self.use_teacher_reward = False
            self.use_teacher_dense_reward = False
            self.use_teacher_route_potential = False
            self.teacher_reward_mode = "none"

        # =====================================================
        # Instance source
        # =====================================================
        self.cus_num = kwargs.get("num_customers", None)
        self.rs_num = kwargs.get("num_charging_stations", None)
        self.perturb_dict = kwargs.get("perturb_dict", {})

        # In train mode, eval_data means fixed training instance.
        # In eval mode, eval_data means evaluation instance.
        self.fixed_data = kwargs.get("eval_data", None)

        # Whether reset() should keep using current/fixed instance.
        # Useful for self-teacher: teacher rollout and student rollout use same instance.
        self.keep_instance_on_reset = bool(kwargs.get("keep_instance_on_reset", False))
        self.keep_teacher_on_reset = bool(kwargs.get("keep_teacher_on_reset", False))

        if self.env_mode == "train":
            if self.fixed_data is None:
                if self.cus_num is None or self.rs_num is None:
                    raise ValueError(
                        "In train mode with random generation, "
                        "num_customers and num_charging_stations must be specified."
                    )

        elif self.env_mode == "eval":
            if self.fixed_data is None:
                raise ValueError("In eval mode, eval_data must be provided.")

            # Hard guarantee: teacher reward must not affect evaluation.
            self.use_teacher_reward = False

        else:
            raise ValueError(f"Unknown env_mode: {self.env_mode}")

        save_path = kwargs.get("save_path", None)
        num_instances = kwargs.get("num_instances", 1)
        plot_instances = kwargs.get("plot_instances", False)

        self.dataset = InstanceGenerator(
            self.config_path,
            save_path=save_path,
            num_instances=num_instances,
            plot_instances=plot_instances,
            config=self.config,
            seed=self.initial_seed,
            kwargs=kwargs,
        )

        self.current_instance = None
        self.context = None
        self.state = None
        self.info = {}

        self.reset()

    # =========================================================
    # Public helpers
    # =========================================================
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.dataset._update_seeds(seed)
        if (
            seed is not None
            and self.initial_seed is not None
            and int(seed) == int(self.initial_seed)
            and self.current_instance is not None
        ):
            # __init__ already generated the first instance with this seed.
            # Keep subsequent resets on the next derived seed instead of
            # replaying the same instance.
            self.dataset._instance_counter = 1

    def set_teacher_reference(
        self,
        teacher_obj=None,
        source=None,
        stage=None,
        enabled=True,
        raw_obj=None,
        gate=None,
        action_sequence=None,
    ):
        """
        Set teacher/reference objective for the current instance.

        teacher_obj:
            None, scalar, or array-like with shape (n_traj,).

        Important:
            teacher_obj must be in the same objective scale as self.objective.
            self.objective uses normalized-coordinate distance.
        """
        self.teacher_obj = teacher_obj
        self.teacher_source = source
        self.teacher_stage = stage
        self.teacher_raw_obj = raw_obj
        if gate is not None:
            self.teacher_reward_gate = float(np.clip(gate, 0.0, 1.0))
        if action_sequence is not None:
            self.teacher_action_sequence = list(action_sequence)
            self._prepare_teacher_route_reference()

        if self.env_mode != "train" or self.reward_mode == "vanilla":
            self.use_teacher_reward = False
        else:
            self.use_teacher_reward = bool(enabled and teacher_obj is not None)

    def clear_teacher_reference(self):
        self.teacher_obj = None
        self.teacher_source = None
        self.teacher_stage = None
        self.teacher_raw_obj = None
        self.teacher_reward_gate = 1.0
        self.teacher_action_sequence = None
        self._prepare_teacher_route_reference()
        self.use_teacher_reward = False

    def freeze_current_instance(self):
        """
        Freeze current generated/fixed instance so future reset() can reuse it.
        """
        if self.current_instance is None:
            raise RuntimeError("Cannot freeze instance before reset/load.")

        self.fixed_data = deepcopy(self.current_instance)
        self.keep_instance_on_reset = True

    def load_fixed_instance(
        self,
        instance,
        teacher_obj=None,
        teacher_source=None,
        teacher_stage=None,
        teacher_raw_obj=None,
        enable_teacher=True,
    ):
        """
        Load a fixed instance into this env.

        Used for ALNS-buffer batches or externally sampled fixed batches.
        """
        if instance is None:
            raise ValueError("instance cannot be None.")

        self.fixed_data = deepcopy(instance)
        self.keep_instance_on_reset = True

        if teacher_obj is not None:
            self.set_teacher_reference(
                teacher_obj=teacher_obj,
                source=teacher_source,
                stage=teacher_stage,
                enabled=enable_teacher,
                raw_obj=teacher_raw_obj,
            )
            self.keep_teacher_on_reset = True
        else:
            self._try_load_teacher_from_context(self.fixed_data)

        return self.reset()

    def reset_same_instance(self, keep_teacher=True):
        """
        Reset runtime state while keeping the same current instance.

        Key API for self-teacher:
            1. reset() creates instance.
            2. teacher rollout gets teacher_obj.
            3. set_teacher_reference(...).
            4. reset_same_instance(keep_teacher=True).
            5. student rollout uses teacher reward.
        """
        if self.current_instance is None:
            raise RuntimeError("Cannot reset same instance before an instance exists.")

        teacher_snapshot = self._snapshot_teacher_reference()

        self.fixed_data = deepcopy(self.current_instance)
        self.keep_instance_on_reset = True

        obs = self.reset()

        if keep_teacher:
            self._restore_teacher_reference(teacher_snapshot)
        else:
            self.clear_teacher_reference()

        self.state = self._update_state()
        return self.state

    def step(self, action):
        return self._STEP(action)

    # =========================================================
    # Teacher reference helpers
    # =========================================================
    def _snapshot_teacher_reference(self):
        return {
            "teacher_obj": deepcopy(self.teacher_obj),
            "teacher_source": self.teacher_source,
            "teacher_stage": self.teacher_stage,
            "teacher_raw_obj": self.teacher_raw_obj,
            "teacher_reward_gate": self.teacher_reward_gate,
            "teacher_action_sequence": deepcopy(self.teacher_action_sequence),
            "use_teacher_reward": bool(self.use_teacher_reward),
        }

    def _restore_teacher_reference(self, snapshot):
        self.teacher_obj = snapshot["teacher_obj"]
        self.teacher_source = snapshot["teacher_source"]
        self.teacher_stage = snapshot["teacher_stage"]
        self.teacher_raw_obj = snapshot["teacher_raw_obj"]
        self.teacher_reward_gate = snapshot.get("teacher_reward_gate", 1.0)
        self.teacher_action_sequence = deepcopy(snapshot.get("teacher_action_sequence", None))

        if self.env_mode != "train" or self.reward_mode == "vanilla":
            self.use_teacher_reward = False
        else:
            self.use_teacher_reward = bool(
                snapshot["use_teacher_reward"] and self.teacher_obj is not None
            )

    def _try_load_teacher_from_context(self, context):
        """
        Load teacher reference from the instance itself.

        Supported formats:
            context["teacher"] = {
                "obj": ...,
                "source": ...,
                "stage": ...,
                "raw_obj": ...
            }

            or direct keys:
                context["teacher_obj"]
                context["teacher_source"]
                context["teacher_stage"]
        """
        if self.env_mode != "train" or self.reward_mode == "vanilla":
            self.use_teacher_reward = False
            return

        if not isinstance(context, dict):
            return

        teacher = context.get("teacher", None)

        teacher_obj = None
        teacher_source = None
        teacher_stage = None
        teacher_raw_obj = None
        teacher_gate = None
        teacher_action_sequence = None

        if isinstance(teacher, dict):
            for k in ["obj", "objective", "teacher_obj", "best_obj", "global_value"]:
                if k in teacher:
                    teacher_obj = teacher.get(k)
                    break

            teacher_source = teacher.get("source", teacher.get("teacher_source", None))
            teacher_stage = teacher.get("stage", teacher.get("teacher_stage", None))
            teacher_raw_obj = teacher.get("raw_obj", teacher.get("raw_teacher_obj", None))
            teacher_gate = teacher.get("gate", teacher.get("teacher_reward_gate", None))
            teacher_action_sequence = teacher.get(
                "action_sequence",
                teacher.get("teacher_action_sequence", None),
            )

        if teacher_obj is None:
            for k in ["teacher_obj", "teacher_objective", "teacher_best_obj"]:
                if k in context:
                    teacher_obj = context.get(k)
                    break

            teacher_source = context.get("teacher_source", teacher_source)
            teacher_stage = context.get("teacher_stage", teacher_stage)
            teacher_raw_obj = context.get("teacher_raw_obj", teacher_raw_obj)
            teacher_gate = context.get("teacher_reward_gate", teacher_gate)
            teacher_action_sequence = context.get(
                "teacher_action_sequence",
                teacher_action_sequence,
            )

        if teacher_obj is not None:
            self.set_teacher_reference(
                teacher_obj=teacher_obj,
                source=teacher_source,
                stage=teacher_stage,
                enabled=True,
                raw_obj=teacher_raw_obj,
                gate=teacher_gate,
                action_sequence=teacher_action_sequence,
            )

    # =========================================================
    # Gym spaces
    # =========================================================
    def _observation_update(self):
        obs_dict = {
            "cus_loc": spaces.Box(
                low=0,
                high=1,
                shape=(self.cus_num, 2),
                dtype=np.float32,
            ),
            "depot_loc": spaces.Box(
                low=0,
                high=1,
                shape=(1, 2),
                dtype=np.float32,
            ),
            "rs_loc": spaces.Box(
                low=0,
                high=1,
                shape=(self.rs_num, 2),
                dtype=np.float32,
            ),
            "demand": spaces.Box(
                low=0,
                high=1,
                shape=(1 + self.cus_num + self.rs_num,),
                dtype=np.float32,
            ),
            "time_window": spaces.Box(
                low=0,
                high=1,
                shape=(1 + self.cus_num + self.rs_num, 2),
                dtype=np.float32,
            ),
            "action_mask": spaces.MultiBinary(
                [self.n_traj, self.cus_num + self.rs_num + 1]
            ),
            "last_node_idx": spaces.MultiDiscrete(
                [self.cus_num + self.rs_num + 1] * self.n_traj
            ),
            "prev_node_idx": spaces.MultiDiscrete(
                [self.cus_num + self.rs_num + 1] * self.n_traj
            ),
            "node_visit_count": spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.n_traj, self.cus_num + self.rs_num + 1),
                dtype=np.float32,
            ),
            "route_order_rank": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj, self.cus_num + self.rs_num + 1),
                dtype=np.float32,
            ),
            "route_event_nodes": spaces.Box(
                low=0,
                high=self.cus_num + self.rs_num,
                shape=(self.n_traj, self.max_route_events),
                dtype=np.int32,
            ),
            "route_event_mask": spaces.MultiBinary(
                [self.n_traj, self.max_route_events]
            ),
            "route_event_order_rank": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj, self.max_route_events),
                dtype=np.float32,
            ),
            "route_event_visit_count": spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.n_traj, self.max_route_events),
                dtype=np.float32,
            ),
            "current_load": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj,),
                dtype=np.float32,
            ),
            "current_battery": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj,),
                dtype=np.float32,
            ),
            "current_time": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj,),
                dtype=np.float32,
            ),
            "service_time": spaces.Box(
                low=0,
                high=1,
                shape=(1 + self.cus_num + self.rs_num,),
                dtype=np.float32,
            ),
            "battery_capacity": spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            ),
            "edge_energy": spaces.Box(
                low=0,
                high=np.inf,
                shape=(
                    self.cus_num + self.rs_num + 1,
                    self.cus_num + self.rs_num + 1,
                ),
                dtype=np.float32,
            ),
            "loading_capacity": spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            ),
            "visited_customers_ratio": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj, 1),
                dtype=np.float32,
            ),
            "route_served_customers_ratio": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj, 1),
                dtype=np.float32,
            ),
            "remain_feasible_customers_ratio": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj, 1),
                dtype=np.float32,
            ),
            "rs_streak_ratio": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_traj, 1),
                dtype=np.float32,
            ),
        }

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete(
            [self.rs_num + self.cus_num + 1] * self.n_traj
        )
        self.reward_space = None

    # =========================================================
    # Reset / data loading
    # =========================================================
    def reset(self):
        self.num_steps = 0
        self.info = {}

        teacher_snapshot = self._snapshot_teacher_reference()

        if self.env_mode == "eval":
            self._fixed_data_generate(is_train=False)

        elif self.env_mode == "train":
            if self.keep_instance_on_reset and self.fixed_data is not None:
                self._fixed_data_generate(is_train=True)
            elif self.fixed_data is not None:
                self._fixed_data_generate(is_train=True)
            else:
                self._train_data_generate()

        else:
            raise ValueError(f"Unknown env_mode: {self.env_mode}")

        if self.keep_teacher_on_reset:
            self._restore_teacher_reference(teacher_snapshot)
        else:
            # For fixed ALNS data, teacher can be stored directly inside instance.
            self._try_load_teacher_from_context(self.current_instance)

        self._ensure_buffers()
        self._reset_runtime_buffers()

        self.state = self._update_state()
        return self.state

    def _ensure_buffers(self):
        need_new = (
            not hasattr(self, "visited")
            or self.visited.shape != (self.n_traj, self.n_nodes)
            or not hasattr(self, "route_event_nodes")
            or self.route_event_nodes.shape != (self.n_traj, self.max_route_events)
        )

        if not need_new:
            return

        self.visited = np.empty((self.n_traj, self.n_nodes), dtype=np.bool_)
        self.node_visit_count = np.empty((self.n_traj, self.n_nodes), dtype=np.int32)
        self.route_visit_order = np.empty((self.n_traj, self.n_nodes), dtype=np.int32)
        self.route_step_count = np.empty(self.n_traj, dtype=np.int32)
        self.route_event_nodes = np.empty(
            (self.n_traj, self.max_route_events),
            dtype=np.int32,
        )
        self.route_event_step = np.empty(
            (self.n_traj, self.max_route_events),
            dtype=np.int32,
        )
        self.route_event_visit_count = np.empty(
            (self.n_traj, self.max_route_events),
            dtype=np.int32,
        )
        self.route_event_mask = np.empty(
            (self.n_traj, self.max_route_events),
            dtype=np.bool_,
        )
        self.last = np.empty(self.n_traj, dtype=np.int32)
        self.prev_node = np.empty(self.n_traj, dtype=np.int32)
        self.load = np.empty(self.n_traj, dtype=np.float32)
        self.battery = np.empty(self.n_traj, dtype=np.float32)
        self.current_time = np.empty(self.n_traj, dtype=np.float32)

        self.done = np.empty(self.n_traj, dtype=np.bool_)
        self.finish = np.empty(self.n_traj, dtype=np.bool_)

        self.served_cus = np.empty(self.n_traj, dtype=np.int32)
        self.objective = np.empty(self.n_traj, dtype=np.float32)

        self.route_served_cus_count = np.empty(self.n_traj, dtype=np.int32)
        self.route_distance = np.empty(self.n_traj, dtype=np.float32)
        self.empty_route_count = np.empty(self.n_traj, dtype=np.int32)
        self.rs_streak = np.empty(self.n_traj, dtype=np.int32)
        self.route_had_useful_rs = np.empty(self.n_traj, dtype=np.bool_)

        self.last_r_lag = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_teacher = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_objective = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_progress = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_task = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_terminal = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_repair_fail = np.zeros(self.n_traj, dtype=np.float32)
        self.last_remaining_repair_dist_norm = np.zeros(self.n_traj, dtype=np.float32)
        self.last_forced_terminal = np.zeros(self.n_traj, dtype=np.bool_)
        self.last_r_teacher_dense = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_teacher_route = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_local_invalid = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_pbrs_customer = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_pbrs_feasible = np.zeros(self.n_traj, dtype=np.float32)
        self.last_r_pbrs_repair = np.zeros(self.n_traj, dtype=np.float32)
        self.last_no_progress_rs = np.zeros(self.n_traj, dtype=np.bool_)
        self.last_empty_no_progress_route = np.zeros(self.n_traj, dtype=np.bool_)
        self.last_teacher_gap = np.full(self.n_traj, np.nan, dtype=np.float32)
        self.last_teacher_success_coef = np.ones(self.n_traj, dtype=np.float32)
        self.teacher_progress = np.zeros(self.n_traj, dtype=np.int32)
        self.teacher_route_rank_progress = np.zeros(self.n_traj, dtype=np.float32)
        self.last_success = np.zeros(self.n_traj, dtype=np.bool_)

    def _reset_runtime_buffers(self):
        self.terminate = False

        self.visited.fill(False)
        self.visited[:, 0] = True
        self.node_visit_count.fill(0)
        self.node_visit_count[:, 0] = 1
        self.route_visit_order.fill(0)
        self.route_visit_order[:, 0] = 1
        self.route_step_count.fill(1)
        self.route_event_nodes.fill(0)
        self.route_event_step.fill(0)
        self.route_event_visit_count.fill(0)
        self.route_event_mask.fill(False)
        self.route_event_nodes[:, 0] = 0
        self.route_event_step[:, 0] = 1
        self.route_event_visit_count[:, 0] = 1
        self.route_event_mask[:, 0] = True

        self.last.fill(0)
        self.prev_node.fill(0)
        self.load.fill(0.0)
        self.battery.fill(0.0)
        self.current_time.fill(0.0)
        self.served_cus.fill(0)

        self.done.fill(False)
        self.finish.fill(False)
        self.objective.fill(0.0)

        self.route_served_cus_count.fill(0)
        self.route_distance.fill(0.0)
        self.empty_route_count.fill(0)
        self.rs_streak.fill(0)
        self.route_had_useful_rs.fill(False)

        self.last_r_lag.fill(0.0)
        self.last_r_teacher.fill(0.0)
        self.last_r_objective.fill(0.0)
        self.last_r_progress.fill(0.0)
        self.last_r_task.fill(0.0)
        self.last_r_terminal.fill(0.0)
        self.last_r_repair_fail.fill(0.0)
        self.last_remaining_repair_dist_norm.fill(0.0)
        self.last_r_pbrs_repair.fill(0.0)
        self.last_forced_terminal.fill(False)
        self.last_r_teacher_dense.fill(0.0)
        self.last_r_teacher_route.fill(0.0)
        self.last_r_local_invalid.fill(0.0)
        self.last_r_pbrs_customer.fill(0.0)
        self.last_r_pbrs_feasible.fill(0.0)
        self.last_no_progress_rs.fill(False)
        self.last_empty_no_progress_route.fill(False)
        self.last_teacher_gap.fill(np.nan)
        self.last_teacher_success_coef.fill(1.0)
        self.teacher_progress.fill(0)
        self.teacher_route_rank_progress.fill(0.0)
        self.last_success.fill(False)
        self._prepare_teacher_route_reference()
        self.teacher_edge_matched = np.zeros(
            (self.n_traj, int(self.teacher_edge_from.size)),
            dtype=np.bool_,
        )

    def _train_data_generate(self):
        context = self.dataset.generate_tensors(
            perturb_dict=self.perturb_dict,
            num_customers=self.cus_num,
            num_charging_stations=self.rs_num,
            format="tensor",
        )

        self._load_context(context)
        self._set_success_bonus()
        self._normalizations(context)
        self._observation_update()

    def _fixed_data_generate(self, is_train: bool):
        context = deepcopy(self.fixed_data)

        if context is None:
            raise ValueError("Fixed data mode requires eval_data/fixed_data.")

        context["env"] = deepcopy(context["env"])
        context["env"]["battery_capacity"] *= self.Q_ratio
        context["env"]["charging_speed"] *= self.r_ratio

        self._load_context(context)
        self._set_success_bonus()
        self._normalizations(context)
        self._observation_update()

        if not is_train:
            self.use_teacher_reward = False

    def _load_context(self, context):
        self.context = context
        self.current_instance = context

        self.depot_num = context["depot"].shape[0]
        self.cus_num = context["customers"].shape[0]
        self.rs_num = context["charging_stations"].shape[0]

    def _set_success_bonus(self):
        base_bonus = 10.0
        ref_cus_num = 50.0
        self.success_bonus = base_bonus * np.sqrt(float(self.cus_num) / ref_cus_num)

    # =========================================================
    # Teacher reward / scaled success bonus
    # =========================================================
    def _get_teacher_obj_vector(self):
        if self.teacher_obj is None:
            return None

        teacher_obj = np.asarray(self.teacher_obj, dtype=np.float32)

        if teacher_obj.ndim == 0:
            return np.full((self.n_traj,), float(teacher_obj), dtype=np.float32)

        teacher_obj = teacher_obj.reshape(-1)

        if teacher_obj.shape[0] == self.n_traj:
            return teacher_obj.astype(np.float32, copy=False)

        if teacher_obj.shape[0] == 1:
            return np.full((self.n_traj,), float(teacher_obj[0]), dtype=np.float32)

        return None

    def _compute_teacher_success_coef(self, student_obj, teacher_obj):
        """
        Compute quality-aware success-bonus coefficient.

        rel_improve:
            > 0: student better than teacher
            = 0: equal to teacher
            < 0: student worse than teacher

        Returned coef is clipped to:
            [teacher_success_worse_floor, teacher_success_max_coef]
        """
        student_obj = np.asarray(student_obj, dtype=np.float32)
        teacher_obj = np.asarray(teacher_obj, dtype=np.float32)

        rel_improve = (teacher_obj - student_obj) / (teacher_obj + 1e-8)

        margin = float(self.teacher_reward_margin)
        worse_floor = float(self.teacher_success_worse_floor)
        equal_coef = float(self.teacher_success_equal_coef)
        max_coef = float(self.teacher_success_max_coef)

        coef = np.full_like(rel_improve, equal_coef, dtype=np.float32)

        # -----------------------------------------------------
        # Case 1: student worse than teacher.
        # Smoothly decay from equal_coef to worse_floor.
        # -----------------------------------------------------
        worse_mask = rel_improve < -margin

        if np.any(worse_mask):
            worse_temp = max(float(self.teacher_success_worse_temp), 1e-8)

            coef[worse_mask] = worse_floor + (equal_coef - worse_floor) * np.exp(
                rel_improve[worse_mask] / worse_temp
            )

        # -----------------------------------------------------
        # Case 2: student better than teacher.
        # -----------------------------------------------------
        better_mask = rel_improve > margin

        if np.any(better_mask):
            improve = rel_improve[better_mask] - margin

            if self.teacher_success_coef_type == "saturation":
                scale = max(float(self.teacher_success_improve_scale), 1e-8)

                coef[better_mask] = equal_coef + (max_coef - equal_coef) * (
                    1.0 - np.exp(-improve / scale)
                )

            elif self.teacher_success_coef_type == "convex":
                target = max(float(self.teacher_success_target_improve) - margin, 1e-8)
                power = max(float(self.teacher_success_power), 1e-8)

                x = improve / target
                coef[better_mask] = equal_coef + (max_coef - equal_coef) * (x ** power)

            elif self.teacher_success_coef_type == "linear":
                target = max(float(self.teacher_success_target_improve) - margin, 1e-8)

                x = improve / target
                coef[better_mask] = equal_coef + (max_coef - equal_coef) * x

            else:
                raise ValueError(
                    f"Unknown teacher_success_coef_type: {self.teacher_success_coef_type}"
                )

        coef = np.clip(coef, worse_floor, max_coef).astype(np.float32)

        return coef, rel_improve.astype(np.float32)

    def _compute_teacher_scaled_success_bonus(self, new_done, success):
        """
        Teacher-calibrated success bonus.

        For successful terminal trajectories:
            r_success = success_bonus * quality_coef

        For failed terminal trajectories:
            by default, no teacher-specific failure penalty here.
            Failure is handled by lambda_fail outside.
        """
        r_scaled_success = np.zeros_like(self.reward, dtype=np.float32)
        teacher_gap = np.full_like(self.reward, np.nan, dtype=np.float32)
        teacher_success_coef = np.ones_like(self.reward, dtype=np.float32)

        if self.env_mode != "train":
            return r_scaled_success, teacher_gap, teacher_success_coef

        if not self.use_teacher_reward:
            return r_scaled_success, teacher_gap, teacher_success_coef

        teacher_obj = self._get_teacher_obj_vector()
        if teacher_obj is None:
            return r_scaled_success, teacher_gap, teacher_success_coef

        valid_teacher = np.isfinite(teacher_obj) & (teacher_obj > 0.0)
        if not np.any(valid_teacher):
            return r_scaled_success, teacher_gap, teacher_success_coef

        success_mask = (
            new_done
            & success
            & valid_teacher
            & np.isfinite(self.objective)
            & (self.objective > 0.0)
        )

        if np.any(success_mask):
            coef, _rel_improve = self._compute_teacher_success_coef(
                student_obj=self.objective[success_mask],
                teacher_obj=teacher_obj[success_mask],
            )
            gate = float(np.clip(self.teacher_reward_gate, 0.0, 1.0))
            base_coef = float(self.plain_success_bonus_coef)
            coef = base_coef + gate * (coef - base_coef)
            if self.teacher_gap_worse_only:
                coef = np.minimum(coef, base_coef)

            # Logging compatibility:
            # old teacher_gap = (student - teacher) / teacher
            teacher_gap[success_mask] = (
                (self.objective[success_mask] - teacher_obj[success_mask])
                / (teacher_obj[success_mask] + 1e-8)
            ).astype(np.float32)

            teacher_success_coef[success_mask] = coef.astype(np.float32)
            r_scaled_success[success_mask] += (
                float(self.success_bonus) * coef
            ).astype(np.float32)

        # Optional teacher-aware failure penalty.
        if self.teacher_scaled_failure:
            fail_mask = new_done & (~success) & valid_teacher

            if np.any(fail_mask):
                served_ratio = self.served_cus.astype(np.float32) / float(self.cus_num)
                unserved_ratio = 1.0 - served_ratio

                failure_scale = float(self.success_bonus) * float(self.teacher_failure_ratio)
                failure_severity = unserved_ratio.copy()

                closure_fail_mask = (
                    fail_mask
                    & (self.served_cus == self.cus_num)
                    & (self.last != 0)
                )

                if np.any(closure_fail_mask):
                    failure_severity[closure_fail_mask] = np.maximum(
                        failure_severity[closure_fail_mask],
                        float(self.teacher_closure_failure_min_ratio),
                    )

                r_scaled_success[fail_mask] -= (
                    float(np.clip(self.teacher_reward_gate, 0.0, 1.0))
                    * failure_scale
                    * failure_severity[fail_mask]
                ).astype(np.float32)

        return r_scaled_success, teacher_gap, teacher_success_coef

    def _compute_teacher_terminal_reward(self, new_done, success):
        """
        Old additive teacher reward.

        Kept only for backward compatibility when:
            teacher_reward_mode == "additive"

        Recommended new mode:
            teacher_reward_mode == "scaled_success"
        """
        r_teacher = np.zeros_like(self.reward, dtype=np.float32)
        teacher_gap = np.full_like(self.reward, np.nan, dtype=np.float32)

        if self.env_mode != "train":
            return r_teacher, teacher_gap

        if not self.use_teacher_reward:
            return r_teacher, teacher_gap

        teacher_obj = self._get_teacher_obj_vector()
        if teacher_obj is None:
            return r_teacher, teacher_gap

        valid_teacher = np.isfinite(teacher_obj) & (teacher_obj > 0.0)
        if not np.any(valid_teacher):
            return r_teacher, teacher_gap

        temp = max(float(self.teacher_reward_temp), 1e-8)
        margin = float(self.teacher_reward_margin)
        gate = float(np.clip(self.teacher_reward_gate, 0.0, 1.0))

        penalty_scale = float(self.success_bonus) * float(self.teacher_penalty_ratio)
        bonus_scale = float(self.success_bonus) * float(self.teacher_bonus_ratio)
        failure_scale = float(self.success_bonus) * float(self.teacher_failure_ratio)

        success_mask = (
            new_done
            & success
            & valid_teacher
            & np.isfinite(self.objective)
            & (self.objective > 0.0)
        )

        if np.any(success_mask):
            gap = (
                self.objective[success_mask] - teacher_obj[success_mask]
            ) / (teacher_obj[success_mask] + 1e-8)

            teacher_gap[success_mask] = gap.astype(np.float32)

            shaped = np.zeros_like(gap, dtype=np.float32)

            worse = gap > margin
            better = gap < -margin

            if np.any(worse):
                shaped[worse] = -gate * penalty_scale * np.tanh(
                    (gap[worse] - margin) / temp
                )

            if np.any(better):
                shaped[better] = gate * bonus_scale * np.tanh(
                    (-gap[better] - margin) / temp
                )

            r_teacher[success_mask] += shaped.astype(np.float32)

        fail_mask = new_done & (~success) & valid_teacher

        if np.any(fail_mask):
            served_ratio = self.served_cus.astype(np.float32) / float(self.cus_num)
            unserved_ratio = 1.0 - served_ratio

            failure_severity = unserved_ratio.copy()

            closure_fail_mask = (
                fail_mask
                & (self.served_cus == self.cus_num)
                & (self.last != 0)
            )

            if np.any(closure_fail_mask):
                failure_severity[closure_fail_mask] = np.maximum(
                    failure_severity[closure_fail_mask],
                    float(self.teacher_closure_failure_min_ratio),
                )

            r_teacher[fail_mask] -= (
                gate * failure_scale * failure_severity[fail_mask]
            ).astype(np.float32)

        return r_teacher, teacher_gap

    def _prepare_teacher_route_reference(self):
        node_count = int(getattr(self, "n_nodes", 0))
        self.teacher_route_rank_by_node = np.full(
            node_count,
            -1.0,
            dtype=np.float32,
        )
        self.teacher_route_rank_denom = 0.0

        seq = self.teacher_action_sequence
        if seq is None:
            self.teacher_edge_from = np.empty(0, dtype=np.int32)
            self.teacher_edge_to = np.empty(0, dtype=np.int32)
            return

        try:
            seq = np.asarray(seq, dtype=np.int64).reshape(-1)
        except Exception:
            self.teacher_edge_from = np.empty(0, dtype=np.int32)
            self.teacher_edge_to = np.empty(0, dtype=np.int32)
            return

        if node_count > 0 and hasattr(self, "cus_start") and hasattr(self, "rs_start"):
            rank = 0
            for raw_action in seq:
                action = int(raw_action)
                if action < self.cus_start or action >= self.rs_start:
                    continue
                if action >= node_count:
                    continue
                if self.teacher_route_rank_by_node[action] >= 0.0:
                    continue
                rank += 1
                self.teacher_route_rank_by_node[action] = float(rank)
            self.teacher_route_rank_denom = float(max(rank, 1))

        if seq.size < 2:
            self.teacher_edge_from = np.empty(0, dtype=np.int32)
            self.teacher_edge_to = np.empty(0, dtype=np.int32)
            return

        valid = (
            (seq[:-1] >= 0)
            & (seq[1:] >= 0)
            & (seq[:-1] != seq[1:])
        )

        if hasattr(self, "n_nodes"):
            valid &= (seq[:-1] < self.n_nodes) & (seq[1:] < self.n_nodes)

        self.teacher_edge_from = seq[:-1][valid].astype(np.int32)
        self.teacher_edge_to = seq[1:][valid].astype(np.int32)

    def _compute_teacher_route_potential_reward(self, prev_last, action):
        r_route = np.zeros(self.n_traj, dtype=np.float32)

        if self.env_mode != "train":
            return r_route
        if not (self.use_teacher_reward and self.use_teacher_route_potential):
            return r_route
        if self.teacher_route_potential_coef == 0.0:
            return r_route

        has_edge_potential = self.teacher_edge_from.size > 0
        has_rank_potential = (
            self.teacher_route_rank_by_node.size > 0
            and self.teacher_route_rank_denom > 0.0
        )
        if not (has_edge_potential or has_rank_potential):
            return r_route

        prev_last = np.asarray(prev_last, dtype=np.int32).reshape(self.n_traj)
        action = np.asarray(action, dtype=np.int32).reshape(self.n_traj)

        pre_parts = []
        post_parts = []

        if has_edge_potential:
            if self.teacher_edge_matched.shape != (
                self.n_traj,
                self.teacher_edge_from.size,
            ):
                self.teacher_edge_matched = np.zeros(
                    (self.n_traj, int(self.teacher_edge_from.size)),
                    dtype=np.bool_,
                )

            denom = max(float(self.teacher_edge_from.size), 1.0)
            pre_edge_phi = (
                self.teacher_edge_matched.sum(axis=1).astype(np.float32) / denom
            )

            for traj_idx in range(self.n_traj):
                matches = np.where(
                    (~self.teacher_edge_matched[traj_idx])
                    & (self.teacher_edge_from == prev_last[traj_idx])
                    & (self.teacher_edge_to == action[traj_idx])
                )[0]
                if matches.size > 0:
                    self.teacher_edge_matched[traj_idx, int(matches[0])] = True

            post_edge_phi = (
                self.teacher_edge_matched.sum(axis=1).astype(np.float32) / denom
            )
            pre_parts.append(pre_edge_phi)
            post_parts.append(post_edge_phi)

        if has_rank_potential:
            rank_denom = max(float(self.teacher_route_rank_denom), 1.0)
            pre_rank_phi = self.teacher_route_rank_progress.astype(np.float32) / rank_denom

            valid_action = (
                (action >= 0)
                & (action < self.teacher_route_rank_by_node.size)
                & (action >= self.cus_start)
                & (action < self.rs_start)
            )
            if np.any(valid_action):
                action_rank = np.full(self.n_traj, -1.0, dtype=np.float32)
                action_rank[valid_action] = self.teacher_route_rank_by_node[
                    action[valid_action]
                ]
                improve = action_rank > self.teacher_route_rank_progress
                self.teacher_route_rank_progress[improve] = action_rank[improve]

            post_rank_phi = self.teacher_route_rank_progress.astype(np.float32) / rank_denom
            pre_parts.append(pre_rank_phi)
            post_parts.append(post_rank_phi)

        pre_phi = np.mean(np.stack(pre_parts, axis=0), axis=0)
        post_phi = np.mean(np.stack(post_parts, axis=0), axis=0)
        r_route = float(self.teacher_route_potential_coef) * (
            float(self.gamma) * post_phi - pre_phi
        )

        clip_value = float(self.teacher_route_potential_clip)
        if clip_value > 0.0:
            r_route = np.clip(r_route, -clip_value, clip_value)

        return r_route.astype(np.float32)

    def _compute_teacher_dense_route_reward(self, action):
        r_dense = np.zeros(self.n_traj, dtype=np.float32)

        if self.env_mode != "train":
            return r_dense
        if not (self.use_teacher_reward and self.use_teacher_dense_reward):
            return r_dense
        if self.teacher_action_sequence is None or len(self.teacher_action_sequence) == 0:
            return r_dense
        if self.teacher_dense_coef <= 0.0:
            return r_dense

        seq = np.asarray(self.teacher_action_sequence, dtype=np.int64).reshape(-1)
        if seq.size == 0:
            return r_dense

        active = ~self.finish
        action_i = action.astype(np.int64)

        progress = np.clip(self.teacher_progress, 0, seq.size - 1)
        expected = seq[progress]
        match = active & (action_i == expected.astype(np.int64))
        matched_pos = progress.copy()
        rank_weight = np.ones(self.n_traj, dtype=np.float32)

        if self.teacher_dense_mode in ("lookahead", "soft"):
            unmatched = active & (~match)
            # Allow local customer reordering without breaking teacher progress.
            # Depot/charging-station moves remain exact to avoid rewarding
            # premature route closure or charging loops.
            customer_action = (
                (action_i >= self.cus_start)
                & (action_i < self.rs_start)
            )
            candidates = np.where(unmatched & customer_action)[0]
            lookahead = max(1, int(self.teacher_dense_lookahead))
            tau = max(float(self.teacher_dense_rank_tau), 1e-6)

            for traj_idx in candidates:
                start = int(progress[traj_idx])
                end = min(seq.size, start + lookahead)
                if start >= end:
                    continue

                window = seq[start:end]
                hits = np.where(window == action_i[traj_idx])[0]
                if hits.size == 0:
                    continue

                rank = int(hits[0])
                match[traj_idx] = True
                matched_pos[traj_idx] = start + rank
                rank_weight[traj_idx] = float(np.exp(-float(rank) / tau))

        if np.any(match):
            self.teacher_progress[match] = np.minimum(
                matched_pos[match] + 1,
                seq.size,
            )
            norm = max(1.0, float(self.cus_num))
            gate = float(np.clip(self.teacher_reward_gate, 0.0, 1.0))
            r_dense[match] += (
                float(self.teacher_dense_coef)
                * gate
                * rank_weight[match]
                / norm
            )

        return r_dense.astype(np.float32)

    def _customer_feasible_ratio(self, action_mask=None):
        if action_mask is None:
            action_mask = self.mask if hasattr(self, "mask") else self._update_mask()

        feasible_unvisited = (
            (~self.visited[:, self.cus_start:self.rs_start])
            & action_mask[:, self.cus_start:self.rs_start]
        )
        return (
            feasible_unvisited.sum(axis=1).astype(np.float32) / float(self.cus_num)
        )

    def _served_customer_potential(self):
        n = float(self.cus_num)
        beta = max(float(self.beta), 1e-8)
        remaining_ratio = (
            (n - self.served_cus.astype(np.float32)) / n
        ).clip(0.0, 1.0)
        return 1.0 - (remaining_ratio ** beta)

    def _direct_served_progress_potential(self, served_counts):
        n = max(float(self.cus_num), 1.0)
        beta = max(float(self.progress_pbrs_beta), 1e-8)
        served_ratio = (
            np.asarray(served_counts, dtype=np.float32) / n
        ).clip(0.0, 1.0)
        remaining_ratio = (1.0 - served_ratio).clip(0.0, 1.0)
        return 1.0 - (remaining_ratio ** beta)

    def _remaining_repair_dist_norm(self):
        unvisited_customers = (
            ~self.visited[:, self.cus_start:self.rs_start]
        ).astype(np.float32)
        remaining_customer_dist = unvisited_customers @ self.single_customer_repair_dist_norm
        current_to_depot = self.node_to_depot_repair_dist_norm[self.last]
        return (current_to_depot + remaining_customer_dist).astype(np.float32)

    def _remaining_customer_repair_dist_norm(self):
        unvisited_customers = (
            ~self.visited[:, self.cus_start:self.rs_start]
        ).astype(np.float32)
        return (
            unvisited_customers @ self.single_customer_repair_dist_norm
        ).astype(np.float32)

    def _remaining_customer_repair_ratio(self):
        total = max(float(getattr(self, "total_customer_repair_dist_norm", 1.0)), 1e-6)
        return np.clip(
            self._remaining_customer_repair_dist_norm() / np.float32(total),
            0.0,
            1.0,
        ).astype(np.float32)

    def _remaining_repair_ratio(self):
        total = max(float(getattr(self, "total_customer_repair_dist_norm", 1.0)), 1e-6)
        return np.maximum(
            self._remaining_repair_dist_norm() / np.float32(total),
            0.0,
        ).astype(np.float32)

    def _served_customer_pbrs_scale(self):
        n = float(self.cus_num)
        beta = max(float(self.beta), 1e-8)
        reward_ref = np.sqrt(2.0)
        return (
            float(self.pbrs_coef)
            * float(self.alpha)
            * reward_ref
            * (n ** beta)
        )

    # =========================================================
    # Step logic
    # =========================================================
    def _STEP(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(self.n_traj)

        compute_progress_pbrs = (
            self.env_mode == "train"
            and self.phi_reward
            and self.pbrs_mode == "progress"
            and self.pbrs_coef != 0.0
        )
        compute_local_invalid = (
            self.env_mode == "train"
            and (
                self.no_progress_rs_penalty != 0.0
                or self.empty_no_progress_route_penalty != 0.0
                or self.rs_streak_penalty != 0.0
            )
        )

        prev_mask = None
        prev_customer_potential = None
        prev_feasible_potential = None
        prev_feasible_ratio = None
        prev_depot_feasible = None
        prev_route_served_cus_count = self.route_served_cus_count.copy()
        prev_route_had_useful_rs = self.route_had_useful_rs.copy()

        if compute_progress_pbrs or compute_local_invalid:
            prev_mask = self.mask if hasattr(self, "mask") else self._update_mask()
            if compute_progress_pbrs:
                prev_customer_potential = self._served_customer_potential()
                prev_feasible_potential = self._customer_feasible_ratio(
                    action_mask=prev_mask
                )
            if compute_local_invalid:
                prev_feasible_ratio = self._customer_feasible_ratio(
                    action_mask=prev_mask
                )
                prev_depot_feasible = prev_mask[:, 0].copy()

        self.last_r_objective.fill(0.0)
        self.last_r_progress.fill(0.0)
        self.last_r_local_invalid.fill(0.0)
        self.last_r_pbrs_customer.fill(0.0)
        self.last_r_pbrs_feasible.fill(0.0)
        self.last_r_pbrs_repair.fill(0.0)
        self.last_r_repair_fail.fill(0.0)
        self.last_remaining_repair_dist_norm.fill(0.0)
        self.last_forced_terminal.fill(False)
        self.last_no_progress_rs.fill(False)
        self.last_empty_no_progress_route.fill(False)

        compute_repair_progress_pbrs = (
            self.env_mode == "train"
            and self.phi_reward
            and self.use_repair_fail_reward
            and float(self.repair_progress_coef) != 0.0
        )
        prev_repair_ratio = None
        if compute_repair_progress_pbrs:
            prev_repair_ratio = self._remaining_repair_ratio()

        self._go_to(action)

        if compute_repair_progress_pbrs:
            post_repair_ratio = self._remaining_repair_ratio()
            repair_pbrs = float(self.repair_progress_coef) * (
                prev_repair_ratio - post_repair_ratio
            )
            repair_pbrs = repair_pbrs.astype(np.float32)
            self.reward += repair_pbrs
            self.last_r_task += repair_pbrs
            self.last_r_progress += repair_pbrs
            self.last_r_pbrs_repair = repair_pbrs.copy()

        post_mask = None
        if compute_progress_pbrs or compute_local_invalid:
            post_mask = self._update_mask()

        if compute_progress_pbrs:
            post_customer_potential = self._served_customer_potential()
            post_feasible_potential = self._customer_feasible_ratio(
                action_mask=post_mask
            )

            customer_pbrs = self._served_customer_pbrs_scale() * (
                float(self.gamma) * post_customer_potential
                - prev_customer_potential
            )
            feasible_pbrs = float(self.pbrs_feasible_coef) * (
                float(self.gamma) * post_feasible_potential
                - prev_feasible_potential
            )
            pbrs_reward = customer_pbrs + feasible_pbrs
            self.reward += pbrs_reward.astype(np.float32)
            self.last_r_task += pbrs_reward.astype(np.float32)
            self.last_r_progress += pbrs_reward.astype(np.float32)
            self.last_r_pbrs_customer = customer_pbrs.astype(np.float32).copy()
            self.last_r_pbrs_feasible = feasible_pbrs.astype(np.float32).copy()

        if compute_local_invalid:
            post_feasible_ratio = self._customer_feasible_ratio(action_mask=post_mask)
            post_depot_feasible = post_mask[:, 0].copy()

            go_to_rs = action >= self.rs_start
            go_to_depot = action == 0

            useful_rs = go_to_rs & (
                (post_feasible_ratio > prev_feasible_ratio + float(self.rs_progress_eps))
                | ((~prev_depot_feasible) & post_depot_feasible)
                | (prev_feasible_ratio <= float(self.rs_progress_eps))
            )
            if np.any(useful_rs):
                self.route_had_useful_rs[useful_rs] = True

            no_progress_rs = (
                go_to_rs
                & (~useful_rs)
                & (self.rs_streak > int(self.rs_streak_free))
            )
            if np.any(no_progress_rs):
                streak_extra = np.maximum(
                    0,
                    self.rs_streak[no_progress_rs] - int(self.rs_streak_free),
                ).astype(np.float32)
                penalty = (
                    float(self.no_progress_rs_penalty)
                    + float(self.rs_streak_penalty) * streak_extra
                )
                self.last_r_local_invalid[no_progress_rs] -= penalty.astype(np.float32)
                self.last_no_progress_rs[no_progress_rs] = True

            empty_no_progress_route = (
                go_to_depot
                & (prev_route_served_cus_count == 0)
                & (~prev_route_had_useful_rs)
            )
            if np.any(empty_no_progress_route):
                self.last_r_local_invalid[empty_no_progress_route] -= float(
                    self.empty_no_progress_route_penalty
                )
                self.last_empty_no_progress_route[empty_no_progress_route] = True

            if np.any(go_to_depot):
                self.route_had_useful_rs[go_to_depot] = False

        self.num_steps += 1

        all_visited = self.is_all_visited()

        if self.terminate:
            self.done = np.ones_like(self.done, dtype=np.bool_)
        else:
            self.done = (action == 0) & all_visited

        new_done = self.done & (~self.finish)
        forced_terminal = bool(self.terminate)

        if self.terminate:
            success = all_visited & (self.last == 0)
        else:
            success = self.done

        r_lag = np.zeros_like(self.reward, dtype=np.float32)
        r_teacher = np.zeros_like(self.reward, dtype=np.float32)
        r_teacher_component = r_teacher.copy()
        self.last_r_terminal.fill(0.0)
        teacher_gap = np.full_like(self.reward, np.nan, dtype=np.float32)
        teacher_success_coef = np.ones_like(self.reward, dtype=np.float32)

        if compute_local_invalid:
            terminal_empty_no_progress_route = (
                new_done
                & (~success)
                & (self.last != 0)
                & (self.route_served_cus_count == 0)
                & (~self.route_had_useful_rs)
            )
            if np.any(terminal_empty_no_progress_route):
                self.last_r_local_invalid[terminal_empty_no_progress_route] -= float(
                    self.empty_no_progress_route_penalty
                )
                self.last_empty_no_progress_route[
                    terminal_empty_no_progress_route
                ] = True

        if self.env_mode == "train":
            terminal_empty_route_mask = (
                new_done
                & (self.last != 0)
                & (self.route_served_cus_count == 0)
            )

            if np.any(terminal_empty_route_mask):
                self.empty_route_count[terminal_empty_route_mask] += 1
                count_penalty = self.empty_route_count_penalty * np.minimum(
                    self.empty_route_count[terminal_empty_route_mask],
                    self.empty_route_count_cap,
                ).astype(np.float32)
                penalty = (
                    self.empty_route_penalty
                    + self.empty_route_length_coef
                    * self.route_distance[terminal_empty_route_mask]
                    + count_penalty
                )
                self.reward[terminal_empty_route_mask] -= penalty.astype(np.float32)
                self.last_r_task[terminal_empty_route_mask] -= penalty.astype(np.float32)

            if self.lag_reward:
                success_mask = new_done & success
                fail_mask = new_done & (~success)

                if self.use_repair_fail_reward:
                    if np.any(success_mask) and self.repair_success_bonus != 0.0:
                        r_lag[success_mask] += float(self.repair_success_bonus)
                        self.last_r_progress[success_mask] += float(
                            self.repair_success_bonus
                        )
                        self.last_r_task[success_mask] += float(
                            self.repair_success_bonus
                        )

                    if np.any(fail_mask):
                        repair_dist = self._remaining_repair_ratio()
                        repair_reward = -float(self.repair_fail_coef) * repair_dist
                        r_lag[fail_mask] += repair_reward[fail_mask].astype(np.float32)
                        self.last_r_repair_fail[fail_mask] = repair_reward[
                            fail_mask
                        ].astype(np.float32)
                        self.last_remaining_repair_dist_norm[fail_mask] = repair_dist[
                            fail_mask
                        ].astype(np.float32)
                        self.last_r_progress[fail_mask] += repair_reward[
                            fail_mask
                        ].astype(np.float32)
                        self.last_r_task[fail_mask] += repair_reward[
                            fail_mask
                        ].astype(np.float32)

                else:
                    # -------------------------------------------------
                    # Success reward
                    # -------------------------------------------------
                    if np.any(success_mask):
                        if (
                            self.use_teacher_reward
                            and self.teacher_reward_mode == "scaled_success"
                        ):
                            r_scaled_success, teacher_gap, teacher_success_coef = (
                                self._compute_teacher_scaled_success_bonus(
                                    new_done=new_done,
                                    success=success,
                                )
                            )
                            r_lag += r_scaled_success

                        elif (
                            self.use_teacher_reward
                            and self.teacher_reward_mode == "additive"
                        ):
                            # Old behavior:
                            # success gets plain success_bonus,
                            # teacher adds separate reward.
                            r_lag[success_mask] += self.success_bonus
                            r_teacher, teacher_gap = self._compute_teacher_terminal_reward(
                                new_done=new_done,
                                success=success,
                            )

                        elif self.teacher_reward_mode == "none":
                            r_lag[success_mask] += (
                                float(self.success_bonus)
                                * float(self.plain_success_bonus_coef)
                            )

                        else:
                            # No valid teacher reference.
                            r_lag[success_mask] += (
                                float(self.success_bonus)
                                * float(self.plain_success_bonus_coef)
                            )

                    # -------------------------------------------------
                    # Failure reward
                    # Keep original lambda_fail failure penalty.
                    # -------------------------------------------------
                    if np.any(fail_mask):
                        served_ratio = self.served_cus.astype(np.float32) / float(self.cus_num)
                        unserved_ratio = 1.0 - served_ratio
                        r_lag[fail_mask] -= (
                            self.lambda_fail * unserved_ratio[fail_mask]
                        ).astype(np.float32)

            if self.use_repair_fail_reward:
                r_terminal_component = np.zeros_like(r_lag, dtype=np.float32)
            else:
                r_terminal_component = r_lag.copy()

            if (
                self.use_teacher_reward
                and self.teacher_reward_mode == "scaled_success"
            ):
                success_mask = new_done & success
                if np.any(success_mask):
                    plain_success = (
                        float(self.success_bonus)
                        * float(self.plain_success_bonus_coef)
                    )
                    teacher_part = r_lag[success_mask] - plain_success
                    r_teacher_component[success_mask] += teacher_part.astype(np.float32)
                    r_terminal_component[success_mask] -= teacher_part.astype(np.float32)

                fail_mask = new_done & (~success)
                if np.any(fail_mask) and self.teacher_scaled_failure:
                    served_ratio = self.served_cus.astype(np.float32) / float(self.cus_num)
                    base_fail = -self.lambda_fail * (1.0 - served_ratio[fail_mask])
                    teacher_part = r_lag[fail_mask] - base_fail.astype(np.float32)
                    r_teacher_component[fail_mask] += teacher_part.astype(np.float32)
                    r_terminal_component[fail_mask] -= teacher_part.astype(np.float32)

            r_teacher_component += self.last_r_teacher_dense + self.last_r_teacher_route
            self.last_r_terminal = r_terminal_component.astype(np.float32).copy()
            self.reward = self.reward + r_lag + r_teacher

        fail_mask_obj = new_done & (~success)
        if np.any(fail_mask_obj):
            self.objective[fail_mask_obj] = np.inf

        self.finish = self.finish | self.done

        if self.terminate:
            self.terminate = False

        self.last_r_lag = r_lag.copy()
        self.last_r_teacher = r_teacher_component.copy()
        self.last_teacher_gap = teacher_gap.copy()
        self.last_teacher_success_coef = teacher_success_coef.copy()
        self.last_success = success.copy()
        if forced_terminal:
            self.last_forced_terminal[new_done] = True

        self.info = {
            "objective": self.objective.copy(),
            "success": success.copy(),
            "forced_terminal": self.last_forced_terminal.copy(),
            "served_cus": self.served_cus.copy(),
            "empty_route_count": self.empty_route_count.copy(),
            "r_lag": r_lag.copy(),
            "r_teacher": self.last_r_teacher.copy(),
            "r_teacher_dense": self.last_r_teacher_dense.copy(),
            "r_teacher_route": self.last_r_teacher_route.copy(),
            "r_objective": self.last_r_objective.copy(),
            "r_progress": self.last_r_progress.copy(),
            "r_repair_fail": self.last_r_repair_fail.copy(),
            "remaining_repair_dist_norm": self.last_remaining_repair_dist_norm.copy(),
            "r_local_invalid": self.last_r_local_invalid.copy(),
            "r_pbrs_customer": self.last_r_pbrs_customer.copy(),
            "r_pbrs_feasible": self.last_r_pbrs_feasible.copy(),
            "r_pbrs_repair": self.last_r_pbrs_repair.copy(),
            "no_progress_rs": self.last_no_progress_rs.copy(),
            "empty_no_progress_route": self.last_empty_no_progress_route.copy(),
            "reward_components": {
                "objective": self.last_r_objective.copy(),
                "progress": self.last_r_progress.copy(),
                "task": self.last_r_task.copy(),
                "terminal": self.last_r_terminal.copy(),
                "repair_fail": self.last_r_repair_fail.copy(),
                "teacher": self.last_r_teacher.copy(),
                "local_invalid": self.last_r_local_invalid.copy(),
                "pbrs_customer": self.last_r_pbrs_customer.copy(),
                "pbrs_feasible": self.last_r_pbrs_feasible.copy(),
                "pbrs_repair": self.last_r_pbrs_repair.copy(),
            },
            "teacher_gap": teacher_gap.copy(),
            "teacher_success_coef": teacher_success_coef.copy(),
            "teacher_reward_gate": np.full(
                self.n_traj,
                float(np.clip(self.teacher_reward_gate, 0.0, 1.0)),
                dtype=np.float32,
            ),
            "teacher_reward_mode": self.teacher_reward_mode,
        }

        teacher_obj_vec = self._get_teacher_obj_vector()
        if teacher_obj_vec is not None:
            self.info["teacher_obj"] = teacher_obj_vec.copy()
            self.info["teacher_source"] = self.teacher_source
            self.info["teacher_stage"] = self.teacher_stage
            self.info["teacher_raw_obj"] = self.teacher_raw_obj
            self.info["use_teacher_reward"] = bool(self.use_teacher_reward)
        else:
            self.info["use_teacher_reward"] = False

        self.state = self._update_state()

        return self.state, self.reward.copy(), self.done.copy(), self.info

    def _go_to(self, destination):
        prev_last = self.last.copy()
        dist_display = self.dist_matrix[self.last, destination].astype(np.float32)

        self.objective += dist_display
        self.route_distance += dist_display

        go_to_depot = destination == 0
        go_to_rs = destination >= self.rs_start
        go_to_cus = (destination >= self.cus_start) & (destination < self.rs_start)
        self.rs_streak[go_to_rs] += 1
        self.rs_streak[go_to_depot | go_to_cus] = 0
        go_to_rs_or_cus = ~go_to_depot

        if self.env_mode == "eval":
            self.reward = -dist_display.astype(np.float32)
            self.last_r_objective = self.reward.astype(np.float32).copy()
            self.last_r_progress.fill(0.0)
            self.last_r_task = self.reward.astype(np.float32).copy()
            self.last_r_teacher_dense.fill(0.0)
            self.last_r_teacher_route.fill(0.0)

        elif self.env_mode == "train":
            r_objective = -dist_display.astype(np.float32).copy()
            r_progress = np.zeros(self.n_traj, dtype=np.float32)
            reward = r_objective.copy()

            if self.phi_reward and self.use_direct_progress_pbrs:
                prev_served_cus = self.served_cus.astype(np.float32)
                next_served_cus = prev_served_cus + go_to_cus.astype(np.float32)

                phi_s = self._direct_served_progress_potential(prev_served_cus)
                phi_sp = self._direct_served_progress_potential(next_served_cus)
                pbrs_reward = float(self.progress_pbrs_coef) * (phi_sp - phi_s)

                reward += pbrs_reward.astype(np.float32)
                r_progress += pbrs_reward.astype(np.float32)
                self.last_r_pbrs_customer = pbrs_reward.astype(np.float32).copy()

            elif self.phi_reward and self.pbrs_mode == "served":
                prev_served_cus = self.served_cus.astype(np.float32)
                next_served_cus = prev_served_cus + go_to_cus.astype(np.float32)

                n = float(self.cus_num)
                beta = float(self.beta)
                gamma = float(self.gamma)
                eta = float(self.alpha)
                reward_ref = np.sqrt(2.0)

                coef = eta * reward_ref * (n ** beta)

                phi_s = -((n - prev_served_cus) / n) ** beta
                phi_sp = -((n - next_served_cus) / n) ** beta

                pbrs_reward = coef * (gamma * phi_sp - phi_s)
                reward += pbrs_reward.astype(np.float32)
                r_progress += pbrs_reward.astype(np.float32)
                self.last_r_pbrs_customer = pbrs_reward.astype(np.float32).copy()

            empty_route_mask = go_to_depot & (self.route_served_cus_count == 0)

            if np.any(empty_route_mask):
                self.empty_route_count[empty_route_mask] += 1
                count_penalty = self.empty_route_count_penalty * np.minimum(
                    self.empty_route_count[empty_route_mask],
                    self.empty_route_count_cap,
                ).astype(np.float32)
                penalty = (
                    self.empty_route_penalty
                    + self.empty_route_length_coef
                    * self.route_distance[empty_route_mask]
                    + count_penalty
                )
                reward[empty_route_mask] -= penalty.astype(np.float32)
                r_progress[empty_route_mask] -= penalty.astype(np.float32)

            r_teacher_dense = self._compute_teacher_dense_route_reward(destination)
            r_teacher_route = self._compute_teacher_route_potential_reward(
                prev_last=prev_last,
                action=destination,
            )
            reward += r_teacher_dense + r_teacher_route
            self.last_r_teacher_dense = r_teacher_dense.astype(np.float32).copy()
            self.last_r_teacher_route = r_teacher_route.astype(np.float32).copy()
            self.last_r_objective = r_objective.astype(np.float32).copy()
            self.last_r_progress += r_progress.astype(np.float32)
            self.last_r_task = (r_objective + r_progress).astype(np.float32).copy()
            self.reward = reward.astype(np.float32)

        else:
            raise ValueError(f"Unknown env_mode: {self.env_mode}")

        self.served_cus[go_to_cus] += 1
        self.route_served_cus_count[go_to_cus] += 1

        self.load[go_to_depot] = 0.0
        self.load[go_to_rs_or_cus] += self.demands[destination[go_to_rs_or_cus]]

        self.current_time[go_to_depot] = 0.0
        self.current_time[go_to_rs_or_cus] += self.travel_time[
            self.last, destination
        ][go_to_rs_or_cus]

        self.current_time[go_to_cus] = np.maximum(
            self.current_time[go_to_cus],
            self.time_window[destination[go_to_cus], 0],
        )

        self.current_time += self.service_time[destination]

        if np.any(go_to_rs):
            self.battery[go_to_rs] += self.edge_energy[self.last, destination][go_to_rs]
            charging_time_norm = self.battery[go_to_rs] * self.charging_beta
            self.current_time[go_to_rs] += charging_time_norm
            self.battery[go_to_rs] = 0.0

        self.battery[go_to_depot] = 0.0
        self.battery[go_to_cus] += self.edge_energy[self.last, destination][go_to_cus]

        self.visited[self.traj_idx, destination] = True
        self.node_visit_count[self.traj_idx, destination] += 1
        non_depot_idx = self.traj_idx[go_to_rs_or_cus]
        if non_depot_idx.size > 0:
            self.route_step_count[non_depot_idx] += 1
            self.route_visit_order[non_depot_idx, destination[go_to_rs_or_cus]] = (
                self.route_step_count[non_depot_idx]
            )
            for traj in non_depot_idx:
                event_node = int(destination[traj])
                event_step = int(self.route_step_count[traj])
                if event_step <= self.max_route_events:
                    event_pos = event_step - 1
                else:
                    self.route_event_nodes[traj, :-1] = self.route_event_nodes[traj, 1:]
                    self.route_event_step[traj, :-1] = self.route_event_step[traj, 1:]
                    self.route_event_visit_count[traj, :-1] = (
                        self.route_event_visit_count[traj, 1:]
                    )
                    self.route_event_mask[traj, :-1] = self.route_event_mask[traj, 1:]
                    event_pos = self.max_route_events - 1

                self.route_event_nodes[traj, event_pos] = event_node
                self.route_event_step[traj, event_pos] = event_step
                self.route_event_visit_count[traj, event_pos] = int(
                    self.node_visit_count[traj, event_node]
                )
                self.route_event_mask[traj, event_pos] = True
        self.prev_node = prev_last.astype(np.int32)
        self.last = destination.astype(np.int32)

        if np.any(go_to_depot):
            depot_idx = self.traj_idx[go_to_depot]
            self.route_visit_order[depot_idx, :] = 0
            self.route_visit_order[depot_idx, 0] = 1
            self.route_step_count[depot_idx] = 1
            self.route_event_nodes[depot_idx, :] = 0
            self.route_event_step[depot_idx, :] = 0
            self.route_event_visit_count[depot_idx, :] = 0
            self.route_event_mask[depot_idx, :] = False
            self.route_event_nodes[depot_idx, 0] = 0
            self.route_event_step[depot_idx, 0] = 1
            self.route_event_visit_count[depot_idx, 0] = self.node_visit_count[
                depot_idx,
                0,
            ]
            self.route_event_mask[depot_idx, 0] = True
            self.route_served_cus_count[go_to_depot] = 0
            self.route_distance[go_to_depot] = 0.0

    # =========================================================
    # State / mask
    # =========================================================
    def is_all_visited(self):
        return self.served_cus == self.cus_num

    def _update_state(self, update_mask=True):
        if update_mask:
            action_mask = self._update_mask()
        else:
            action_mask = self.mask

        remain_feasible_customers_ratio = (
            (
                (~self.visited[:, self.cus_start:self.rs_start])
                & action_mask[:, self.cus_start:self.rs_start]
            ).sum(axis=1, keepdims=True)
            / float(self.cus_num)
        ).astype(np.float32)

        obs = {
            "cus_loc": self.cus_nodes,
            "depot_loc": self.depot_nodes,
            "rs_loc": self.rs_nodes,
            "demand": self.demands,
            "time_window": self.time_window,
            "action_mask": action_mask,
            "last_node_idx": self.last,
            "prev_node_idx": self.prev_node,
            "node_visit_count": self.node_visit_count.astype(np.float32),
            "route_order_rank": (
                self.route_visit_order.astype(np.float32)
                / np.maximum(self.route_step_count[:, None].astype(np.float32), 1.0)
            ),
            "route_event_nodes": self.route_event_nodes.astype(np.int32),
            "route_event_mask": self.route_event_mask.copy(),
            "route_event_order_rank": (
                self.route_event_step.astype(np.float32)
                / np.maximum(self.route_step_count[:, None].astype(np.float32), 1.0)
            ),
            "route_event_visit_count": self.route_event_visit_count.astype(np.float32),
            "current_load": self.load,
            "current_battery": self.battery,
            "current_time": self.current_time,
            "service_time": self.service_time,
            "battery_capacity": self.battery_capacity_arr,
            "edge_energy": self.edge_energy,
            "loading_capacity": self.loading_capacity_arr,
            "visited_customers_ratio": (
                self.served_cus.astype(np.float32) / float(self.cus_num)
            )[:, None],
            "route_served_customers_ratio": (
                self.route_served_cus_count.astype(np.float32) / float(self.cus_num)
            )[:, None],
            "remain_feasible_customers_ratio": remain_feasible_customers_ratio,
            "rs_streak_ratio": (
                np.minimum(self.rs_streak.astype(np.float32), 5.0) / 5.0
            )[:, None],
        }

        return obs

    def _update_mask(self):
        self.mask = _update_mask_numba(
            self.visited,
            self.last,
            self.load,
            self.battery,
            self.current_time,
            self.served_cus,
            self.route_served_cus_count,
            self.demands,
            self.travel_time,
            self.edge_energy,
            self.tw_close_all,
            self.tw_open_cus,
            self.service_time_cus,
            self.time_cus_to_rs,
            self.energy_cus_to_rs,
            self.rs_cols,
            self.rs_time_to_depot_2d,
            np.float32(self.charging_beta),
            np.float32(self.loading_capacity),
            np.float32(self.battery_capacity),
            np.float32(self.instance_max_time),
            int(self.cus_num),
            int(self.cus_start),
            int(self.rs_start),
        )
        action_mask = self.mask
        return action_mask

    # =========================================================
    # Normalization
    # =========================================================
    def _normalizations(self, context):
        nodes_raw = np.concatenate(
            (
                context["depot"],
                context["customers"],
                context["charging_stations"],
            ),
            axis=0,
        ).astype(np.float32)

        positions = np.zeros_like(nodes_raw, dtype=np.float32)

        demands_abs = context["demands"].astype(np.float32)
        time_window_abs = context["tw"].astype(np.float32)
        service_time_abs = context["service_time"].astype(np.float32)

        data = context["env"]

        consumption_per_km = np.float32(data["consumption_per_distance"])
        battery_capacity_abs = np.float32(data["battery_capacity"])
        velocity_abs = np.float32(data["speed"])
        loading_capacity_abs = np.float32(data["loading_capacity"])
        charging_power_abs = np.float32(
            data["charging_speed"] * data["charging_efficiency"]
        )
        instance_max_time_abs = np.float32(data["instance_endTime"])
        pos_scale = data["area_size"]

        x_scale = np.float32(pos_scale[0][1] - pos_scale[0][0])
        y_scale = np.float32(pos_scale[1][1] - pos_scale[1][0])

        positions[:, 0] = (nodes_raw[:, 0] - pos_scale[0][0]) / x_scale
        positions[:, 1] = (nodes_raw[:, 1] - pos_scale[1][0]) / y_scale

        demands_norm_cus = demands_abs / loading_capacity_abs
        demands_norm_depot = np.zeros((1,), dtype=np.float32)
        demands_norm_rs = np.zeros((self.rs_num,), dtype=np.float32)

        t_scale = instance_max_time_abs

        time_window_norm_cus = np.clip(
            time_window_abs / t_scale,
            0.0,
            1.0,
        ).astype(np.float32)

        time_window_norm_depot = np.array([[0.0, 1.0]], dtype=np.float32)

        time_window_norm_rs = np.tile(
            np.array([[0.0, 1.0]], dtype=np.float32),
            (self.rs_num, 1),
        )

        service_time_norm_cus = np.clip(
            service_time_abs / t_scale,
            0.0,
            1.0,
        ).astype(np.float32)

        service_time_norm_depot = np.zeros((1,), dtype=np.float32)
        service_time_norm_rs = np.zeros((self.rs_num,), dtype=np.float32)

        dist_matrix_raw = gen_dist_matrix(nodes_raw, nodes_raw).astype(np.float32)
        travel_time_norm = (dist_matrix_raw / velocity_abs) / t_scale
        edge_energy_frac = dist_matrix_raw * consumption_per_km / battery_capacity_abs
        charging_beta = battery_capacity_abs / (charging_power_abs * t_scale)
        repair_dist_scale = float(
            data.get(
                "repair_dist_scale",
                np.hypot(float(x_scale), float(y_scale)),
            )
        )
        repair_dist_scale = max(repair_dist_scale, 1e-6)

        self.raw_speed = velocity_abs
        self.nodes_raw = nodes_raw
        self.nodes = positions.astype(np.float32)

        self.depot_nodes = self.nodes[0:1]
        self.cus_nodes = self.nodes[1:1 + self.cus_num]
        self.rs_nodes = self.nodes[1 + self.cus_num:1 + self.cus_num + self.rs_num]

        self.demands = np.concatenate(
            (demands_norm_depot, demands_norm_cus, demands_norm_rs),
            axis=0,
        ).astype(np.float32)

        self.time_window = np.concatenate(
            (time_window_norm_depot, time_window_norm_cus, time_window_norm_rs),
            axis=0,
        ).astype(np.float32)

        self.service_time = np.concatenate(
            (service_time_norm_depot, service_time_norm_cus, service_time_norm_rs),
            axis=0,
        ).astype(np.float32)

        self.travel_time = travel_time_norm.astype(np.float32)
        self.edge_energy = edge_energy_frac.astype(np.float32)
        self.charging_beta = np.float32(charging_beta)

        self.loading_capacity = np.float32(1.0)
        self.battery_capacity = np.float32(1.0)
        self.instance_max_time = np.float32(1.0)

        rs_time_to_depot_abs = context["env"]["cs_time_to_depot"]
        self.rs_time_to_depot = np.concatenate(
            [
                np.zeros((1,), dtype=np.float32),
                (rs_time_to_depot_abs / t_scale).astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        self.dist_matrix = gen_dist_matrix(self.nodes, self.nodes).astype(np.float32)

        self.cus_start = 1
        self.rs_start = 1 + self.cus_num
        self.n_nodes = 1 + self.cus_num + self.rs_num
        self.max_route_events = max(
            2,
            int(self.kwargs.get("max_route_events", 16)),
        )

        self.battery_capacity_arr = np.array([self.battery_capacity], dtype=np.float32)
        self.loading_capacity_arr = np.array([self.loading_capacity], dtype=np.float32)

        self.traj_idx = np.arange(self.n_traj, dtype=np.int32)
        self.cus_idx = np.arange(self.cus_start, self.rs_start, dtype=np.int32)
        self.rs_cols = np.concatenate(
            (
                np.array([0], dtype=np.int32),
                np.arange(self.rs_start, self.n_nodes, dtype=np.int32),
            ),
            axis=0,
        )

        single_repair_norm = data.get("single_customer_repair_dist_norm", None)
        if single_repair_norm is None:
            single_repair_abs = data.get("single_customer_repair_dist", None)
            if single_repair_abs is not None:
                single_repair_norm = (
                    np.asarray(single_repair_abs, dtype=np.float32)
                    / np.float32(repair_dist_scale)
                )
        if single_repair_norm is None:
            customer_cols = np.arange(self.cus_start, self.rs_start, dtype=np.int32)
            single_repair_norm = (
                2.0
                * dist_matrix_raw[0, customer_cols]
                / np.float32(repair_dist_scale)
            )
        single_repair_norm = np.asarray(single_repair_norm, dtype=np.float32).reshape(-1)
        if single_repair_norm.size != self.cus_num:
            customer_cols = np.arange(self.cus_start, self.rs_start, dtype=np.int32)
            single_repair_norm = (
                2.0
                * dist_matrix_raw[0, customer_cols]
                / np.float32(repair_dist_scale)
            ).astype(np.float32)
        self.single_customer_repair_dist_norm = single_repair_norm.astype(np.float32)
        self.total_customer_repair_dist_norm = float(
            np.maximum(self.single_customer_repair_dist_norm.sum(), np.float32(1e-6))
        )

        customer_back_norm = data.get("customer_dist_to_depot_norm", None)
        if customer_back_norm is None:
            customer_back_abs = data.get("customer_dist_to_depot", None)
            if customer_back_abs is not None:
                customer_back_norm = (
                    np.asarray(customer_back_abs, dtype=np.float32)
                    / np.float32(repair_dist_scale)
                )
        if customer_back_norm is None:
            customer_back_norm = (
                dist_matrix_raw[self.cus_start:self.rs_start, 0]
                / np.float32(repair_dist_scale)
            )
        customer_back_norm = np.asarray(customer_back_norm, dtype=np.float32).reshape(-1)
        if customer_back_norm.size != self.cus_num:
            customer_back_norm = (
                dist_matrix_raw[self.cus_start:self.rs_start, 0]
                / np.float32(repair_dist_scale)
            ).astype(np.float32)

        rs_back_norm = data.get("cs_dist_to_depot_norm", None)
        if rs_back_norm is None:
            rs_back_abs = data.get("cs_dist_to_depot", None)
            if rs_back_abs is not None:
                rs_back_norm = (
                    np.asarray(rs_back_abs, dtype=np.float32)
                    / np.float32(repair_dist_scale)
                )
        if rs_back_norm is None:
            rs_back_norm = (
                dist_matrix_raw[self.rs_start:self.n_nodes, 0]
                / np.float32(repair_dist_scale)
            )
        rs_back_norm = np.asarray(rs_back_norm, dtype=np.float32).reshape(-1)
        if rs_back_norm.size != self.rs_num:
            rs_back_norm = (
                dist_matrix_raw[self.rs_start:self.n_nodes, 0]
                / np.float32(repair_dist_scale)
            ).astype(np.float32)

        self.node_to_depot_repair_dist_norm = np.concatenate(
            (
                np.zeros((1,), dtype=np.float32),
                customer_back_norm.astype(np.float32),
                rs_back_norm.astype(np.float32),
            ),
            axis=0,
        ).astype(np.float32)

        self.tw_open_cus = self.time_window[
            self.cus_start:self.rs_start,
            0,
        ].astype(np.float32)

        self.tw_close_all = self.time_window[:, 1].astype(np.float32)

        self.service_time_cus = self.service_time[
            self.cus_start:self.rs_start
        ].astype(np.float32)

        self.time_cus_to_rs = self.travel_time[
            np.ix_(self.cus_idx, self.rs_cols)
        ].astype(np.float32)

        self.energy_cus_to_rs = self.edge_energy[
            np.ix_(self.cus_idx, self.rs_cols)
        ].astype(np.float32)

        self.rs_time_to_depot_2d = self.rs_time_to_depot.astype(np.float32)

        self.charge_factor_2d = np.ones((1, self.rs_num + 1), dtype=np.float32)
        self.charge_factor_2d[0, 0] = 0.0

    def _print_matrix(self, array, idx):
        for i in range(len(array)):
            print(array[i][idx], end="\t")
        print()
