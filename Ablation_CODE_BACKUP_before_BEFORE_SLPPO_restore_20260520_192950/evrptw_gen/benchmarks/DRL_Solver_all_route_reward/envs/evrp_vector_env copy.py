import gym
import numpy as np
import random
from gym import spaces
import time

from evrptw_gen.generator import InstanceGenerator
from evrptw_gen.benchmarks.DRL_Solver.envs.evrptw_data import EVRPTWDataset

import numpy as np
from numba import njit

@njit(cache=True)
def _ffp_customer_numba(
    base_cus_mask,        # (n_traj, n_cus) bool
    battery,              # (n_traj,)
    current_time,         # (n_traj,)
    time_to_customer,     # (n_traj, n_cus)
    energy_to_customer,   # (n_traj, n_cus)
    tw_open_cus,          # (n_cus,)
    service_time_cus,     # (n_cus,)
    time_cus_to_rs,       # (n_cus, n_rs+1)
    energy_cus_to_rs,     # (n_cus, n_rs+1)
    charging_beta,        # scalar
    RS_time_to_depot,     # (n_rs+1,)
    battery_capacity,     # scalar
    instance_max_time     # scalar
):
    n_traj, n_cus = base_cus_mask.shape
    n_rs1 = RS_time_to_depot.shape[0]

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

                # r == 0 means depot, no charging time
                if r != 0:
                    finish_time += battery_at_rs * charging_beta

                finish_time += RS_time_to_depot[r]

                if finish_time <= instance_max_time:
                    feasible = True
                    break

            out[t, c] = feasible

    return out

def assign_env_config(self, kwargs):
    """
    Set self.key = value for each key in kwargs.
    """
    for key, value in kwargs.items():
        setattr(self, key, value)

def gen_dist_matrix(nodes1: np.ndarray, nodes2: np.ndarray) -> np.ndarray:
    # float32 + contiguous faster
    x = np.ascontiguousarray(nodes1, dtype=np.float32)
    y = np.ascontiguousarray(nodes2, dtype=np.float32)

    x2 = np.sum(x * x, axis=1, keepdims=True)          # (N1, 1)
    y2 = np.sum(y * y, axis=1, keepdims=True).T        # (1, N2)
    # clip
    d2 = np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)     # (N1, N2)
    return np.sqrt(d2, out=d2).astype(np.float32, copy=False)

class EVRPTWVectorEnv(gym.Env):
    """
    Vectorized EVRPTW environment with fully normalized internal dynamics.

    Internal conventions:
        - Time is normalized by T_scale = instance_max_time_abs  => [0, 1].
        - Load is normalized by loading_capacity => in [0, 1].
        - Battery is represented as "consumed SoC fraction" in [0, 1]:
              0.0 => just fully charged (no consumption yet)
              1.0 => completely exhausted
        - edge_energy[i, j] = SoC fraction consumed when traveling i -> j.
        - charging_beta = normalized time needed to charge 1.0 SoC.
    """

    metadata = {"render.modes": []}

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.phi_reward = True
        self.lag_reward = True

        # ====== Configuration ======
        self.terminate = False
        self.config_path = kwargs.get("config_path", None)
        self.config = kwargs.get("config", None)
        self.n_traj = kwargs.get("n_traj", 100)
        self.Q_ratio = kwargs.get("Q_ratio", 1.0)  # battery capacity ratio
        self.r_ratio = kwargs.get("r_ratio", 1.0)  # charging rate ratio

        # ====== Load dataset / config ======
        save_path = kwargs.get("save_path", None)
        num_instances = kwargs.get("num_instances", 1)
        plot_instances = kwargs.get("plot_instances", False)

        self.dataset = InstanceGenerator(
            self.config_path,
            save_path=save_path,
            num_instances=num_instances,
            plot_instances=plot_instances,
            config = self.config,
            kwargs=kwargs,
        )

        self.cus_num = self.kwargs.get("num_customers", None)
        self.rs_num = self.kwargs.get("num_charging_stations", None)
        self.perturb_dict = self.kwargs.get("perturb_dict", {})
        
        self.env_mode = kwargs.get("env_mode", "train")  # "train" or "eval"   
        if self.env_mode == "train" and (self.cus_num is None or self.rs_num is None):
            raise ValueError("In 'train' mode, num_customers and num_charging_stations must be specified!")

        assign_env_config(self, kwargs)
        self.gamma = kwargs.get("gamma", 0.99)
        self.alpha = kwargs.get("alpha", 5.0)
        self.beta  = kwargs.get("beta", 0.5)
        self.lambda_fail = kwargs.get("lambda_fail", 5.0)
        self.success_bonus = kwargs.get("success_bonus", 10.00)
        # self.snap_shot = {}

        # ====== Observation / Action spaces ======
        if self.env_mode == 'train':
            self._observation_update()
        self.reset()

    def _observation_update(self):
        obs_dict = {
            "cus_loc": spaces.Box(low=0, high=1, shape=(self.cus_num, 2)),
            "depot_loc": spaces.Box(low=0, high=1, shape=(1, 2)),
            "rs_loc": spaces.Box(low=0, high=1, shape=(self.rs_num, 2)),
            "demand": spaces.Box(
                low=0, high=1, shape=(1 + self.cus_num + self.rs_num,)
            ),
            "time_window": spaces.Box(
                low=0, high=1, shape=(1 + self.cus_num + self.rs_num, 2)
            ),
            "action_mask": spaces.MultiBinary(
                [self.n_traj, self.cus_num + self.rs_num + 1]
            ),  # 1: OK, 0: cannot go
            "last_node_idx": spaces.MultiDiscrete(
                [self.cus_num + self.rs_num + 1] * self.n_traj
            ),
            "current_load": spaces.Box(low=0, high=1, shape=(self.n_traj,)),
            "current_battery": spaces.Box(low=0, high=1, shape=(self.n_traj,)),
            "current_time": spaces.Box(low=0, high=1, shape=(self.n_traj,)),
            "service_time": spaces.Box(
                low=0, high=1, shape=(1 + self.cus_num + self.rs_num,)
            ),
            # capacity scalars are kept for info / logging (can be 1.0 in normalized world)
            "battery_capacity": spaces.Box(low=0, high=np.inf, shape=(1,)),
            "loading_capacity": spaces.Box(low=0, high=np.inf, shape=(1,)),
            "visited_customers_raio": spaces.Box(low=0, high=1, shape=(self.n_traj, 1)),
            "remain_feasible_customers_raio": spaces.Box(low=0, high=1, shape=(self.n_traj, 1)),
        }

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete(
            [self.rs_num + self.cus_num + 1] * self.n_traj
        )
        self.reward_space = None

    # ======================================================================
    #  Gym API
    # ======================================================================

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.dataset._update_seeds(seed)

    def reset(self):
        self.num_steps = 0
        # self.traj = []
        self.info = {}

        if self.env_mode == "eval":
            self._eval_data_generate()
        elif self.env_mode == "train":
            self._train_data_generate()
        else:
            raise ValueError(f"Unknown mode: {self.env_mode}")

        if (not hasattr(self, "visited")) or (self.visited.shape != (self.n_traj, self.n_nodes)):
            self.visited = np.empty((self.n_traj, self.n_nodes), dtype=np.bool_)
            self.last = np.empty(self.n_traj, dtype=np.int32)
            self.load = np.empty(self.n_traj, dtype=np.float32)
            self.battery = np.empty(self.n_traj, dtype=np.float32)
            self.current_time = np.empty(self.n_traj, dtype=np.float32)
            self.done = np.empty(self.n_traj, dtype=np.bool_)
            self.finish = np.empty(self.n_traj, dtype=np.bool_)
            self.served_cus = np.empty(self.n_traj, dtype=np.int32)

        self.visited.fill(False)
        self.visited[:, 0] = True
        self.last.fill(0)
        self.load.fill(0.0)
        self.battery.fill(0.0)
        self.current_time.fill(0.0)
        self.served_cus.fill(0)
        self.done.fill(False)
        self.finish.fill(False)

        self.state = self._update_state()
        return self.state


    def step(self, action):
        """
        One vectorized step.

        Args:
            action: np.ndarray of shape (n_traj,) with node indices.

        Returns:
            obs, reward, done, info
        """
        return self._STEP(action)

    # ======================================================================
    #  Internal step / reset helpers
    # ======================================================================

    def _STEP(self, action):
        self._go_to(action)
        self.num_steps += 1
        self.state = self._update_state()

        all_visited = self.is_all_visited()
        if self.terminate:
            self.done = np.ones_like(self.done, dtype=np.bool_)
        else:
            self.done = (action == 0) & all_visited

        if self.env_mode == "train":
            new_done = self.done & (~self.finish)
            if self.terminate:
                success = all_visited & (self.last == 0)
            else:
                success = self.done

            r_lag = np.zeros_like(self.reward, dtype=np.float32)
            if self.lag_reward:
                r_lag[new_done & success] += self.success_bonus

                fail_mask = new_done & (~success)
                if np.any(fail_mask):
                    served_ratio = np.float32(self.served_cus / self.cus_num)
                    unserved_ratio = 1.0 - served_ratio
                    r_lag[fail_mask] -= self.lambda_fail * unserved_ratio[fail_mask]

            self.reward = self.reward + r_lag

        self.finish = self.finish | self.done
        if self.terminate:
            self.terminate = False
        return self.state, self.reward, self.done, self.info


    def is_all_visited(self):
        """
        Check if all customers (not depot, not RSs) have been visited.
        Node order: [0: depot, 1..cus_num: customers, cus_num+1..: RSs]
        """
        return (self.served_cus == self.cus_num)

    def _update_state(self, update_mask=True):
        """
        Build current observation dict from internal normalized state.
        """
        obs = {
            "cus_loc": self.cus_nodes,      # (n_cus, 2)
            "depot_loc": self.depot_nodes,  # (1, 2)
            "rs_loc": self.rs_nodes,        # (n_rs, 2)
            "demand": self.demands,         # (1+n_cus+n_rs,)
            "time_window": self.time_window,
            "action_mask": self._update_mask() if update_mask else self.mask,
            "last_node_idx": self.last,
            "current_load": self.load,
            "current_battery": self.battery,
            "current_time": self.current_time,
            "service_time": self.service_time,
            "battery_capacity": self.battery_capacity_arr,
            "loading_capacity": self.loading_capacity_arr,
            "visited_customers_raio": (self.served_cus/self.cus_num)[:, None],
            "remain_feasible_customers_raio": ((~self.visited[:, 1:1+self.cus_num]) & (self.mask[:, 1:1+self.cus_num])).sum(axis=1, keepdims=True)/self.cus_num,
        }
        return obs

    # ======================================================================
    #  Mask (feasibility + FFP)
    # ======================================================================
    def _update_mask(self):
        """
        Compute action mask based on:
            (1) visited customers
            (2) load capacity
            (3) battery capacity
            (4) time window feasibility
            (5) Future Feasibility Pruning (FFP)
        All checks are in normalized units.
        """
        # False: cannot go, True: can go
        # base: cannot revisit nodes marked visited
        action_mask = ~self.visited

        # Depot & RSs can always be visited
        action_mask[:, 0] = True
        action_mask[:, self.rs_start:] = True

        # cannot visit itself
        action_mask[self.traj_idx, self.last] = False

        # (2) load feasibility: load + demand <= capacity (1.0)
        load_mask = (self.load[:, None] + self.demands[None, :]) <= self.loading_capacity
        action_mask &= load_mask

        # (3) battery feasibility: current consumption + edge consumption <= 1.0
        battery_need = self.battery[:, None] + self.edge_energy[self.last, :].astype(np.float32)
        battery_mask = battery_need <= self.battery_capacity
        action_mask &= battery_mask

        # (4) time window (arrival <= close)
        time_after_arrival = self.current_time[:, None] + self.travel_time[self.last, :]
        time_mask = time_after_arrival <= self.tw_close_all[None, :]

        action_mask &= time_mask

        # (5) Future Feasibility Pruning (FFP): from current -> candidate customer -> some RS/depot -> depot
        cus_idx = self.cus_idx
        rs_cols = self.rs_cols

        # 1) current -> customer
        time_to_customer = self.travel_time[self.last[:, None], cus_idx]         # (n_traj, n_cus)
        energy_to_customer = self.edge_energy[self.last[:, None], cus_idx]      # (n_traj, n_cus)
        battery_at_customer = self.battery[:, None] + energy_to_customer        # (n_traj, n_cus)

        # start service time: max(arrival, TW open)
        tw_open = self.tw_open_cus[None, :]
        arrival_time = self.current_time[:, None] + time_to_customer           # (n_traj, n_cus)
        start_service_time = np.maximum(arrival_time, tw_open)                 # (n_traj, n_cus)
        time_after_service = start_service_time + self.service_time_cus[None, :]

        # 2) customer -> RS/depot
        time_cust_to_rs = self.time_cus_to_rs[None, :, :]
        energy_cust_to_rs = self.energy_cus_to_rs[None, :, :]

        battery_at_customer_3d = battery_at_customer[:, :, None]   # (n_traj, n_cus, 1)
        time_after_service_3d = time_after_service[:, :, None]     # (n_traj, n_cus, 1)

        battery_at_rs = battery_at_customer_3d + energy_cust_to_rs  # (n_traj, n_cus, n_rs+1), consumed SoC
        time_at_rs = time_after_service_3d + time_cust_to_rs        # (n_traj, n_cus, n_rs+1)

        # 3) charging to full then RS/depot -> depot
        # remaining SoC at RS = battery_at_rs
        time_charge_at_rs = battery_at_rs * self.charging_beta
        time_charge_at_rs[:, :, 0] = 0.0  # depot no charging time

        RS_time_to_depot_3d = self.RS_time_to_depot[None, None, :]  # (1, 1, n_rs+1), normalized

        total_finish_time = time_at_rs + time_charge_at_rs + RS_time_to_depot_3d  # normalized
        time_feasible = total_finish_time <= self.instance_max_time              # <= 1.0
        battery_feasible = battery_at_rs <= self.battery_capacity                # consumed SoC <= 1.0

        feasible = time_feasible & battery_feasible
        FFP_cus_mask = feasible.any(axis=2)  # (n_traj, n_cus)

        # only apply FFP on customers
        action_mask[:, self.cus_start:self.rs_start] &= FFP_cus_mask

        # (6) FFP on Charging Stations
        # if time Cur Node -> RS -> depot is infeasible, mask RS
        # (self.last, self.battery, self.current_time)
        time_to_rs = self.travel_time[self.last[:, None], rs_cols]         # (n_traj, n_rs+1)
        energy_to_rs = self.edge_energy[self.last[:, None], rs_cols]      # (n_traj, n_rs+1)
        battery_at_rs = self.battery[:, None] + energy_to_rs        # (n_traj, n_rs+1)
        time_at_rs = self.current_time[:, None] + time_to_rs        # (n_traj, n_rs+1)

        # charging to full
        time_charge_at_rs = battery_at_rs * self.charging_beta
        time_charge_at_rs[:, 0] = 0.0  # depot no charging time
        RS_time_to_depot_2d = self.RS_time_to_depot_2d[None, :]
        total_finish_time_rs = time_at_rs + time_charge_at_rs + RS_time_to_depot_2d  # normalized
        rs_time_feasible = total_finish_time_rs <= self.instance_max_time              # <= 1.0
        rs_battery_feasible = battery_at_rs <= self.battery_capacity
        rs_feasible = rs_time_feasible & rs_battery_feasible
        action_mask[:, rs_cols] &= rs_feasible

        # customer visited mission complete
        customer_has_been_visited = self.is_all_visited()
        action_mask[customer_has_been_visited, 0] = True

        # mission complete: cannot go to any other nodes except stay at depot
        customer_has_been_visited_and_at_depot = customer_has_been_visited & (self.last == 0)
        action_mask[customer_has_been_visited_and_at_depot, 1:] = False

        self.mask = action_mask
        return action_mask

    def _print_matrix(self, array, idx):
        for i in range(len(array)):
            print(array[i][idx], end="\t")
            
    # ======================================================================
    #  Data generation / normalization
    # ======================================================================

    def _train_data_generate(self):
        """
        Generate one training instance and normalize all static data
        into the internal normalized representation.
        """
        context = self.dataset.generate_tensors(perturb_dict=self.perturb_dict,
                                                num_customers=self.cus_num,
                                                num_charging_stations=self.rs_num,
                                                format = "tensor")
        self.context = context
        self.depot_num = context["depot"].shape[0]
        self.cus_num = context["customers"].shape[0]
        self.rs_num = context["charging_stations"].shape[0]

        self._normalizations(context)

    def _eval_data_generate(self, mode=None, **kwargs):
        eval_data = self.kwargs.get("eval_data", None)
        if eval_data is None:
            raise ValueError("Please provide 'eval_data' for solomon eval mode!")
        context = eval_data

        context["env"]["battery_capacity"] *= self.Q_ratio
        context["env"]["charging_speed"] *= self.r_ratio    

        self.context = context
        self.depot_num = context["depot"].shape[0]
        self.cus_num = context["customers"].shape[0]
        self.rs_num = context["charging_stations"].shape[0]

        self._normalizations(context)
        self._observation_update()

    def _normalizations(self, context):
        """
        Normalize all static quantities:
            - node positions -> [0,1]^2
            - demand -> fraction of loading_capacity
            - time window / service time -> fraction of instance_max_time
            - travel_time -> fraction of instance_max_time
            - edge_energy -> fraction of battery capacity (SoC drop)
            - RS_time_to_depot -> fraction of instance_max_time
        """
        # ----- Raw node coordinates -----
        nodes_raw = np.concatenate(
            (context["depot"], context["customers"], context["charging_stations"])
        ).astype(np.float32)
        positions = np.zeros_like(nodes_raw, dtype=np.float32)

        # ----- Demand & time-related raw data -----
        demands_abs = context["demands"].astype(np.float32)       # (n_cus,)
        time_window_abs = context["tw"].astype(np.float32)        # (n_cus, 2)
        service_time_abs = context["service_time"].astype(np.float32)  # (n_cus,)

        data = context["env"]
        consumption_per_km = data["consumption_per_distance"]     # kWh / km
        b_s = data["battery_capacity"]                            # kWh, E_max
        velocity_abs = data["speed"]                              # km / hour
        loading_capacity = data["loading_capacity"]               # Q_max
        charging_power_abs = (
            data["charging_speed"] * data["charging_efficiency"]
        )  # kW
        instance_max_time_abs = data["instance_endTime"]          # T_scale
        pos_scale = data["area_size"]

        # --------- 1. Position normalization (for obs only) ---------
        x_scale = pos_scale[0][1] - pos_scale[0][0]
        y_scale = pos_scale[1][1] - pos_scale[1][0]
        positions[:, 0] = (nodes_raw[:, 0] - pos_scale[0][0]) / x_scale
        positions[:, 1] = (nodes_raw[:, 1] - pos_scale[1][0]) / y_scale

        # --------- 2. Demand normalization ---------
        demands_norm_cus = demands_abs / loading_capacity
        demands_norm_depot = np.zeros((1,), dtype=np.float32)
        demands_norm_rs = np.zeros((self.rs_num,), dtype=np.float32)

        # --------- 3. Time normalization (T_scale = instance_max_time_abs) ---------
        T_scale = instance_max_time_abs

        time_window_norm_cus = np.clip(time_window_abs / T_scale, 0.0, 1.0)
        time_window_norm_depot = np.array([[0.0, 1.0]], dtype=np.float32)
        time_window_norm_rs = np.array(
            [[0.0, 1.0] * self.rs_num], dtype=np.float32
        ).reshape(self.rs_num, 2)

        service_time_norm_cus = np.clip(service_time_abs / T_scale, 0.0, 1.0)
        service_time_norm_depot = np.zeros((1,), dtype=np.float32)
        service_time_norm_rs = np.zeros((self.rs_num,), dtype=np.float32)

        # --------- 4. Raw distance matrix (for time & energy) ---------
        dist_matrix_raw = gen_dist_matrix(nodes_raw, nodes_raw).astype(np.float32)  # km

        # travel time in normalized units
        travel_time_norm = (dist_matrix_raw / velocity_abs) / T_scale

        # --------- 5. Edge energy consumption (normalized SoC) ---------
        # each edge consumes: dist * (kWh/km) / (kWh)
        edge_energy_frac = dist_matrix_raw * consumption_per_km / b_s

        # --------- 6. Charging: time to charge 1.0 SoC (normalized) ---------
        charging_beta = b_s / (charging_power_abs * T_scale)  # E_max / (P_chg * T_scale)

        # --------- 7. Update env internal static state ---------
        self.raw_speed = velocity_abs
        self.nodes_raw = nodes_raw
        self.nodes = positions
        self.depot_nodes = positions[0:1]
        self.cus_nodes = positions[1 : 1 + self.cus_num]
        self.rs_nodes = positions[1 + self.cus_num : 1 + self.cus_num + self.rs_num]

        self.demands = np.concatenate(
            (demands_norm_depot, demands_norm_cus, demands_norm_rs)
        )
        self.time_window = np.concatenate(
            (time_window_norm_depot, time_window_norm_cus, time_window_norm_rs)
        )
        self.service_time = np.concatenate(
            (service_time_norm_depot, service_time_norm_cus, service_time_norm_rs)
        )

        # normalized travel time & energy
        self.travel_time = travel_time_norm
        self.edge_energy = edge_energy_frac
        self.charging_beta = charging_beta

        # in normalized world, capacities are 1.0
        self.loading_capacity = 1.0
        self.battery_capacity = 1.0
        self.instance_max_time = 1.0  # all TW & current_time in [0,1]

        # RS -> depot time in normalized units (prepend depot itself with 0)
        RS_time_to_depot_abs = context["env"]["cs_time_to_depot"]  # (n_rs,)
        self.RS_time_to_depot = np.concatenate(
            [
                np.zeros((1,), dtype=np.float32),
                (RS_time_to_depot_abs / T_scale).astype(np.float32),
            ]
        )

        # distance in normalized coordinate space (only for reward)
        self.dist_matrix = gen_dist_matrix(self.nodes, self.nodes).astype(np.float32)

        # Auxilliary Function
        self.cus_start = 1
        self.rs_start  = 1 + self.cus_num
        self.n_nodes   = 1 + self.cus_num + self.rs_num

        self.battery_capacity_arr = np.array([self.battery_capacity], dtype=np.float32)
        self.loading_capacity_arr = np.array([self.loading_capacity], dtype=np.float32)

        self.traj_idx = np.arange(self.n_traj, dtype=np.int32)
        self.cus_idx  = np.arange(self.cus_start, self.rs_start)
        self.rs_cols  = np.concatenate(([0], np.arange(self.rs_start, self.n_nodes)))

        # cache
        self.tw_open_cus = self.time_window[self.cus_start:self.rs_start, 0].astype(np.float32)
        self.tw_close_all = self.time_window[:, 1].astype(np.float32)
        self.service_time_cus = self.service_time[self.cus_start:self.rs_start].astype(np.float32)

        # customer -> (depot+RS)
        self.time_cus_to_rs = self.travel_time[np.ix_(self.cus_idx, self.rs_cols)].astype(np.float32)    # (n_cus, n_rs+1)
        self.energy_cus_to_rs = self.edge_energy[np.ix_(self.cus_idx, self.rs_cols)].astype(np.float32)  # (n_cus, n_rs+1)

        # RS->depot
        self.RS_time_to_depot_2d = self.RS_time_to_depot.astype(np.float32)  # (n_rs+1,)


    # ======================================================================
    #  Transition dynamics (in normalized space)
    # ======================================================================

    def _go_to(self, destination):
        """
        Transition for n_traj parallel vehicles going to 'destination'.

        Args:
            destination: np.ndarray of shape (n_traj,), each entry in [0, n_nodes-1]
        """
        # reward uses normalized Euclidean distance in [0, ~1.4]
        dist_display = self.dist_matrix[self.last, destination]   # shape (n_traj,)

        go_to_depot = destination == 0
        go_to_rs = destination >= self.rs_start
        go_to_cus = (destination >= self.cus_start) & (destination < self.rs_start)
        go_to_rs_or_cus = ~go_to_depot

        # -------- Reward update --------
        # You can rescale this later (e.g., multiply by pos_scale)
        if self.env_mode == "eval":
            self.reward = -dist_display
        elif self.env_mode == "train":
            # 1) objevtive: negative travel distance
            reward = -dist_display.astype(np.float32)

            # 2) potential-based shaping: Phi(s) = alpha * N_served(s)
            if self.phi_reward:
                prev_served_cus = self.served_cus
                next_served_cus = prev_served_cus + go_to_cus.astype(np.float32)

                phi_s  = -((self.cus_num - prev_served_cus) / self.cus_num)**self.beta
                phi_sp = -((self.cus_num - next_served_cus) / self.cus_num)**self.beta

                PBRS_reward = self.alpha * (self.gamma * phi_sp - phi_s)
                PBRS_reward.clip(-0.2, 0.2, out=PBRS_reward)
                # gamma*Phi(s') - Phi(s)
                reward += PBRS_reward

            self.reward = reward

        else:
            raise ValueError(f"Unknown Mode: {self.env_mode}")

        # -------- Load update (normalized) --------
        self.served_cus[go_to_cus] += 1

        # going to depot: unload all
        self.load[go_to_depot] = 0.0
        # going to customers/RS: add demand (RS demand is 0)
        self.load[go_to_rs_or_cus] += self.demands[destination[go_to_rs_or_cus]]

        # -------- Time update (normalized) --------
        self.current_time[go_to_depot] = 0.0  # at depot, time reset to 0
        # 1) travel time
        self.current_time[go_to_rs_or_cus] += self.travel_time[
            self.last, destination
        ][go_to_rs_or_cus]

        # 2) waiting for customer time window open
        self.current_time[go_to_cus] = np.maximum(
            self.current_time[go_to_cus],
            self.time_window[destination[go_to_cus], 0],
        )

        # 3) service time
        self.current_time += self.service_time[destination]

        # 4) charging at RS (charge to full)
        if go_to_rs.any():
            # battery = consumed SoC after arriving at RS
            self.battery[go_to_rs] += self.edge_energy[self.last, destination][go_to_rs]
            charging_time_norm = self.battery[go_to_rs] * self.charging_beta
            self.current_time[go_to_rs] += charging_time_norm
            # after charging, consider "consumed SoC" reset to 0
            self.battery[go_to_rs] = 0.0

        # -------- Battery update (normalized consumed SoC) --------
        # at depot we assume fully charged, so consumed = 0
        self.battery[go_to_depot] = 0.0

        # already handled RS above (set to 0)
        # here we only add travel consumption when going to customers
        self.battery[go_to_cus] += self.edge_energy[self.last, destination][go_to_cus]

        # -------- Visited & last node update --------
        self.visited[self.traj_idx, destination] = True
        self.last = destination

