# evrptw_gen/generator.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import copy
from tqdm import tqdm

from .configs.load_config import Config
from .policies.position import CustomerPositionPolicies
from .policies.time_window import TimeWindowPolicies
from .policies.service_time import ServiceTimePolicies
from .policies.demand import DemandPolicies
from .policies.perturb import Perturbation

from .utils.geometry import clamp
from .utils.feasibility import cs_min_time_to_depot, effective_charging_power_kw
from .utils.visualization import plot_instance, save_instances
from .utils.energy_consumption_model import consumption_model


class InstanceGenerator:
    def __init__(self, config_path: str, **kwargs):
        self.config= kwargs.get("config", None)
        if self.config is None:
            self.config = Config(config_path)

        self.save_path: Optional[str] = kwargs.get("save_path")
        self.num_instances: int = int(kwargs.get("num_instances", 100))
        raw_env: Dict = self.config.setup_env_parameters()

        # Persist RNG metadata.  A generator owns one seed stream for deriving
        # per-instance seeds, and one active RNG used inside the current
        # instance.  This makes each instance internally deterministic while
        # consecutive generated instances still differ.
        seed = kwargs.get("seed", raw_env.get("rng_seed", None))
        if seed is not None:
            seed = int(seed)
            raw_env["rng_seed"] = seed
        self.base_seed = seed
        self._instance_counter = 0
        self._seed_rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed)
        self._current_instance_seed = seed
        self.env = self._prepare_env(raw_env)

    def _add_perturb_env(self, env: Dict, perturb_dict: Dict, perturber, rng=None) -> Dict:
        # Add perturbation logic here
        update_value = perturber.perturb(env, perturb_keys=perturb_dict, rng=rng)

        for key, value in update_value.items():
            # print("Before Perturb:", key, env[key], "After Perturb:", value)
            env[key] = value
        return env

    def _update_seeds(self, seed):
        seed = None if seed is None else int(seed)
        self.base_seed = seed
        self._instance_counter = 0
        self._seed_rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed)
        self._current_instance_seed = seed

    def _next_instance_seed(self):
        idx = int(self._instance_counter)
        self._instance_counter += 1
        if self.base_seed is None:
            return int(self._seed_rng.integers(0, np.iinfo(np.uint32).max))
        seed_seq = np.random.SeedSequence([int(self.base_seed), idx])
        return int(seed_seq.generate_state(1, dtype=np.uint32)[0])

    def _begin_instance_rng(self, seed=None):
        if seed is None:
            seed = self._next_instance_seed()
        seed = int(seed)
        self._current_instance_seed = seed
        self.rng = np.random.default_rng(seed)
        return seed

    def _prepare_env(self, raw_env: Dict) -> Tuple[Dict, Dict]:
        """
        Parse raw config into a flat env dict and extract a perturbation spec.

        Returns
        -------
        env : Dict
        perturb_dict : Dict[str, Tuple[float, float]]
            e.g., {"num_customers_p": (-0.1, +0.2), ...}
        """
        not_copy_key = [
            "vehicles_profiles",
            "charging_profiles",
            "instance_time_range",
            "working_schedule_profiles",
        ]

        def time_to_minutes(hhmm: str) -> int:
            h, m = hhmm.split(":")
            return 60 * int(h) + int(m)

        env: Dict = {}
        for k, v in raw_env.items():
            if k not in not_copy_key:
                env[k] = v

        # Vehicles (assume homogeneous: index 0)
        vprof = raw_env["vehicles_profiles"][0]
        # Keep canonical names consistent downstream
        env["speed"] = float(vprof["speed"])  # km/h
        env["battery_capacity"] = float(vprof["battery_capacity"])  # kWh
        env["consumption_per_distance"] = float(vprof["consumption_per_distance"])  # kWh/km
        env["loading_capacity"] = float(vprof["loading_capacity"])

        # Charging (kW) and efficiency (default charging stations: DC_Fast_150kW)
        cprof = raw_env["charging_profiles"][1]
        env["charging_speed"] = float(cprof["power_kw"])  # kW
        env["charging_efficiency"] = float(cprof.get("efficiency", 1.0))
        # Optional vehicle AC limit (kW). If absent, assume no additional limit.
        if "vehicle_ac_limit_kw" in raw_env:
            env["vehicle_ac_limit_kw"] = float(raw_env["vehicle_ac_limit_kw"])

        # Time horizon and working window
        inst_start, inst_end = raw_env["instance_time_range"]

        ws_idx = 0
        work_start = raw_env["working_schedule_profiles"][ws_idx]["start"]
        work_end = raw_env["working_schedule_profiles"][ws_idx]["end"]
        env["instance_startTime"] = time_to_minutes(inst_start)
        env["instance_endTime"] = time_to_minutes(inst_end)
        env["working_startTime"] = time_to_minutes(work_start)
        env["working_endTime"] = time_to_minutes(work_end)

        # Persist RNG metadata (optional)
        if "rng_seed" in raw_env:
            env["rng_seed"] = raw_env["rng_seed"]

        return env

    def generate(self, perturb_dict=None, node_generater_scheduler=None, **kwargs) -> List[Dict]:
        if perturb_dict is None:
            perturb_dict = {}

        instances = []
        node_generate_policy = kwargs.get("node_generate_policy", "fixed")
        save_solomon = kwargs.get("save_template_solomon", False)
        save_pickle = kwargs.get("save_template_pickle", False)
        plot_instances = kwargs.get("plot_instances", False)

        if save_pickle:
            node_generate_policy = "fixed"

        perturber = Perturbation()

        # cache
        base_env = self.env
        add_perturb = self._add_perturb_env
        gen_one = self._generate_one_instance
        scheduler = node_generater_scheduler

        # scheduler
        if scheduler is None:
            raise ValueError("node_generater_scheduler is None, but generate() expects a callable scheduler.")

        num_cus, num_cs = scheduler(policy_name=node_generate_policy)
        base_env['num_customers'] = num_cus
        base_env['num_charging_stations'] = num_cs
            
        for id in tqdm(range(self.num_instances)):
            instance_seed = self._begin_instance_rng()

            # deep copy
            env = copy.deepcopy(base_env)
            env["rng_seed"] = instance_seed

            # add perturbations
            env = add_perturb(env, perturb_dict, perturber, rng=self.rng)
            inst = gen_one(env)
            inst['id'] = id
            inst['rng_seed'] = instance_seed
            instances.append(inst)

        # save
        if self.save_path:
            if save_solomon:
                save_instances(instances, self.save_path, template="solomon")
            if save_pickle:
                save_instances(instances, self.save_path, template="pickle")
            if plot_instances:
                instance_save_path = os.path.join(self.save_path, "plot_instances")
                plot_instance(instances, instance_save_path)
        return instances

    def generate_tensors(self, env = None, **kwargs):
        if env == None:
            env = self.env

        instance_seed = kwargs.get("rng_seed", None)
        instance_seed = self._begin_instance_rng(instance_seed)

        perturber = Perturbation()
        perturb_dict = kwargs.get("perturb_dict", {})
        env = copy.deepcopy(env)
        env["rng_seed"] = instance_seed
        env["num_customers"] = kwargs.get("num_customers", env['num_customers'])
        env["num_charging_stations"] = kwargs.get("num_charging_stations", env['num_charging_stations'])
        perturb_env = self._add_perturb_env(env, perturb_dict, perturber, rng=self.rng)
        perturb_env = copy.deepcopy(perturb_env)
        context = self._generate_one_instance(perturb_env)
        context["rng_seed"] = instance_seed
        return context

    def _generate_one_instance(self, env: Dict) -> Dict:
        # Select policies from env
        pos_policy = CustomerPositionPolicies.from_env(env, rng=self.rng)
        tw_policy  = TimeWindowPolicies.from_env(env, rng=self.rng)

        # time unit: hours
        service_time_policy = ServiceTimePolicies.from_env(env, rng=self.rng)
        demand_policy = DemandPolicies.from_env(env, rng=self.rng)

        depot_pos = self._get_depot_position(env)
        cs_pos, cs_time_to_depot, depot_time_to_cs = self._get_CSs_positions(env, depot_pos)
        # In case we may need for different policies.
        env['instance_type'] = pos_policy.NAME
        env['cs_time_to_depot'] = cs_time_to_depot
        env['time_window_type'] = tw_policy.NAME
        env['service_time_type'] = service_time_policy.NAME
        env['demand_type'] = demand_policy.NAME

        # vehicle info
        # self.velocity = float(instance["vehicle"]["v"])          # km/h
        # self.consume_rate = float(instance["vehicle"]["r"])      # kWh/km
        # self.charging_power = 1.0 / float(instance["vehicle"]["g"])  # g: inverse charging power (h/kWh); charging power: (kwh/h)
        # self.fuel_cap = float(instance["vehicle"]["Q"])          # kWh
        # self.load_cap = float(instance["vehicle"]["C"])         # tons

        cus_pos, service_time, t_earliest, t_latest, demand = pos_policy.sample(
            env, depot_pos, cs_pos, cs_time_to_depot, depot_time_to_cs, service_time_policy, rng=self.rng, demand_policy=demand_policy
        )
        if env['num_customers'] != cus_pos.shape[0]:
            raise ValueError("Inconsistent shapes in customer position generation.")

        # tensor use minutes as format
        tw = tw_policy.build(
        env, t_earliest, t_latest, service_time, rng=self.rng, tw_format = "minutes"
        )
        customers_pos = cus_pos
        demand = demand
        service_time = service_time * 60  # convert to minutes
        self._attach_repair_distance_metadata(env, depot_pos, cs_pos, customers_pos)

        return {"env": env, 
                "depot": depot_pos, 
                "customers": customers_pos, 
                "charging_stations": cs_pos, 
                "demands":demand, 
                "tw":tw,
                "service_time":service_time}

    def _stop_shortest_time_distance(
        self,
        env: Dict,
        depot_pos: np.ndarray,
        cs_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shortest-time paths on Depot+CS nodes, carrying path distance too."""
        stops = np.vstack((depot_pos.reshape(1, 2), cs_pos)).astype(float)
        num_stops = stops.shape[0]
        diff = stops[:, None, :] - stops[None, :, :]
        leg_dist = np.linalg.norm(diff, axis=-1)

        consumption_per_distance = consumption_model(env, model_type=None)
        battery_capacity = float(env["battery_capacity"])
        speed = float(env["speed"])
        p_eff = effective_charging_power_kw(env)
        max_leg = battery_capacity / max(float(consumption_per_distance), 1e-12)

        path_time = np.full((num_stops, num_stops), np.inf, dtype=float)
        path_dist = np.full((num_stops, num_stops), np.inf, dtype=float)
        np.fill_diagonal(path_time, 0.0)
        np.fill_diagonal(path_dist, 0.0)

        feasible = leg_dist <= max_leg + 1e-9
        for i in range(num_stops):
            for j in range(num_stops):
                if i == j or not feasible[i, j]:
                    continue
                travel_h = leg_dist[i, j] / max(speed, 1e-12)
                charge_h = 0.0
                if j != 0:
                    charge_h = leg_dist[i, j] * consumption_per_distance / max(p_eff, 1e-12)
                path_time[i, j] = travel_h + charge_h
                path_dist[i, j] = leg_dist[i, j]

        for k in range(num_stops):
            cand_time = path_time[:, k:k + 1] + path_time[k:k + 1, :]
            cand_dist = path_dist[:, k:k + 1] + path_dist[k:k + 1, :]
            improve = (
                (cand_time < path_time - 1e-9)
                | (
                    np.isclose(cand_time, path_time, atol=1e-9, rtol=0.0)
                    & (cand_dist < path_dist)
                )
            )
            path_time[improve] = cand_time[improve]
            path_dist[improve] = cand_dist[improve]

        return path_time, path_dist

    def _attach_repair_distance_metadata(
        self,
        env: Dict,
        depot_pos: np.ndarray,
        cs_pos: np.ndarray,
        cus_pos: np.ndarray,
    ) -> None:
        """Precompute a distance-scale repair proxy for terminal failure shaping."""
        (xmin, xmax), (ymin, ymax) = env["area_size"]
        repair_dist_scale = float(np.hypot(xmax - xmin, ymax - ymin))
        repair_dist_scale = max(repair_dist_scale, 1e-6)

        _, stop_dist = self._stop_shortest_time_distance(env, depot_pos, cs_pos)
        stops = np.vstack((depot_pos.reshape(1, 2), cs_pos)).astype(float)
        customers = np.asarray(cus_pos, dtype=float).reshape(-1, 2)

        consumption_per_distance = consumption_model(env, model_type=None)
        battery_capacity = float(env["battery_capacity"])
        stop_to_customer = np.linalg.norm(
            stops[:, None, :] - customers[None, :, :],
            axis=-1,
        )
        customer_to_stop = stop_to_customer.T

        single_customer_repair = np.full(customers.shape[0], np.inf, dtype=float)
        customer_to_depot = np.full(customers.shape[0], np.inf, dtype=float)

        for customer_idx in range(customers.shape[0]):
            best_service_dist = np.inf
            best_back_dist = np.inf
            for first_stop in range(stops.shape[0]):
                start_dist = stop_dist[0, first_stop]
                if not np.isfinite(start_dist):
                    continue
                d_to_customer = stop_to_customer[first_stop, customer_idx]
                for last_stop in range(stops.shape[0]):
                    back_to_depot = stop_dist[last_stop, 0]
                    if not np.isfinite(back_to_depot):
                        continue
                    d_from_customer = customer_to_stop[customer_idx, last_stop]
                    customer_leg_energy = (
                        d_to_customer + d_from_customer
                    ) * consumption_per_distance
                    if customer_leg_energy > battery_capacity + 1e-9:
                        continue
                    total_dist = (
                        start_dist
                        + d_to_customer
                        + d_from_customer
                        + back_to_depot
                    )
                    if total_dist < best_service_dist:
                        best_service_dist = total_dist
                    return_dist = d_from_customer + back_to_depot
                    if return_dist < best_back_dist:
                        best_back_dist = return_dist

            direct_dist = float(np.linalg.norm(customers[customer_idx] - depot_pos.reshape(2)))
            if not np.isfinite(best_service_dist):
                best_service_dist = 2.0 * direct_dist
            if not np.isfinite(best_back_dist):
                best_back_dist = direct_dist
            single_customer_repair[customer_idx] = best_service_dist
            customer_to_depot[customer_idx] = best_back_dist

        cs_dist_to_depot = stop_dist[1:, 0] if stop_dist.shape[0] > 1 else np.empty(0, dtype=float)
        depot_dist_to_cs = stop_dist[0, 1:] if stop_dist.shape[0] > 1 else np.empty(0, dtype=float)
        cs_direct = np.linalg.norm(cs_pos - depot_pos.reshape(1, 2), axis=1) if cs_pos.size else np.empty(0, dtype=float)
        cs_dist_to_depot = np.where(np.isfinite(cs_dist_to_depot), cs_dist_to_depot, cs_direct)
        depot_dist_to_cs = np.where(np.isfinite(depot_dist_to_cs), depot_dist_to_cs, cs_direct)

        env["repair_dist_scale"] = repair_dist_scale
        env["single_customer_repair_dist"] = single_customer_repair
        env["single_customer_repair_dist_norm"] = (
            single_customer_repair / repair_dist_scale
        )
        env["customer_dist_to_depot"] = customer_to_depot
        env["customer_dist_to_depot_norm"] = customer_to_depot / repair_dist_scale
        env["cs_dist_to_depot"] = cs_dist_to_depot
        env["cs_dist_to_depot_norm"] = cs_dist_to_depot / repair_dist_scale
        env["depot_dist_to_cs"] = depot_dist_to_cs
        env["depot_dist_to_cs_norm"] = depot_dist_to_cs / repair_dist_scale

    def _get_depot_position(self, env: Dict) -> np.ndarray:
        """
        Uniformly sample a depot position within the valid instance area.

        env['area_size'] must be ((x_min, x_max), (y_min, y_max))
        Returns a (1, 2) array rounded to 2 decimals.
        """
        (xmin, xmax), (ymin, ymax) = env["area_size"]
        x = self.rng.uniform(xmin, xmax)
        y = self.rng.uniform(ymin, ymax)
        return np.round(np.array([[x, y]], dtype=float), 3)

    def _get_CSs_positions(self, env: Dict, depot_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample charging station (CS) positions within the full-charge reach region,
        and compute each CS's minimal travel time to the depot under the
        full-charging policy.

        Returns
        -------
        cs_positions : (N, 2) float array
        cs_time_to_depot : (N,) float array of hours
        """
        (xmin, xmax), (ymin, ymax) = env["area_size"]

        # Max distance on a full charge (km): battery(kWh) / consumption(kWh/km)
        consumption_per_distance = consumption_model(env, model_type = None)
        radius_cs = float(env['battery_capacity']) / consumption_per_distance

        speed = float(env['speed'])  # km/h
        p_eff = effective_charging_power_kw(env)  # kW (kWh/h)

        # Initial sampling box centered at the depot and clamped to area bounds
        cx, cy = float(depot_pos[0, 0]), float(depot_pos[0, 1])
        sxmin, sxmax = clamp(cx - radius_cs, xmin, xmax), clamp(cx + radius_cs, xmin, xmax)
        symin, symax = clamp(cy - radius_cs, ymin, ymax), clamp(cy + radius_cs, ymin, ymax)

        num_cs = int(env['num_charging_stations'])
        cs_positions: List[np.ndarray] = []
        cs_time_to_depot: List[float] = []
        depot_time_to_cs: List[float] = []

        max_trials = env.get('max_trials_per_cs', 5000)  # hard cap to avoid infinite loops
        
        trials = 0
        while len(cs_positions) < num_cs and trials < max_trials:
            trials += 1
            x = self.rng.uniform(sxmin, sxmax)
            y = self.rng.uniform(symin, symax)
            cand = np.array([round(x, 2), round(y, 2)], dtype=float)

            output = cs_min_time_to_depot(
                env=env,
                depot_pos=depot_pos[0],          # (2,)
                candidate_cs_pos=cand,           # (2,)
                cs_positions=cs_positions,       # list of (2,)
                cs_time_to_depot=cs_time_to_depot,
                depot_time_to_cs=depot_time_to_cs,
                radius=radius_cs,
                speed=speed,
                p_eff=p_eff,
                use_cs_range=False
            )

            feasible, time_candidate_depot, time_depot_candidate = output

            if feasible:
                cs_positions.append(cand)
                cs_time_to_depot.append(time_candidate_depot)
                depot_time_to_cs.append(time_depot_candidate)

                # Optionally re-center local sampling window around the last accepted CS
                x0, y0 = cand
                sxmin = clamp(x0 - radius_cs, xmin, xmax)
                sxmax = clamp(x0 + radius_cs, xmin, xmax)
                symin = clamp(y0 - radius_cs, ymin, ymax)
                symax = clamp(y0 + radius_cs, ymin, ymax)

        return (
                np.asarray(cs_positions, dtype=float),
                np.asarray(cs_time_to_depot, dtype=float),
                np.asarray(depot_time_to_cs, dtype=float),
                )
