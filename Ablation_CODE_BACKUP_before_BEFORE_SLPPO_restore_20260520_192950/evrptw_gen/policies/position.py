# evrptw_gen/policies/positions.py
from __future__ import annotations
from typing import Dict, Tuple, Protocol
import numpy as np
from evrptw_gen.utils.geometry import clamp, clamp_rect
from evrptw_gen.utils.feasibility import effective_charging_power_kw, cus_min_time_to_depot
from evrptw_gen.utils.energy_consumption_model import consumption_model

from .rc_customer_assign import RCPolicies
from .cluster_assignment import ClusterAssignmentPolicies
from .cluster_number import ClusterNumberPolicies

def dist_vec(a, b):
    """Vectorized distance between two sets of points a and b.
    a: (N,D), b: (M,D) -> returns (N,M) array of pairwise distances.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]  # (N,M,D)
    return np.linalg.norm(diff, axis=-1)  # (N,M)

def _process_serve_time(env, time_customer_depot, time_depot_customer, service_time):
        """
        Compute the feasible service window [earliest, latest] for each customer
        given travel times and global time constraints.

        Inputs
        -------
        env : Mapping
            Must provide minutes-based times:
            - 'instance_start_time' (min)
            - 'instance_end_time'   (min)
            - 'working_start_time'  (min)
            - 'working_end_time'    (min)
        time_customer_depot : array-like, shape (C,)
            Minimal time from customer to depot (hours). (Cus -> ... -> Depot)
        time_depot_customer : array-like, shape (C,)
            Minimal time from depot (or CS1) to customer (hours). (Depot/CS1 -> ... -> Cus)
        service_time : array-like, shape (C,)
            Service time at customer (hours).

        Returns
        -------
        earliest_service_time : np.ndarray, shape (C,), minutes
            Earliest feasible service start time per customer (absolute minutes).
        latest_service_time   : np.ndarray, shape (C,), minutes
            Latest feasible service start time per customer (absolute minutes).
            If infeasible, will be < earliest (you can post-filter).
        """
        # Ensure ndarray and float dtype
        t_cus_dep = np.asarray(time_customer_depot, dtype=float)   # hours
        t_dep_cus = np.asarray(time_depot_customer, dtype=float)   # hours
        svc       = np.asarray(service_time, dtype=float)          # hours

        # Read env (minutes)
        inst_start = float(env.get('instance_start_time', 0.0))
        inst_end   = float(env.get('instance_end_time', 1440.0))
        work_start = float(env.get('working_start_time', 480.0))
        work_end   = float(env.get('working_end_time', 1200.0))

        # Convert hours -> minutes once
        dep_cus_min = t_dep_cus * 60.0
        cus_dep_min = t_cus_dep * 60.0
        svc_min     = svc * 60.0

        # Earliest start: cannot begin service before we arrive or before work starts
        earliest_service_time = max(work_start, inst_start) + dep_cus_min

        # Latest start: must finish service and still be able to return before instance end,
        # and also cannot start after work_end.
        # latest_by_instance = inst_end - cus_dep_min - svc_min
        # latest_by_working  = work_end
        latest_service_time = np.minimum(work_end, inst_end - cus_dep_min - svc_min)

        return earliest_service_time, latest_service_time

class CustomerPositionPolicy(Protocol):

    def sample(
        self,
        env: Dict,
        depot_pos: np.ndarray,
        cs_pos: np.ndarray,
        time_depot_to_css: np.ndarray,
        service_time_policy: np.ndarray,
        rng: np.random.Generator,
        demand_policy
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

class RandomPositionPolicy:
    NAME = "R"

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def __init__(self, env=None, rng=None):
        pass

    def sample(self,
               env,
               depot_pos,
               cs_pos,
               time_depot_to_css,   # (S,) hours: Depot -> CS
               time_css_to_depot,   # (S,) hours: CS -> Depot
               service_time_policy,
               rng,
               demand_policy = None,
               num_customers = None,
               energy_consumption_model_type = None):

        if num_customers is None:
            num_customers = int(env['num_customers']) 
        consumption_per_distance = consumption_model(env, model_type = energy_consumption_model_type)
        (xmin_area, xmax_area), (ymin_area, ymax_area) = env['area_size']
        R = float(env["battery_capacity"]) / consumption_per_distance
        if rng is None:
            rng = env.get("rng", np.random.default_rng())
        dummy_size = int(env.get("dummy_size", 5))
        radius_cus = (float(env['battery_capacity']) / consumption_per_distance) / 2
        speed = float(env['speed'])              # km/h
        p_eff = effective_charging_power_kw(env) # kW

        customers = []
        time_customer_depot = []   # fastest return time (hours)
        time_depot_customer = []   # earliest arrival time (hours)
        service_times_list = []
        demands_list = []

        # Ensure depot_pos is (2,), cs_pos is (S,2)
        candidates_css = np.vstack([depot_pos, cs_pos])  # (1+S, 2)

        # Build union bbox of R/2 rectangles around depot/CS (clipped to area bounds)
        uxmin, uxmax = float('inf'), float('-inf')
        uymin, uymax = float('inf'), float('-inf')
        for (cx, cy) in candidates_css:
            (rxmin, rxmax), (rymin, rymax) = clamp_rect(
                (float(cx), float(cy)), R / 2.0, ((xmin_area, xmax_area), (ymin_area, ymax_area))
            )
            if rxmax > rxmin and rymax > rymin:
                uxmin = min(uxmin, rxmin); uxmax = max(uxmax, rxmax)
                uymin = min(uymin, rymin); uymax = max(uymax, rymax)

        if not np.isfinite([uxmin, uxmax, uymin, uymax]).all():
            uxmin, uxmax, uymin, uymax = xmin_area, xmax_area, ymin_area, ymax_area

        # Proposal sampling loop
        while len(customers) < num_customers:
            B = num_customers * dummy_size
            xs = rng.uniform(uxmin, uxmax, size=B)
            ys = rng.uniform(uymin, uymax, size=B)
            candidate = np.round(np.stack([xs, ys], axis=1), 3)       # (B, 2)
            demands = demand_policy.build(env, num_customers=B, rng=rng)  # (B,)
            service_times = service_time_policy.build(env, num_customers=B, rng=rng, demand=demands)  # (B,) hours

            # Compute feasibility & minimal times for this batch
            out = cus_min_time_to_depot(
                env=env,
                depot_pos=np.asarray(depot_pos).reshape(-1)[:2],  # (2,)
                candidate_cus_pos=candidate,                      # (B,2)
                cs_positions=cs_pos,                              # (S,2)
                cs_time_to_depot=time_css_to_depot,               # CS -> Depot
                time_depot_to_cs=time_depot_to_css,               # Depot -> CS
                radius=radius_cus,
                speed=speed,
                p_eff=p_eff,
                service_times=service_times
            )

            feas_vec, min_time_cus_dep, min_time_dep_cus = out  # (B,), (B,), (B,)
            # Keep only feasible candidates from this batch
            if np.any(feas_vec):
                customers.append(candidate[feas_vec, :])
                time_customer_depot.append(min_time_cus_dep[feas_vec])
                time_depot_customer.append(min_time_dep_cus[feas_vec])
                service_times_list.append(service_times[feas_vec])
                demands_list.append(demands[feas_vec])

            # Optional: early stop if we’ve already exceeded target by a lot
            if sum(x.shape[0] for x in customers) >= num_customers * 2:
                break

        # Concatenate and truncate to num_customers
        if customers:
            customers = np.concatenate(customers, axis=0)[:num_customers]
            time_customer_depot = np.concatenate(time_customer_depot, axis=0)[:num_customers]
            time_depot_customer = np.concatenate(time_depot_customer, axis=0)[:num_customers]
            service_times_list = np.concatenate(service_times_list, axis=0)[:num_customers]
            demands_list = np.concatenate(demands_list, axis=0)[:num_customers]
        else:
            customers = np.empty((0, 2)); time_customer_depot = np.empty((0,))
            time_depot_customer = np.empty((0,)); service_times_list = np.empty((0,))

        # Compute feasible service windows (minutes, absolute clock)
        t_earliest, t_latest = _process_serve_time(
            env, time_customer_depot, time_depot_customer, service_times_list
        )

        return (
            np.asarray(customers, dtype=float),
            np.asarray(service_times_list, dtype=float),   # hours
            np.asarray(t_earliest, dtype=float),           # minutes
            np.asarray(t_latest, dtype=float),             # minutes
            np.asarray(demands_list, dtype=float)
        )
        
class ClusterPositionPolicy:
    NAME = "C"

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


    def __init__(self, env, rng=None):
        self.cluster_number_fun = ClusterNumberPolicies.from_env(env, rng=rng)  
        self.num_customers_per_cluster = ClusterAssignmentPolicies.from_env(env, rng=rng)

    def sample(self,
                env, 
                depot_pos, 
                cs_pos,
                time_css_to_depot,
                time_depot_to_css,
                service_time_policy, 
                rng,
                demand_policy = None,
                num_customers = None,
                energy_consumption_model_type = None):

        if num_customers is None:
            num_customers = int(env['num_customers']) 

        consumption_per_distance = consumption_model(env, model_type = energy_consumption_model_type)
        dummy_size = int(env.get("dummy_size", 5))
        css_positions = np.concatenate([depot_pos.reshape(1,2), cs_pos], axis=0)  # (S+1, 2)
        radius_cus = (float(env['battery_capacity']) / consumption_per_distance) / 2 # round trip radius
        max_iter = 100
        dist_threshold = 10.0  # min distance between cluster centers

        cluster_number = self.cluster_number_fun.build(env, rng=rng)
        cluster_number = min(cluster_number, num_customers)  # avoid over-clustering
        num_customers_per_cluster_list = self.num_customers_per_cluster.build(env, cluster_number, rng=rng, num_customers = num_customers)
        k = num_customers_per_cluster_list.shape[0]

        speed = float(env['speed'])              # km/h
        p_eff = effective_charging_power_kw(env) # kW
        

        # valid sample area
        for (cx, cy) in css_positions:
            min_x = max(env['area_size'][0][0], cx - radius_cus)
            max_x = min(env['area_size'][0][1], cx + radius_cus)
            min_y = max(env['area_size'][1][0], cy - radius_cus)
            max_y = min(env['area_size'][1][1], cy + radius_cus)
        
        # Part1: cluster center sampling
        # tentative: randomly sample k points, then do rejection sampling to enforce min distance (if too close, resample)
        # if it exceed max trials, just accept current points.
        cluster_positions = []
        # monitor tries
        monitor_tries = []
        for _ in range(k):
            try_iter = 0
            while True:
                cx = np.round(rng.uniform(min_x, max_x), 2)
                cy = np.round(rng.uniform(min_y, max_y), 2)

                # check feasibility:
                candidate = np.array([cx, cy])
                if dist_vec(candidate.reshape(1,2), css_positions).min() > radius_cus - 1e-6:
                    continue

                # check distance to existing cluster centers
                if try_iter == max_iter or all(np.linalg.norm(candidate - np.array(cp)) >= dist_threshold for cp in cluster_positions):
                    cluster_positions.append(candidate)
                    monitor_tries.append(try_iter)
                    break
                else:
                    try_iter += 1
        customers = []
        time_depot_customer = []
        time_customer_depot = []
        service_times_list = []
        demands_list = []

        # Part2: within-cluster customer sampling
        # Policy1: Isotropic
        lb, ub = 0.7, 1.0
        record = []
        for idx in range(k):
            customers_k = np.zeros((0,2))
            time_depot_to_customer_k = np.zeros((0,))
            time_customer_to_depot_k = np.zeros((0,))
            service_times_k = np.zeros((0,))
            demands_k = np.zeros((0,))

            cluster_range = rng.uniform(lb, ub) * dist_threshold
            num_customers_per_cluster = num_customers_per_cluster_list[idx]
            try_time = 0
            while len(customers_k) < num_customers_per_cluster:
                demands = demand_policy.build(env, num_customers=num_customers_per_cluster * dummy_size, rng=rng)  # (B,)
                service_times = service_time_policy.build(env, num_customers=num_customers_per_cluster * dummy_size, rng=rng, demand = demands)  # (B,) hours
                angle = rng.uniform(0, 2*np.pi, size = num_customers_per_cluster * dummy_size)
                r = rng.uniform(0, cluster_range, size = num_customers_per_cluster * dummy_size)
                cx = cluster_positions[idx][0] + r * np.cos(angle)
                cy = cluster_positions[idx][1] + r * np.sin(angle)

                candidate = np.stack((cx, cy), axis=1)
                # check feasibility:
                # Compute feasibility & minimal times for this batch
                out = cus_min_time_to_depot(
                    env=env,
                    depot_pos=np.asarray(depot_pos).reshape(-1)[:2],  # (2,)
                    candidate_cus_pos=candidate,                      # (B,2)
                    cs_positions=cs_pos,                              # (S,2)
                    cs_time_to_depot=time_css_to_depot,               # CS -> Depot
                    time_depot_to_cs=time_depot_to_css,               # Depot -> CS
                    radius=radius_cus,
                    speed=speed,
                    p_eff=p_eff,
                    service_times=service_times
                )

                feas_vec, min_time_cus_dep, min_time_dep_cus = out  # (B,), (B,), (B,)
                # Keep only feasible candidates from this batch
                if np.any(feas_vec):
                    customers_k = np.concatenate((customers_k, candidate[feas_vec, :]), axis=0)
                    time_customer_to_depot_k = np.concatenate((time_customer_to_depot_k, min_time_cus_dep[feas_vec]), axis=0)
                    time_depot_to_customer_k = np.concatenate((time_depot_to_customer_k, min_time_dep_cus[feas_vec]), axis=0)
                    service_times_k = np.concatenate((service_times_k, service_times[feas_vec]), axis=0)
                    demands_k = np.concatenate((demands_k, demands[feas_vec]), axis=0)
                try_time += 1
                if try_time >= 5000:
                    # breakpoint()
                    print("Uncommon ISSUE in Position Policy")

            customers_k = customers_k[:num_customers_per_cluster]
            time_depot_to_customer_k = time_depot_to_customer_k[:num_customers_per_cluster]
            time_customer_to_depot_k = time_customer_to_depot_k[:num_customers_per_cluster]
            service_times_k = service_times_k[:num_customers_per_cluster]
            demands_k = demands_k[:num_customers_per_cluster]

            customers.append(customers_k)
            time_depot_customer.append(time_depot_to_customer_k)
            time_customer_depot.append(time_customer_to_depot_k)
            service_times_list.append(service_times_k)
            demands_list.append(demands_k)

        customers = np.vstack(customers)  # (num_customers, 2)
        time_depot_customer = np.hstack(time_depot_customer)  # (num_customers,)
        time_customer_depot = np.hstack(time_customer_depot)  # (num_customers,)
        service_times_list = np.hstack(service_times_list)          # (num
        demands_list = np.hstack(demands_list)                    # (num_customers,)
        if customers.shape[0] != num_customers:
            raise ValueError("Inconsistent shapes in customer position generation.")
        # Policy2: Anisotropic (ellipse)
        # TODO

        # Compute feasible service windows (minutes, absolute clock)
        t_earliest, t_latest = _process_serve_time(
            env, time_customer_depot, time_depot_customer, service_times_list
        )

        return (
            np.asarray(customers, dtype=float),
            np.asarray(service_times_list, dtype=float),   # hours
            np.asarray(t_earliest, dtype=float),           # minutes
            np.asarray(t_latest, dtype=float),             # minutes
            np.asarray(demands_list, dtype=float)
        )

class MixedRCPositionPolicy:
    NAME = "RC"

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def __init__(self, env, random_policy = None, cluster_policy = None, rng=None):
        self.rc_policies = RCPolicies.from_env(env, rng=rng)
        self.random_policy = random_policy or RandomPositionPolicy(env)
        self.cluster_policy = cluster_policy or ClusterPositionPolicy(env, rng=rng)
        self.cluster_number_fun = ClusterNumberPolicies.from_env(env, rng=rng)  
        self.num_customers_per_cluster = ClusterAssignmentPolicies.from_env(env, rng=rng)
            
    def sample(self,
                env, 
                depot_pos, 
                cs_pos,
                time_css_to_depot,
                time_depot_to_css,
                service_time_policy, 
                rng,
                demand_policy = None,
                num_customers = None,
                energy_consumption_model_type = None):
            
            if num_customers == None:
                num_customers = env['num_customers']

            num_random_customer, num_cluster_customer = self.rc_policies.build(env, rng = rng)
            cluster_number = self.cluster_number_fun.build(env, rng=rng)
            cluster_number = min(cluster_number, num_cluster_customer)
            assignments = self.num_customers_per_cluster.build(env, cluster_number, rng=rng, num_customers = num_cluster_customer)
            env['num_customers_per_cluster'] = assignments

            random_output = self.random_policy(env, 
                                               depot_pos, 
                                               cs_pos,
                                               time_css_to_depot,
                                               time_depot_to_css,
                                               service_time_policy, 
                                               rng,
                                               demand_policy = demand_policy,
                                               num_customers = num_random_customer,
                                               energy_consumption_model_type = energy_consumption_model_type)

            cluster_output = self.cluster_policy(env, 
                                                depot_pos, 
                                                cs_pos,
                                                time_css_to_depot,
                                                time_depot_to_css,
                                                service_time_policy, 
                                                rng,
                                                demand_policy = demand_policy,
                                                num_customers = num_cluster_customer,
                                                energy_consumption_model_type = energy_consumption_model_type) 

            random_customers, random_service_time, random_t_earliest, random_t_latest, random_demands = random_output
            cluster_customers, clusterservice_time, cluster_t_earliest, cluster_t_latest, cluster_demands = cluster_output

            customers = np.concatenate((random_customers, cluster_customers), axis = 0)
            t_earliest = np.concatenate((random_t_earliest, cluster_t_earliest))
            t_latest = np.concatenate((random_t_latest, cluster_t_latest))
            service_times_list = np.concatenate((random_service_time, clusterservice_time))
            demands_list = np.concatenate((random_demands, cluster_demands))
            if customers.shape[0] != env['num_customers']:
                breakpoint()
                raise ValueError("Inconsistent shapes in customer position generation.")
            return (
                np.asarray(customers, dtype=float),
                np.asarray(service_times_list, dtype=float),   # hours
                np.asarray(t_earliest, dtype=float),           # minutes
                np.asarray(t_latest, dtype=float),             # minutes
                np.asarray(demands_list, dtype=float)
            )
class CustomerPositionPolicies:
    REGISTRY = {
        "R": RandomPositionPolicy,
        "C": ClusterPositionPolicy,
        "RC": MixedRCPositionPolicy,
    }

    @classmethod
    def from_env(cls, env: Dict, rng: np.random.Generator = None) -> "CustomerPositionPolicy":
        """
        Select a customer position policy from the environment configuration.

        Priority:
        1. If 'test_instance_type' is present in env, use it (for debugging / deterministic testing).
        2. Otherwise, sample from 'instance_type_distribution' according to its probabilities.
        3. If both missing or invalid, raise ValueError.

        The selected type will be stored back into env['instance_type'] for reproducibility.
        """
        dist = env.get("instance_type_distribution", None)

        if "test_instance_type" in env:
            choice = env["test_instance_type"]
        
        elif dist is not None and isinstance(dist, dict) and len(dist) > 0:
            keys = list(dist.keys())
            probs = np.array(list(dist.values()), dtype=float)
            probs /= probs.sum()  # normalize to avoid rounding drift
            if rng is None:
                rng = np.random.default_rng()
            choice = rng.choice(keys, p=probs)

        else:
            raise ValueError(
                "Neither 'instance_type_distribution' nor 'test_instance_type' "
                "is provided or valid in env."
            )

        if choice not in cls.REGISTRY:
            raise ValueError(f"Unknown instance_type: {choice}")
        env["instance_type"] = choice  # record sampled type for reference/logging
        return cls.REGISTRY[choice](env, rng=rng)
