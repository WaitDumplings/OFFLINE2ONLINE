# evrptw_gen/utils/feasibility.py
# from __future__ import annotations
from typing import List, Tuple, Dict, NamedTuple
import math
import numpy as np
from .customize import min_R, max_R

class CandidateCSTimes(NamedTuple):
    ok: bool
    cand_to_depot_h: float         # minimal candidate -> ... -> depot (no charge at depot)
    depot_to_cand_full_h: float    # minimal depot -> ... -> candidate, arrive candidate FULL

def effective_charging_power_kw(env: Dict) -> float:
    """Effective charging power (kW) = min(station, vehicle_limit) * efficiency."""
    p_station = float(env.get("charging_speed", 0.0))  # kW
    eff = float(env.get("charging_efficiency", 1.0))
    p_vehicle_limit = float(env.get("vehicle_ac_limit_kw", p_station))
    return max(0.0, min(p_station, p_vehicle_limit) * eff)

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def dist_vec(a, b):
    """Vectorized distance between two sets of points a and b.
    a: (N,D), b: (M,D) -> returns (N,M) array of pairwise distances.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]  # (N,M,D)
    return np.linalg.norm(diff, axis=-1)  # (N,M)

import math
import numpy as np
from typing import List, Tuple

def cs_min_time_to_depot(
    env: dict,
    depot_pos: np.ndarray,                   # (2,), depot coordinates in km
    candidate_cs_pos: np.ndarray,            # (2,), candidate charging station position (km)
    cs_positions: List[np.ndarray],          # list of existing CS positions (km)
    cs_time_to_depot: List[float],           # precomputed: each CS "full-charge → depot" minimal total time (hours)
    depot_time_to_cs: List[float],           # precomputed: depot "full-charge → CS" minimal total time (hours)
    radius: float,                           # km, max drivable distance per full charge
    speed: float,                            # km/h
    p_eff: float,                            # kW (= kWh/h), effective charging power (min of station & vehicle limit)
    use_cs_range: bool = False,              # whether to enforce [min_R, max_R] band constraint
    eps: float = 1e-9
) -> Tuple[bool, float]:
    """
    Evaluate whether a newly generated candidate charging station (CS) is feasible
    and compute its minimal total travel time back to the depot.

    This function is used in *sequential* CS generation:
      - The candidate must be reachable from the existing network (Depot + CS set).
      - Each newly added CS must connect to the graph (single-hop reachability).
      - We assume a full-charge policy at each CS: every departure is from 100% battery.

    Workflow:
      1) Connectivity Check:
         The candidate must have at least one reachable point in the existing set S = {depot} ∪ CS.
         Formally: min_dist(S, candidate) ≤ radius.
         Otherwise, the candidate is discarded (unreachable).

      2) Optional Band Constraint:
         If enabled (use_cs_range=True), the candidate’s nearest distance to S must lie within
         [min_R, max_R], controlling spatial density and outward expansion.

      3) Travel Time Evaluation:
         - Direct path: candidate → depot (travel only, since reaching depot ends the route).
         - One-hop path: candidate → CS_k (drive + recharge at CS_k)
                         + (CS_k full → depot time, precomputed).

      4) Return:
         time unit: hours
         (feasibility_flag, minimal_cs_depot_minuts, minimal_depot_cs_minuts, last_step_to_depot)
    """
    # --- Basic sanity checks ---
    if not (radius > 0 and speed > 0 and p_eff > 0):
        CandidateCSTimes(False, math.inf, math.inf)
    if len(cs_positions) != len(cs_time_to_depot):
        CandidateCSTimes(False, math.inf, math.inf)

    cpd = env['consumption_per_distance']  # kWh per km

    # --- Step 1: Connectivity & Range Validation ---
    S = [depot_pos] + list(cs_positions)
    d_min = min(dist(candidate_cs_pos, s) for s in S) if S else math.inf

    # Sequential connectivity: candidate must be within radius of the existing network
    if d_min > radius + eps:
        return CandidateCSTimes(False, math.inf, math.inf)  # unreachable → discard this candidate

    # Optional band constraint (anti-clustering and outward control)
    if use_cs_range:
        min_radius = min_R(env, depot_pos, cs_positions, radius, speed, p_eff)
        max_radius = max_R(env, depot_pos, cs_positions, radius, speed, p_eff)
        if d_min + eps < min_radius:  # too close → clustering
            return CandidateCSTimes(False, math.inf, math.inf)
        if d_min - eps > max_radius:  # too far → exceeds expansion bound
            return CandidateCSTimes(False, math.inf, math.inf)

    # --- Step 2: Evaluate minimal travel time back to depot ---
    time_candidate_depot, time_depot_candidate, ok, last_step_to_depot = math.inf, math.inf, False, -1
    d_cd = float(np.linalg.norm(candidate_cs_pos - depot_pos))

    # (a) Direct route: candidate → depot
    # Only travel time is considered since reaching depot terminates the route.
    if d_cd <= radius + eps:

        # (directly) candidate cs → depot
        time_candidate_depot = d_cd / speed
        
        # (directly) depot → candidate cs
        time_travel_depot_candidate = d_cd / speed
        time_charge_depot_candidate = d_cd * cpd / p_eff  # recharge energy used during this leg
        time_depot_candidate = time_travel_depot_candidate + time_charge_depot_candidate

        # (directly) feasible
        ok = True
    else:
        # --- multi-hops route via existing CSs ---
        for k, cs_k in enumerate(cs_positions):
            d_candidate_cs_k = dist(candidate_cs_pos, cs_k)

            # feasibility check: candidate → CS_k
            if d_candidate_cs_k <= radius + eps:
                # travel + charge time from candidate → CS_k
                time_travel_candidate_cs_k = d_candidate_cs_k / speed
                time_charge_candidate_cs_k = (time_travel_candidate_cs_k * speed) * cpd / p_eff  # recharge energy used during this leg
                total_time_candidate_cs_k = time_travel_candidate_cs_k + time_charge_candidate_cs_k

                # total time: candidate → CS_k + CS_k full → depot
                total_time = total_time_candidate_cs_k + cs_time_to_depot[k]

                if total_time < time_candidate_depot - eps:
                    total_time_cs_k_candidate = total_time_candidate_cs_k
                    # (indirectly) candidate cs → depot
                    time_candidate_depot = total_time
                    # (indirectly) depot → candidate cs
                    time_depot_candidate = depot_time_to_cs[k] + total_time_cs_k_candidate
                    # (indirectly) feasible
                    ok = True

    # hours -> minutes
    return CandidateCSTimes(ok, time_candidate_depot, time_depot_candidate)
        
def cus_min_time_to_depot(
    env: Dict,                              # Environment parameters (e.g., consumption rates, capacities)
    depot_pos: np.ndarray,                  # (2,) Depot coordinates in km
    candidate_cus_pos: np.ndarray,          # (C, 2) Candidate customer coordinates in km
    cs_positions: List[np.ndarray],         # (S, 2) Charging station coordinates in km
    cs_time_to_depot: List[float],          # (S,) Travel time from each CS to depot in hours
    time_depot_to_cs: List[float],          # (S,) Travel time from depot to each CS in hours
    radius: float,                          # km, maximum travel distance per full charge
    speed: float,                           # km/h
    p_eff: float,                           # kW (= kWh/h), effective charging power
    service_times: np.ndarray               # (C,) Service time per customer in hours
    ) -> Tuple[bool, float]:
        """
        Compute the minimal feasible travel times between depot, charging stations, and customers.

        For each customer, check if there exists at least one feasible route:
            Depot/CS₁ → Customer → CS₂/Depot,
        that satisfies all operational and energy constraints.

        Constraints:
        1) Energy constraint:
        - Each leg must satisfy dist(A, B) ≤ radius (full-charge limit).
        - The total energy consumption (CS₁→Cus→CS₂) ≤ battery_capacity.
        2) Working time window:
        - Arrival time at the customer ≤ working_endTime.
        3) Instance time window:
        - Total return time to the depot ≤ instance_endTime.

        Under the full-charge policy:
        - (Depot→CS) and (CS→Depot) legs are always feasible.
        - Only (CS₁→Cus→CS₂) segments require explicit feasibility checks.
        """
        # Combine depot and charging stations into a single position array
        css_pos = np.vstack((depot_pos, cs_positions))  # (1+S, 2)
        # Include depot-to-CS and CS-to-depot travel times with depot at index 0
        time_depot_to_cs = np.concatenate(([0.0], time_depot_to_cs), axis=0)
        cs_time_to_depot = np.concatenate(([0.0], cs_time_to_depot), axis=0)

        # Basic parameters
        cpd = env['consumption_per_distance']  # kWh per km
        working_start_time = float(env.get('working_startTime', 0.0)) / 60.0  # hours
        working_end_time = float(env.get('working_endTime', 1440.0)) / 60.0    # hours
        instance_end_time = float(env.get('instance_endTime', 1440.0)) / 60.0  # hours

        # ----------------------------
        # Compute leg-level travel times and energy consumption
        # ----------------------------
        time_cus_to_cs = dist_vec(candidate_cus_pos, css_pos) / speed          # (C, 1+S), Cus→CS
        time_cs_to_cus = time_cus_to_cs.transpose((1, 0))                     # (1+S, C), CS→Cus

        # Total energy consumption for CS₁→Cus→CS₂
        soc_cs_to_cs = (time_cs_to_cus[:, :, None] + time_cus_to_cs[None, :, :]) * speed * cpd  # (1+S, C, 1+S)

        # Charging durations
        time_charging_depot_to_cs = (time_depot_to_cs * speed) * cpd / p_eff   # (1+S,)
        time_charging_cs_to_cs = soc_cs_to_cs / p_eff                          # (1+S, C, 1+S)

        # Arrival time at each customer (Depot/CS₁ → Cus)
        arriving_time_at_cus = (
            time_depot_to_cs[:, None] + time_cs_to_cus + time_charging_depot_to_cs[:, None]
        )  # (1+S, C)

        # Time to complete the route (Depot/CS₁ → Cus → CS₂ → Depot)
        time_first_half = arriving_time_at_cus + service_times.reshape(1, -1)   # (1+S, C)
        time_second_half = (
            time_cus_to_cs[None, :, :] + time_charging_cs_to_cs + cs_time_to_depot.reshape(1, 1, -1)
        )  # (1+S, C, 1+S)
        arriving_time_back_to_depot = time_first_half[:, :, None] + time_second_half  # (1+S, C, 1+S)

        # ----------------------------
        # Feasibility mask construction
        # ----------------------------
        mask = np.zeros_like(arriving_time_back_to_depot, dtype=bool)
        # True = infeasible (any violated constraint)

        # (1) Energy constraint: total energy usage must not exceed capacity
        mask |= soc_cs_to_cs > env['battery_capacity'] + 1e-9

        # # (2) Working window constraint: arrival at customer must be within work hours
        # mask |= (working_start_time + arriving_time_at_cus > working_end_time + 1e-9)[:, :, None]

        # (3) Instance window constraint: total return to depot must be within instance time
        mask |= (working_start_time + arriving_time_back_to_depot > instance_end_time + 1e-9)

        # ----------------------------
        # Feasibility evaluation
        # ----------------------------
        feasible_mask = ~mask  # True = all constraints satisfied (feasible)
        feasibility = np.any(feasible_mask, axis=(0, 2))  # (C,), whether each customer is serviceable

        # Masks for partial legs
        first_half_mask = np.any(feasible_mask, axis=2)   # (1+S, C), feasible (CS₁→Cus)
        second_half_mask = np.any(feasible_mask, axis=0)  # (C, 1+S), feasible (Cus→CS₂)

        # ----------------------------
        # Minimal travel times (hours)
        # ----------------------------
        # Earliest arrival time at customer (only considering feasible complete routes)
        time_depot_candidate = np.min(
            np.where(first_half_mask, arriving_time_at_cus, np.inf), axis=0
        )  # (C,)

        # Fastest return time to depot (feasible paths only)
        time_candidate_depot = np.min(
            np.where(feasible_mask, time_second_half, np.inf), axis=(0, 2)
        )  # (C,)

        return CandidateCSTimes(feasibility, time_candidate_depot, time_depot_candidate)







