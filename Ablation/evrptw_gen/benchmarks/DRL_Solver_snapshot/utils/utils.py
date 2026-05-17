import numpy as np
import gym
from evrptw_gen.benchmarks.DRL_Solver.wrappers.recordWrapper import RecordEpisodeStatistics


def update_lambda_fail(
    lambda_fail: float,
    success_rate: float,
    target_success: float,
    lambda_max: float,
    lr_up: float,
    lr_down: float,
    tolerance: float,
    high_success_threshold: float = 0.97,
    high_success_decay: float = 0.98,
) -> float:
    """
    Update rule for failure penalty lambda_fail.

    Main logic:
    - success_rate < target_success - tolerance:
        increase lambda_fail
    - success_rate > target_success + tolerance:
        decrease lambda_fail
    - within tolerance band:
        keep lambda_fail unchanged

    Extra logic:
    - if success_rate is very high, apply an additional multiplicative decay
      so that lambda_fail can gradually retreat instead of staying stuck
      at a positive value forever.
    """
    lambda_fail = float(lambda_fail)
    success_rate = float(success_rate)
    target_success = float(target_success)

    fail_rate = 1.0 - success_rate
    target_fail = 1.0 - target_success
    gap = fail_rate - target_fail   # >0: fail too much; <0: fail too little

    # ===== Main update around target =====
    if gap > tolerance:
        # fail too much -> increase penalty
        lambda_fail = lambda_fail + lr_up * gap
    elif gap < -tolerance:
        # fail much less than target -> decrease penalty
        lambda_fail = lambda_fail - lr_down * (-gap)

    # ===== Extra decay when success is already very high =====
    # This ensures lambda_fail can gradually retreat in the late stage.
    if success_rate >= high_success_threshold:
        lambda_fail = lambda_fail * high_success_decay

    # clip to valid range
    lambda_fail = float(np.clip(lambda_fail, 0.0, lambda_max))
    return lambda_fail


def make_env(env_id, seed, cfg=None):
    if cfg is None:
        cfg = {}
    cfg = dict(cfg)
    cfg.setdefault("seed", int(seed))

    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(int(seed))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def _mean(x): return float(np.mean(x)) if len(x) else 0.0
def _max(x):  return float(np.max(x)) if len(x) else 0.0
def _min(x):  return float(np.min(x)) if len(x) else 0.0
def _p90(x):  return float(np.percentile(x, 90)) if len(x) else 0.0
def _p10(x):  return float(np.percentile(x, 10)) if len(x) else 0.0


def grad_norm(params):
    tot = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        tot += g.pow(2).sum().item()
    return tot ** 0.5

import numpy as np

def assemble_nodes(instance, order="depot-customer-station"):
    """
    Return:
        nodes: list[dict]
        dist_matrix: np.ndarray, shape (N, N)

    order:
        - "depot-station-customer"
        - "depot-customer-station"
    """

    depot = instance["depot"]
    customers = instance.get("customers", [])
    charging_stations = instance.get("stations", [])
    env = instance["env"]

    working_start = float(env["working_startTime"]) / 60.0
    working_end = float(env["working_endTime"]) / 60.0

    def to_2d(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return arr.reshape(0, 2)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    depot = to_2d(depot)
    customers = to_2d(customers)
    charging_stations = to_2d(charging_stations)

    def get_customer_field(candidates, n, default=0.0):
        for key in candidates:
            if key in instance:
                arr = np.asarray(instance[key], dtype=float).reshape(-1)
                if len(arr) != n:
                    raise ValueError(
                        f"Field '{key}' length = {len(arr)}, but number of customers = {n}"
                    )
                return arr
        return np.full(n, default, dtype=float)

    n_customers = len(customers)

    customer_demands = get_customer_field(
        ["customer_demand", "demands", "demand"], n_customers, default=0.0
    )
    customer_ready = get_customer_field(
        ["customer_ready", "ready_times", "ready_time", "ready"], n_customers, default=0.0
    )
    customer_due = get_customer_field(
        ["customer_due", "due_times", "due_time", "due"], n_customers, default=0.0
    )
    customer_service = get_customer_field(
        ["customer_service", "service_times", "service_time", "service"], n_customers, default=0.0
    )

    if len(depot) != 1:
        raise ValueError(f"Expect exactly 1 depot, but got {len(depot)}")

    depot_node = {
        "id": "D0",
        "type": "d",
        "x": float(depot[0, 0]),
        "y": float(depot[0, 1]),
        "demand": 0.0,
        "ready": working_start,
        "due": working_end,
        "service": 0.0,
    }

    customer_nodes = []
    for i in range(n_customers):
        customer_nodes.append({
            "id": f"C{i}",
            "type": "c",
            "x": float(customers[i, 0]),
            "y": float(customers[i, 1]),
            "demand": float(customer_demands[i]),
            "ready": float(customer_ready[i]),
            "due": float(customer_due[i]),
            "service": float(customer_service[i]),
        })

    station_nodes = []
    for i in range(len(charging_stations)):
        station_nodes.append({
            "id": f"S{i+1}",
            "type": "f",
            "x": float(charging_stations[i, 0]),
            "y": float(charging_stations[i, 1]),
            "demand": 0.0,
            "ready": working_start,
            "due": working_end,
            "service": 0.0,
        })

    nodes = [depot_node]
    if order == "depot-station-customer":
        nodes.extend(station_nodes)
        nodes.extend(customer_nodes)
    elif order == "depot-customer-station":
        nodes.extend(customer_nodes)
        nodes.extend(station_nodes)
    else:
        raise ValueError("order must be 'depot-station-customer' or 'depot-customer-station'")

    coords = np.array([[node["x"], node["y"]] for node in nodes], dtype=float)

    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    return nodes, dist_matrix
