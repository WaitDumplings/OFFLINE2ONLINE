import heapq
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


def reconstruct_path(nxt: Dict[int, Dict[int, Optional[int]]], u: int, v: int) -> List[int]:
    """Return node sequence [u, ..., v] using next-hop table; empty if unreachable."""
    if u not in nxt or v not in nxt[u] or nxt[u][v] is None:
        return []
    path = [u]
    cur = u
    seen = {u}
    while cur != v:
        cur = nxt[cur][v]
        if cur is None or cur in seen:
            return []
        seen.add(cur)
        path.append(cur)
    return path


@dataclass
class PathInfo:
    arrival_node: int                 # customer index (0-based)
    arrival_load: float
    arrival_time: float               # absolute time AFTER service at customer
    arrival_soc: float                # SoC when arriving at customer (before service)
    path_nodes: List[int]             # global node sequence from start -> ... -> customer (inclusive)

    # cache the feasible return-to-depot plan from this customer-state
    return_path_nodes: List[int]      # full path from this customer -> ... -> depot
    return_arrival_time: float        # arrival time at depot if returning immediately


class GreedySolver:
    """
    Nodes indexing (global):
      0                      : depot
      1..num_customers       : customers (global = 1 + cus_idx)
      1+num_customers..end   : charging stations (CS)
    """

    def __init__(self, instance: Dict[str, Any], format='txt') -> None:
        self.format = format
        if format == 'tensor':
            self.instance = self._init_tensor(instance)
        elif format == 'txt':
            self.instance = self._init_txt(instance)
        else:
            raise ValueError(f"Unknown instance format: {format}")

    def _init_txt(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        self.instance = instance
        customers = instance["customers"]
        depot = instance["depot"]
        css = instance["stations"]

        # global nodes
        self.customers = np.array([(c["x"], c["y"]) for c in customers], dtype=float)
        self.depot = np.array([[depot["x"], depot["y"]]], dtype=float)
        self.css = np.array([(cs["x"], cs["y"]) for cs in css], dtype=float)
        self.nodes = np.vstack([self.depot, self.customers, self.css])

        # optional: string ids
        self.id_strs = {i: node["id"] for i, node in enumerate([depot] + customers + css)}
        self.num_customers = len(customers)
        self.num_cs = len(css)
        self.num_nodes = len(self.nodes)

        # global indices for CS
        self.depot_node = 0
        self.cs_start = 1 + self.num_customers
        self.cs_end = self.cs_start + self.num_cs
        self.css_idx = list(range(self.cs_start, self.cs_end))

        # ===== vehicle params =====
        self.velocity = float(instance["vehicle"]["v"])                 # km/h
        self.consume_rate = float(instance["vehicle"]["r"])             # kWh/km
        self.charging_power = 1.0 / float(instance["vehicle"]["g"])    # g: h/kWh => charging power: kWh/h
        self.fuel_cap = float(instance["vehicle"]["Q"])                # kWh
        self.load_cap = float(instance["vehicle"]["C"])                # tons

        # ===== time settings =====
        self.working_start = float(instance["meta"]["working_startTime"])
        self.working_end = float(instance["meta"]["working_endTime"])
        self.instance_end_time = float(instance["meta"]["instance_endTime"])

        # customer data (0-based cus_idx)
        self.service_time = [float(customers[i]["service"]) for i in range(self.num_customers)]
        self.time_windows = [(float(customers[i]["ready"]), float(customers[i]["due"])) for i in range(self.num_customers)]
        self.demands = [float(customers[i]["demand"]) for i in range(self.num_customers)]

        # geometry
        self.distance_matrix = self._distance_matrix(self.nodes)

        # stop-graph allpairs on {depot} U CS
        self.time, self.nxt = self._precompute_stopgraph_allpairs()

        # visited & state (for current vehicle)
        self.visited = [False] * self.num_customers
        self.state = {
            "id": self.depot_node,
            "time": self.working_start,
            "soc": self.fuel_cap,
            "load": 0.0,
        }

        self.last_move_path: List[int] = []
        self.cached_return_path: Optional[List[int]] = None
        self.cached_return_arrival_time: Optional[float] = None

    def _init_tensor(self, instance: Dict[str, Any]) -> None:
        self.instance = instance

        # global nodes
        self.customers = instance["customers"]
        self.depot = instance["depot"]
        self.css = instance["charging_stations"]
        self.nodes = np.vstack([self.depot, self.customers, self.css])

        # optional: string ids
        depot_id = ["D0"]
        cus_id = ["C" + str(i) for i in range(len(self.customers))]
        cs_id = ["S" + str(i) for i in range(1, len(self.css) + 1)]
        self.id_strs = {i: node_id for i, node_id in enumerate(depot_id + cus_id + cs_id)}

        self.num_customers = len(self.customers)
        self.num_cs = len(self.css)
        self.num_nodes = len(self.nodes)

        # global indices for CS
        self.depot_node = 0
        self.cs_start = 1 + self.num_customers
        self.cs_end = self.cs_start + self.num_cs
        self.css_idx = list(range(self.cs_start, self.cs_end))

        # ===== vehicle params =====
        self.velocity = float(instance["env"]["speed"])
        self.consume_rate = float(instance["env"]["consumption_per_distance"])
        self.charging_power = float(instance["env"]["charging_speed"])
        self.fuel_cap = float(instance["env"]["battery_capacity"])
        self.load_cap = float(instance["env"]["loading_capacity"])

        # ===== time settings =====
        self.working_start = float(instance["env"]["working_startTime"] / 60.0)
        self.working_end = float(instance["env"]["working_endTime"] / 60.0)
        self.instance_end_time = float(instance["env"]["instance_endTime"] / 60.0)

        # customer data
        self.service_time = instance["service_time"].astype(float) / 60.0
        self.time_windows = instance["tw"].astype(float) / 60.0
        self.demands = instance["demands"].astype(float)

        # geometry
        self.distance_matrix = self._distance_matrix(self.nodes)

        # stop-graph allpairs on {depot} U CS
        self.time, self.nxt = self._precompute_stopgraph_allpairs()

        # visited & state (for current vehicle)
        self.visited = [False] * self.num_customers
        self.state = {
            "id": self.depot_node,
            "time": self.working_start,
            "soc": self.fuel_cap,
            "load": 0.0,
        }

        self.last_move_path: List[int] = []
        self.cached_return_path: Optional[List[int]] = None
        self.cached_return_arrival_time: Optional[float] = None

    # ------------------------ physics helpers ------------------------

    def _distance_matrix(self, nodes: np.ndarray) -> np.ndarray:
        return np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)

    def travel_time(self, u: int, v: int) -> float:
        return float(self.distance_matrix[u][v] / self.velocity)

    def travel_energy(self, u: int, v: int) -> float:
        return float(self.consume_rate * self.distance_matrix[u][v])

    def is_cs(self, node: int) -> bool:
        return self.cs_start <= node < self.cs_end

    def is_customer(self, node: int) -> bool:
        return 1 <= node <= self.num_customers

    # ------------------------ stop-graph allpairs ------------------------

    def _precompute_stopgraph_allpairs(self) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, Optional[int]]]]:
        """
        All-pairs shortest time and path on stop-graph nodes = {depot} U CS.

        Edge rule (between stop nodes only):
          u->v feasible if travel_energy(u,v) <= fuel_cap
          cost = travel_time(u,v) + (travel_energy(u,v)/P if v is CS else 0)
          (no charging term if v is depot)
        """
        depot = self.depot_node
        cs_nodes = list(self.css_idx)
        nodes = [depot] + cs_nodes
        is_cs_set = set(cs_nodes)

        INF = float("inf")
        P = float(self.charging_power)
        tol = 1e-9

        dist: Dict[int, Dict[int, float]] = {u: {v: INF for v in nodes} for u in nodes}
        nxt: Dict[int, Dict[int, Optional[int]]] = {u: {v: None for v in nodes} for u in nodes}

        for u in nodes:
            dist[u][u] = 0.0
            nxt[u][u] = u

        # direct edges
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                e = self.travel_energy(u, v)
                if e <= self.fuel_cap + tol:
                    if P <= 0 and v in is_cs_set:
                        continue
                    t = self.travel_time(u, v)
                    w = t if v == depot else (t + e / P)
                    if w < dist[u][v]:
                        dist[u][v] = w
                        nxt[u][v] = v

        # Floyd–Warshall
        for k in nodes:
            dk = dist[k]
            for i in nodes:
                dik = dist[i][k]
                if dik >= INF / 2:
                    continue
                di = dist[i]
                ni = nxt[i]
                for j in nodes:
                    if dk[j] >= INF / 2:
                        continue
                    cand = dik + dk[j]
                    if cand + 1e-12 < di[j]:
                        di[j] = cand
                        ni[j] = ni[k]

        return dist, nxt

    def _stop_path(self, u: int, v: int) -> List[int]:
        return reconstruct_path(self.nxt, u, v)

    # ------------------------ return-to-depot ------------------------

    def _can_return_to_depot(self, cur_global: int, cur_time: float, cur_soc: float) -> bool:
        """
        Use the exact same logic as path construction.
        If a concrete return path exists, then returning to depot is feasible.
        """
        return self._return_to_depot_path(cur_global, cur_time, cur_soc) is not None

    def _return_to_depot_path(self, cur_global: int, cur_time: float, cur_soc: float) -> Optional[Tuple[List[int], float]]:
        """
        Build a feasible path from current node to depot, returning:
            (path_nodes, arrival_time_at_depot)

        Important:
          - If current is a customer and direct customer->depot is infeasible,
            we explicitly build customer->CS->...->depot using Floyd stop-path.
          - If current is a CS, and current SoC is not enough to follow the precomputed
            stop-graph plan, we first charge to full at this CS, then follow stop-path.
        """
        depot = self.depot_node
        INF = float("inf")
        deadline = self.instance_end_time
        tol = 1e-9

        if cur_global == depot:
            return [depot], cur_time

        # ---------- current node is a CS ----------
        if self.is_cs(cur_global):
            # direct CS -> depot without additional charging
            e_direct = self.travel_energy(cur_global, depot)
            t_direct = self.travel_time(cur_global, depot)
            if e_direct <= cur_soc + tol and cur_time + t_direct <= deadline + tol:
                return [cur_global, depot], cur_time + t_direct

            P = float(self.charging_power)
            if P <= 0:
                return None

            # If using stop-graph from current CS, assume we charge to full first.
            charge_to_full = max(0.0, self.fuel_cap - cur_soc) / P

            p = self._stop_path(cur_global, depot)
            if not p:
                return None

            t_cs_dep = self.time[cur_global].get(depot, INF)
            if t_cs_dep >= INF / 2:
                return None

            arrive_time = cur_time + charge_to_full + t_cs_dep
            if arrive_time > deadline + tol:
                return None

            return p, arrive_time

        # ---------- invalid node ----------
        if not self.is_customer(cur_global):
            return None

        # ---------- current node is a customer ----------
        # direct customer -> depot
        e = self.travel_energy(cur_global, depot)
        t = self.travel_time(cur_global, depot)
        if e <= cur_soc + tol and cur_time + t <= deadline + tol:
            return [cur_global, depot], cur_time + t

        # via one CS then stop-graph
        P = float(self.charging_power)
        if P <= 0:
            return None

        best: Optional[Tuple[float, List[int]]] = None  # (arrive_time_depot, path_nodes)

        for cs in self.css_idx:
            e1 = self.travel_energy(cur_global, cs)
            if e1 > cur_soc + tol:
                continue

            t1 = self.travel_time(cur_global, cs)
            if cur_time + t1 > deadline + tol:
                continue

            soc_at_cs = cur_soc - e1
            charge_to_full = max(0.0, self.fuel_cap - soc_at_cs) / P

            t_cs_dep = self.time[cs].get(depot, INF)
            if t_cs_dep >= INF / 2:
                continue

            p_mid = self._stop_path(cs, depot)
            if not p_mid:
                continue

            # full expanded path: customer -> cs -> ... -> depot
            path_nodes = [cur_global, cs] + p_mid[1:]

            arrive_time_depot = cur_time + t1 + charge_to_full + t_cs_dep
            if arrive_time_depot > deadline + tol:
                continue

            if best is None or arrive_time_depot < best[0] - 1e-12:
                best = (arrive_time_depot, path_nodes)

        if best is None:
            return None

        return best[1], best[0]

    # ------------------------ shortest path: cur -> customer ------------------------

    def shortest_time_cur_to_customer(self, cur_global: int, cus_idx: int) -> Optional[PathInfo]:
        # load feasibility
        cur_load = float(self.state["load"])
        if cur_load + self.demands[cus_idx] > self.load_cap + 1e-9:
            return None

        cur_time = float(self.state["time"])
        cur_soc = float(self.state["soc"])

        cus_global = 1 + cus_idx
        ready, due = self.time_windows[cus_idx]
        service_t = self.service_time[cus_idx]

        INF = float("inf")
        P = float(self.charging_power)
        if P <= 0:
            return None

        # ---------- direct ----------
        e_direct = self.travel_energy(cur_global, cus_global)
        t_direct = self.travel_time(cur_global, cus_global)
        arrival_raw = cur_time + t_direct

        if e_direct <= cur_soc + 1e-9 and arrival_raw <= due + 1e-9:
            start_service = max(arrival_raw, ready)
            finish_time = start_service + service_t
            arrive_soc = cur_soc - e_direct

            ret_plan = self._return_to_depot_path(cus_global, finish_time, arrive_soc)
            if ret_plan is not None:
                return_path_nodes, return_arrival_time = ret_plan
                return PathInfo(
                    arrival_node=cus_idx,
                    arrival_load=cur_load + self.demands[cus_idx],
                    arrival_time=finish_time,
                    arrival_soc=arrive_soc,
                    path_nodes=[cur_global, cus_global],
                    return_path_nodes=return_path_nodes,
                    return_arrival_time=return_arrival_time,
                )

        # ---------- multi-hop via CS ----------
        best: Optional[Tuple[float, float, List[int], List[int], float]] = None
        # best = (finish_time, arrive_soc, path_nodes, return_path_nodes, return_arrival_time)

        def try_candidate(travel_total_time: float, arrive_soc2: float, path_nodes2: List[int]) -> None:
            nonlocal best
            arrival_raw2 = cur_time + travel_total_time
            if arrival_raw2 > due + 1e-9:
                return

            start_service2 = max(arrival_raw2, ready)
            finish_time2 = start_service2 + service_t

            ret_plan = self._return_to_depot_path(cus_global, finish_time2, arrive_soc2)
            if ret_plan is None:
                return

            return_path_nodes2, return_arrival_time2 = ret_plan

            if best is None or finish_time2 < best[0] - 1e-12:
                best = (
                    finish_time2,
                    arrive_soc2,
                    path_nodes2,
                    return_path_nodes2,
                    return_arrival_time2,
                )

        # Case A: start is depot
        if cur_global == self.depot_node:
            depot = self.depot_node
            for cs_last in self.css_idx:
                mid = self.time[depot].get(cs_last, INF)
                if mid >= INF / 2:
                    continue

                e_last = self.travel_energy(cs_last, cus_global)
                if e_last > self.fuel_cap + 1e-9:
                    continue
                t_last = self.travel_time(cs_last, cus_global)

                travel_total = mid + t_last
                arrive_soc2 = self.fuel_cap - e_last

                p_mid = self._stop_path(depot, cs_last)
                if not p_mid:
                    continue

                path_nodes2 = p_mid + [cus_global]
                try_candidate(travel_total, arrive_soc2, path_nodes2)

        # Case B: start is customer
        elif self.is_customer(cur_global):
            for cs_enter in self.css_idx:
                e1 = self.travel_energy(cur_global, cs_enter)
                if e1 > cur_soc + 1e-9:
                    continue
                t1 = self.travel_time(cur_global, cs_enter)

                soc_at_cs = cur_soc - e1
                charge_to_full = (self.fuel_cap - soc_at_cs) / P

                for cs_last in self.css_idx:
                    mid = self.time[cs_enter].get(cs_last, INF)
                    if mid >= INF / 2:
                        continue

                    e_last = self.travel_energy(cs_last, cus_global)
                    if e_last > self.fuel_cap + 1e-9:
                        continue
                    t_last = self.travel_time(cs_last, cus_global)

                    travel_total = t1 + charge_to_full + mid + t_last
                    arrive_soc2 = self.fuel_cap - e_last

                    p_mid = self._stop_path(cs_enter, cs_last)
                    if not p_mid:
                        continue

                    path_nodes2 = [cur_global, cs_enter] + p_mid[1:] + [cus_global]
                    try_candidate(travel_total, arrive_soc2, path_nodes2)

        else:
            return None

        if best is None:
            return None

        finish_time, arrive_soc2, path_nodes2, return_path_nodes2, return_arrival_time2 = best
        return PathInfo(
            arrival_node=cus_idx,
            arrival_load=cur_load + self.demands[cus_idx],
            arrival_time=finish_time,
            arrival_soc=arrive_soc2,
            path_nodes=path_nodes2,
            return_path_nodes=return_path_nodes2,
            return_arrival_time=return_arrival_time2,
        )

    # ------------------------ greedy choice ------------------------

    def find_next_customer(self) -> Optional[Tuple[int, PathInfo]]:
        cur_pos = int(self.state["id"])
        remain = [i for i in range(self.num_customers) if not self.visited[i]]
        if not remain:
            return None

        # greedy ordering: closest by Euclidean distance from current node
        scored = [(self.distance_matrix[cur_pos][1 + i], i) for i in remain]
        scored.sort()

        for _, cus_idx in scored:
            info = self.shortest_time_cur_to_customer(cur_pos, cus_idx)
            if info is None:
                continue
            return cus_idx, info

        return None

    def update_state_after_customer(self, cus_idx: int, info: PathInfo) -> None:
        self.last_move_path = info.path_nodes
        self.state["id"] = 1 + cus_idx
        self.state["time"] = info.arrival_time
        self.state["soc"] = info.arrival_soc
        self.state["load"] = info.arrival_load
        self.visited[cus_idx] = True

        # cache exact return plan from this customer-state
        self.cached_return_path = info.return_path_nodes
        self.cached_return_arrival_time = info.return_arrival_time

    # ------------------------ multi-vehicle solve ------------------------

    def reset_new_vehicle(self) -> None:
        """
        Start a NEW vehicle (parallel) at depot:
          - time resets to working_start
          - full battery
          - empty load
        """
        self.state["id"] = self.depot_node
        self.state["time"] = self.working_start
        self.state["soc"] = self.fuel_cap
        self.state["load"] = 0.0

        self.cached_return_path = None
        self.cached_return_arrival_time = None

    def solve(self) -> List[int]:
        """
        Return FULL path as a global-node sequence, e.g. [0, 1, 4, 2, 0, 3, 0].

        Multi-vehicle semantics:
          - Each time we "return to depot and reset", we dispatch a new vehicle.
          - Therefore, we reset time to working_start for that new vehicle.
          - Each customer is visited at most once globally.
        """
        full_route: List[int] = []

        def append_move(move: List[int]) -> None:
            if not move:
                return
            if not full_route:
                full_route.extend(move)
            else:
                if full_route[-1] == move[0]:
                    full_route.extend(move[1:])
                else:
                    full_route.extend(move)

        while not all(self.visited):
            self.reset_new_vehicle()
            append_move([self.depot_node])

            visited_this_vehicle = 0

            while True:
                picked = self.find_next_customer()

                if picked is None:
                    cur_id = int(self.state["id"])
                    cur_t = float(self.state["time"])
                    cur_soc = float(self.state["soc"])

                    if cur_id != self.depot_node:
                        # prefer cached exact return plan from the current customer-state
                        if (
                            self.cached_return_path is not None
                            and len(self.cached_return_path) > 0
                            and self.cached_return_path[0] == cur_id
                        ):
                            back_path = self.cached_return_path
                            append_move(back_path)
                        else:
                            ret = self._return_to_depot_path(cur_id, cur_t, cur_soc)
                            if ret is None:
                                raise RuntimeError(
                                    f"Return-to-depot path construction failed from "
                                    f"node={cur_id}, time={cur_t:.6f}, soc={cur_soc:.6f}"
                                )
                            back_path, _arrive_t = ret
                            append_move(back_path)
                    break

                cus_idx, info = picked
                append_move(info.path_nodes)
                self.update_state_after_customer(cus_idx, info)
                visited_this_vehicle += 1

            # if this vehicle couldn't visit anyone, remaining customers are infeasible
            if visited_this_vehicle == 0:
                break

        # ensure final ends at depot
        if full_route and full_route[-1] != self.depot_node:
            cur_id = int(self.state["id"])
            cur_t = float(self.state["time"])
            cur_soc = float(self.state["soc"])

            if (
                self.cached_return_path is not None
                and len(self.cached_return_path) > 0
                and self.cached_return_path[0] == cur_id
            ):
                append_move(self.cached_return_path)
            else:
                ret = self._return_to_depot_path(cur_id, cur_t, cur_soc)
                if ret is None:
                    raise RuntimeError(
                        f"Final depot return failed from "
                        f"node={cur_id}, time={cur_t:.6f}, soc={cur_soc:.6f}"
                    )
                back_path, _arrive_t = ret
                append_move(back_path)

        if self.format == 'tensor':
            return full_route[1:]

        str_route = [self.id_strs[full_route[i]] for i in range(len(full_route))]
        array_route = np.array(str_route)
        route_idx = np.where(array_route == "D0")[0].tolist()

        routes = []
        for i in range(1, len(route_idx)):
            start = route_idx[i - 1]
            end = route_idx[i]
            cur_route = str_route[start:end] + ["D0"]
            routes.append("->".join(cur_route))

        self.global_value = sum(
            [self.distance_matrix[full_route[i]][full_route[i - 1]] for i in range(1, len(full_route))]
        )

        return routes