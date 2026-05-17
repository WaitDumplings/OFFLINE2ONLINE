#!/usr/bin/env python3
"""Demo for reference-route potential shaping on offline ALNS routes.

The proposed potential is based on a reference solution, not on independent
single-customer repairs.  For a state with remaining customers U(s), we project
the reference routes by deleting already-served customers and measure the
remaining route distance:

    H_ref(s) = cost(Project(reference_routes, U(s))).

Serving customer j receives progress

    r_j = (H_ref(s) - H_ref(s')) / H_ref(s0),

which is positive when deleting j shortens the projected reference completion.
This script verifies the sign and telescoping behavior on buffer_1k without
touching the training code or modifying any data files.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSet, Sequence, Tuple

import numpy as np


DEFAULT_ROOT = Path("/data/Maojie/Github2/EVRP-TW-D-B_Weekend")
DEFAULT_INSTANCE_PKL = (
    DEFAULT_ROOT / "dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl"
)
DEFAULT_PROGRESS_PKL = (
    DEFAULT_ROOT / "dataset/unanchored/Cus_50/buffer_1k/progress/buffer_progress.pkl"
)
DEFAULT_OUT_CSV = DEFAULT_ROOT / "reference_route_pbrs_demo.csv"


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def instance_key(instance: Mapping) -> str:
    return str(instance.get("file") or f"{instance.get('instance_id')}.txt")


def build_instance_map(instances: Sequence[Mapping]) -> Dict[str, Mapping]:
    by_file: Dict[str, Mapping] = {}
    for inst in instances:
        file_name = instance_key(inst)
        by_file[file_name] = inst
        if file_name.endswith(".txt"):
            by_file[file_name[:-4]] = inst
    return by_file


def build_distance_matrix(instance: Mapping) -> Tuple[np.ndarray, int]:
    depot = np.asarray(instance["depot"], dtype=np.float64)
    customers = np.asarray(instance["customers"], dtype=np.float64)
    stations = np.asarray(instance.get("charging_stations", []), dtype=np.float64)
    if stations.size == 0:
        coords = np.concatenate([depot, customers], axis=0)
    else:
        coords = np.concatenate([depot, customers, stations], axis=0)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist, int(customers.shape[0])


def normalize_route(route: Sequence[int]) -> List[int]:
    clean = [int(x) for x in route]
    if not clean:
        return [0, 0]
    if clean[0] != 0:
        clean.insert(0, 0)
    if clean[-1] != 0:
        clean.append(0)
    return clean


def route_has_remaining_customer(route: Sequence[int], remaining: MutableSet[int], num_customers: int) -> bool:
    return any(1 <= node <= num_customers and node in remaining for node in route)


def project_route(
    route: Sequence[int],
    remaining: MutableSet[int],
    num_customers: int,
    keep_stations: bool,
) -> List[int]:
    projected: List[int] = []
    for node in normalize_route(route):
        if node == 0:
            projected.append(node)
        elif 1 <= node <= num_customers:
            if node in remaining:
                projected.append(node)
        elif keep_stations:
            projected.append(node)

    if not any(1 <= node <= num_customers for node in projected):
        return []

    if projected[0] != 0:
        projected.insert(0, 0)
    if projected[-1] != 0:
        projected.append(0)

    collapsed: List[int] = []
    for node in projected:
        if collapsed and collapsed[-1] == 0 and node == 0:
            continue
        collapsed.append(node)
    return collapsed


def project_routes(
    routes: Sequence[Sequence[int]],
    remaining: MutableSet[int],
    num_customers: int,
    keep_stations: bool,
) -> List[List[int]]:
    projected_routes = []
    for route in routes:
        if route_has_remaining_customer(route, remaining, num_customers):
            projected = project_route(route, remaining, num_customers, keep_stations)
            if projected:
                projected_routes.append(projected)
    return projected_routes


def route_cost(routes: Sequence[Sequence[int]], dist: np.ndarray) -> float:
    total = 0.0
    n = dist.shape[0]
    for route in routes:
        clean = normalize_route(route)
        for a, b in zip(clean[:-1], clean[1:]):
            if not (0 <= a < n and 0 <= b < n):
                raise ValueError(f"route node out of range: {a}->{b}, dist matrix has {n} nodes")
            total += float(dist[a, b])
    return total


def route_customer_order(routes: Sequence[Sequence[int]], num_customers: int) -> List[int]:
    order: List[int] = []
    seen = set()
    for route in routes:
        for node in route:
            node = int(node)
            if 1 <= node <= num_customers and node not in seen:
                order.append(node)
                seen.add(node)
    return order


def static_neighbor_deltas(
    routes: Sequence[Sequence[int]],
    dist: np.ndarray,
    num_customers: int,
    keep_stations: bool,
) -> Dict[int, float]:
    """Single-pass route marginal d(prev,j)+d(j,next)-d(prev,next)."""
    deltas: Dict[int, float] = {}
    for route in routes:
        clean = normalize_route(route)
        if not keep_stations:
            clean = [node for node in clean if node == 0 or 1 <= node <= num_customers]
            clean = normalize_route(clean)
        for idx, node in enumerate(clean):
            if 1 <= node <= num_customers:
                prev_node = clean[idx - 1]
                next_node = clean[idx + 1]
                deltas[node] = float(
                    dist[prev_node, node] + dist[node, next_node] - dist[prev_node, next_node]
                )
    return deltas


def evaluate_record(
    file_name: str,
    record: Mapping,
    instance: Mapping,
    keep_stations: bool,
) -> Dict[str, float | int | str]:
    dist, num_customers = build_distance_matrix(instance)
    routes = record.get("best_routes") or record.get("current_routes") or []
    if not routes:
        raise ValueError(f"{file_name}: no best_routes/current_routes in progress record")

    route_customers = route_customer_order(routes, num_customers)
    missing = sorted(set(range(1, num_customers + 1)) - set(route_customers))
    duplicate_count = sum(len([n for n in route if 1 <= int(n) <= num_customers]) for route in routes) - len(route_customers)

    remaining = set(route_customers)
    h0 = route_cost(project_routes(routes, remaining, num_customers, keep_stations), dist)
    raw_route_cost = route_cost(routes, dist)

    deltas: List[float] = []
    h_prev = h0
    worst_telescoping_error = 0.0
    neg_steps = 0
    min_delta = math.inf
    max_delta = -math.inf

    for customer in route_customers:
        remaining.remove(customer)
        h_next = route_cost(project_routes(routes, remaining, num_customers, keep_stations), dist)
        delta = h_prev - h_next
        deltas.append(delta)
        if delta < -1e-7:
            neg_steps += 1
        min_delta = min(min_delta, delta)
        max_delta = max(max_delta, delta)
        reconstructed = h0 - sum(deltas)
        worst_telescoping_error = max(worst_telescoping_error, abs(reconstructed - h_next))
        h_prev = h_next

    sum_delta = float(sum(deltas))
    final_h = h_prev
    norm_sum = sum_delta / h0 if h0 > 0 else 0.0

    static_d = static_neighbor_deltas(routes, dist, num_customers, keep_stations)
    dynamic_by_customer = dict(zip(route_customers, deltas))
    static_vals = []
    dynamic_vals = []
    for customer in route_customers:
        if customer in static_d:
            static_vals.append(static_d[customer])
            dynamic_vals.append(dynamic_by_customer[customer])
    if len(static_vals) >= 2:
        corr = float(np.corrcoef(np.asarray(static_vals), np.asarray(dynamic_vals))[0, 1])
        mae_static_dynamic = float(np.mean(np.abs(np.asarray(static_vals) - np.asarray(dynamic_vals))))
    else:
        corr = float("nan")
        mae_static_dynamic = float("nan")

    stored = record.get("expert_delta_j")
    if isinstance(stored, list) and len(stored) >= num_customers:
        stored_sum = float(sum(float(x) for x in stored[:num_customers]))
    elif isinstance(stored, dict):
        stored_sum = float(sum(float(v) for v in stored.values()))
    else:
        stored_sum = float("nan")

    return {
        "file": file_name,
        "num_routes": len(routes),
        "num_route_customers": len(route_customers),
        "num_missing_customers": len(missing),
        "duplicate_customer_visits": duplicate_count,
        "raw_route_cost": raw_route_cost,
        "projected_initial_h": h0,
        "progress_delta_sum": sum_delta,
        "final_h": final_h,
        "normalized_progress_sum": norm_sum,
        "negative_steps": neg_steps,
        "min_step_delta": min_delta if deltas else 0.0,
        "max_step_delta": max_delta if deltas else 0.0,
        "worst_telescoping_error": worst_telescoping_error,
        "global_value": float(record.get("global_value", float("nan"))),
        "stored_expert_delta_sum": stored_sum,
        "static_dynamic_corr": corr,
        "static_dynamic_mae": mae_static_dynamic,
    }


def summarize(rows: Sequence[Mapping[str, float | int | str]]) -> Dict[str, float]:
    numeric_keys = [
        "raw_route_cost",
        "projected_initial_h",
        "progress_delta_sum",
        "final_h",
        "normalized_progress_sum",
        "negative_steps",
        "worst_telescoping_error",
        "static_dynamic_corr",
        "static_dynamic_mae",
    ]
    summary = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if not math.isnan(float(row[key]))]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_max"] = float(np.max(vals))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-pkl", type=Path, default=DEFAULT_INSTANCE_PKL)
    parser.add_argument("--progress-pkl", type=Path, default=DEFAULT_PROGRESS_PKL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--drop-stations",
        action="store_true",
        help="Drop charging-station nodes when projecting routes. Default keeps the reference station plan.",
    )
    args = parser.parse_args()

    instances = load_pickle(args.instance_pkl)
    progress = load_pickle(args.progress_pkl)
    instance_by_file = build_instance_map(instances)

    keys = list(progress.keys())
    rng = random.Random(args.seed)
    if args.samples > 0 and args.samples < len(keys):
        keys = rng.sample(keys, args.samples)
    else:
        keys = sorted(keys)

    rows = []
    keep_stations = not args.drop_stations
    for key in keys:
        inst = instance_by_file.get(key) or instance_by_file.get(key[:-4] if key.endswith(".txt") else key)
        if inst is None:
            raise KeyError(f"cannot find instance for progress key {key}")
        rows.append(evaluate_record(key, progress[key], inst, keep_stations=keep_stations))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reference-route PBRS demo rows: {len(rows)}")
    print(f"stations_in_projection: {keep_stations}")
    print(f"csv: {args.out_csv}")
    print()
    for row in rows[: min(5, len(rows))]:
        print(
            "{file} | routes={num_routes} customers={num_route_customers} "
            "H0={projected_initial_h:.3f} sum_delta={progress_delta_sum:.3f} "
            "norm_sum={normalized_progress_sum:.6f} neg={negative_steps} "
            "tel_err={worst_telescoping_error:.3e} static_corr={static_dynamic_corr:.3f}".format(**row)
        )
    print()
    for key, value in summarize(rows).items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
