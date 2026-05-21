from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from solver import ALNS_Solver
except ImportError:  # pragma: no cover - used when imported as a package module.
    from .solver import ALNS_Solver

DELTA_VERSION = "best_feasible_reinsert_v1"
LEGACY_DELTA_VERSION = "route_neighbor_v1"


def _stem(x):
    if x is None:
        return None
    x = str(x).strip().strip("/")
    if not x:
        return None
    return os.path.splitext(os.path.basename(x))[0]


def _safe_register(index: Dict[str, Dict[str, Any]], alias, instance):
    if alias is None or not isinstance(instance, dict):
        return
    alias = str(alias).strip().strip("/")
    if not alias:
        return
    index[alias] = instance
    stem = _stem(alias)
    if stem:
        index[stem] = instance


def build_instance_index(instance_data) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}

    def add_instance(key, inst):
        if not isinstance(inst, dict):
            return
        _safe_register(index, key, inst)
        for k in (
            "instance_id",
            "id",
            "name",
            "file",
            "filename",
            "path",
            "instance_path",
            "txt_path",
        ):
            if k in inst:
                _safe_register(index, inst.get(k), inst)
        for outer_key in ("metadata", "meta", "info", "env"):
            obj = inst.get(outer_key, None)
            if not isinstance(obj, dict):
                continue
            for k in (
                "instance_id",
                "id",
                "name",
                "file",
                "filename",
                "path",
                "instance_path",
                "txt_path",
            ):
                if k in obj:
                    _safe_register(index, obj.get(k), inst)

    if isinstance(instance_data, list):
        for i, inst in enumerate(instance_data):
            add_instance(i, inst)
    elif isinstance(instance_data, dict):
        for wrapper_key in ("instances", "data", "dataset"):
            if wrapper_key in instance_data:
                return build_instance_index(instance_data[wrapper_key])
        for key, inst in instance_data.items():
            add_instance(key, inst)
    else:
        raise ValueError(f"Unsupported instance pickle format: {type(instance_data)}")

    return index


def find_instance_for_progress_entry(
    key: str,
    entry: Dict[str, Any],
    instance_index: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    file_name = entry.get("file", key)
    instance_id = entry.get("instance_id", None)
    aliases = [
        instance_id,
        file_name,
        key,
        _stem(instance_id),
        _stem(file_name),
        _stem(key),
    ]
    for alias in aliases:
        if alias is None:
            continue
        alias = str(alias).strip().strip("/")
        if alias in instance_index:
            return instance_index[alias], alias
        stem = _stem(alias)
        if stem and stem in instance_index:
            return instance_index[stem], stem
    return None, None


def repair_dist_scale(instance: Dict[str, Any]) -> float:
    env = instance.get("env", {})
    area_size = np.asarray(env.get("area_size", [[0.0, 1.0], [0.0, 1.0]]), dtype=np.float32)
    if area_size.shape != (2, 2):
        return 1.0
    return float(max(np.hypot(area_size[0, 1] - area_size[0, 0], area_size[1, 1] - area_size[1, 0]), 1e-6))


def node_coordinates(instance: Dict[str, Any]) -> np.ndarray:
    depot = np.asarray(instance["depot"], dtype=np.float32).reshape(1, 2)
    customers = np.asarray(instance.get("customers", []), dtype=np.float32).reshape(-1, 2)
    stations = np.asarray(instance.get("charging_stations", []), dtype=np.float32).reshape(-1, 2)
    return np.concatenate([depot, customers, stations], axis=0)


def distance_matrix(instance: Dict[str, Any]) -> np.ndarray:
    coords = node_coordinates(instance)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)


def iter_customer_triplets(routes: Iterable[Iterable[int]], num_customers: int):
    customer_start = 1
    customer_end = 1 + int(num_customers)
    for route in routes or []:
        route = [int(x) for x in route]
        if len(route) < 3:
            continue
        for pos in range(1, len(route) - 1):
            node = int(route[pos])
            if customer_start <= node < customer_end:
                yield int(route[pos - 1]), node, int(route[pos + 1])


def compute_route_neighbor_delta(
    instance: Dict[str, Any],
    routes: Iterable[Iterable[int]],
) -> Dict[str, Any]:
    n_customers = int(len(instance.get("customers", [])))
    dist = distance_matrix(instance)
    delta = np.zeros(n_customers, dtype=np.float32)
    seen = np.zeros(n_customers, dtype=np.bool_)

    for prev_node, customer_node, next_node in iter_customer_triplets(routes, n_customers):
        customer_idx = customer_node - 1
        marginal = (
            float(dist[prev_node, customer_node])
            + float(dist[customer_node, next_node])
            - float(dist[prev_node, next_node])
        )
        delta[customer_idx] = max(0.0, marginal)
        seen[customer_idx] = True

    missing = np.flatnonzero(~seen)
    if missing.size > 0:
        customer_nodes = missing + 1
        delta[missing] = 2.0 * dist[0, customer_nodes]

    scale = repair_dist_scale(instance)
    delta_norm = (delta / np.float32(scale)).astype(np.float32)

    return {
        "expert_delta_j": delta.astype(np.float32).tolist(),
        "expert_delta_j_norm": delta_norm.tolist(),
        "expert_delta_total": float(delta.sum()),
        "expert_delta_total_norm": float(delta_norm.sum()),
        "expert_delta_missing_count": int(missing.size),
        "expert_delta_scale": float(scale),
        "expert_delta_version": LEGACY_DELTA_VERSION,
    }


def _normalise_routes_for_solver(
    routes: Iterable[Iterable[int]],
) -> List[List[int]]:
    clean_routes: List[List[int]] = []
    for route in routes or []:
        try:
            r = [int(x) for x in route]
        except Exception:
            continue
        if not r:
            continue
        if r[0] != 0:
            r = [0] + r
        if r[-1] != 0:
            r = r + [0]
        compact = [r[0]]
        for node in r[1:]:
            if node == 0 and compact[-1] == 0:
                continue
            compact.append(node)
        if len(compact) >= 2:
            clean_routes.append(compact)
    return clean_routes


def _fallback_single_customer_delta(
    instance: Dict[str, Any],
    customer_node: int,
) -> float:
    dist = distance_matrix(instance)
    idx = int(customer_node)
    if 0 <= idx < dist.shape[0]:
        return float(2.0 * dist[0, idx])
    return 0.0


def compute_best_feasible_reinsert_delta(
    instance: Dict[str, Any],
    routes: Iterable[Iterable[int]],
) -> Dict[str, Any]:
    """Estimate each customer marginal cost by feasible re-insertion.

    For customer j, remove j from the expert solution, then ask the ALNS
    solver for the cheapest feasible position to insert j again. This is
    more faithful than the local edge delta dist(prev,j)+dist(j,next)-dist(prev,next)
    because the insertion helper can add charging stations and respects
    route-level feasibility checks.
    """
    n_customers = int(len(instance.get("customers", [])))
    delta = np.zeros(n_customers, dtype=np.float32)
    used_fallback = np.zeros(n_customers, dtype=np.bool_)

    solver_format = "tensor" if "env" in instance and "vehicle" not in instance else None
    solver = ALNS_Solver(instance, seed=0, format=solver_format)
    base_routes = _normalise_routes_for_solver(routes)
    try:
        base_routes = solver._cleanup_solution(base_routes)
    except Exception:
        pass

    customer_start = 1
    customer_end = 1 + n_customers
    routed_customers = {
        int(node)
        for route in base_routes
        for node in route
        if customer_start <= int(node) < customer_end
    }

    for customer_idx in range(n_customers):
        customer_node = customer_idx + 1
        try:
            if customer_node in routed_customers:
                removed_routes = solver._remove_customers_with_mode(
                    deepcopy(base_routes),
                    [customer_node],
                    mode="customer_only",
                )
            else:
                removed_routes = deepcopy(base_routes)

            best = solver._best_customer_insertion(
                removed_routes,
                customer_node,
                mode="distance",
            )
            if best is not None:
                marginal = float(best[2])
            else:
                single_route = solver._make_single_customer_route(customer_node)
                if single_route is not None:
                    marginal = float(solver._route_distance(single_route))
                else:
                    marginal = _fallback_single_customer_delta(instance, customer_node)
                    used_fallback[customer_idx] = True
        except Exception:
            marginal = _fallback_single_customer_delta(instance, customer_node)
            used_fallback[customer_idx] = True

        delta[customer_idx] = max(0.0, marginal)

    scale = repair_dist_scale(instance)
    delta_norm = (delta / np.float32(scale)).astype(np.float32)

    return {
        "expert_delta_j": delta.astype(np.float32).tolist(),
        "expert_delta_j_norm": delta_norm.tolist(),
        "expert_delta_total": float(delta.sum()),
        "expert_delta_total_norm": float(delta_norm.sum()),
        "expert_delta_missing_count": int(n_customers - len(routed_customers)),
        "expert_delta_fallback_count": int(used_fallback.sum()),
        "expert_delta_scale": float(scale),
        "expert_delta_version": DELTA_VERSION,
    }


def attach_expert_delta_to_state(
    state: Dict[str, Any],
    instance: Dict[str, Any],
    routes_key: str = "best_routes",
) -> Dict[str, Any]:
    routes = state.get(routes_key, None)
    if routes is None:
        routes = state.get("current_routes", None)
    delta_payload = compute_best_feasible_reinsert_delta(instance, routes)
    state.update(delta_payload)
    return state


def backfill_progress_delta(
    progress: Dict[str, Dict[str, Any]],
    instance_data,
    overwrite: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    instance_index = build_instance_index(instance_data)
    updated = deepcopy(progress)
    rows: List[Dict[str, Any]] = []

    for key, entry in updated.items():
        if not isinstance(entry, dict):
            rows.append({"key": key, "status": "invalid_entry"})
            continue
        if (not overwrite) and "expert_delta_j_norm" in entry:
            rows.append({"key": key, "status": "exists"})
            continue

        instance, matched_key = find_instance_for_progress_entry(key, entry, instance_index)
        if instance is None:
            rows.append({"key": key, "status": "missing_instance"})
            continue

        try:
            attach_expert_delta_to_state(entry, instance)
            rows.append(
                {
                    "key": key,
                    "status": "updated",
                    "matched_key": matched_key,
                    "delta_total": entry.get("expert_delta_total"),
                    "delta_total_norm": entry.get("expert_delta_total_norm"),
                    "missing_count": entry.get("expert_delta_missing_count"),
                }
            )
        except Exception as exc:
            rows.append({"key": key, "status": "error", "error": str(exc)})

    return updated, rows


def _backfill_one_delta_task(task):
    key, entry, instance, matched_key = task
    entry_out = deepcopy(entry)
    try:
        attach_expert_delta_to_state(entry_out, instance)
        row = {
            "key": key,
            "status": "updated",
            "matched_key": matched_key,
            "delta_total": entry_out.get("expert_delta_total"),
            "delta_total_norm": entry_out.get("expert_delta_total_norm"),
            "missing_count": entry_out.get("expert_delta_missing_count"),
            "fallback_count": entry_out.get("expert_delta_fallback_count"),
            "delta_version": entry_out.get("expert_delta_version"),
        }
    except Exception as exc:
        row = {"key": key, "status": "error", "error": str(exc)}
    return key, entry_out, row


def backfill_progress_delta_parallel(
    progress: Dict[str, Dict[str, Any]],
    instance_data,
    overwrite: bool = False,
    num_workers: int = 1,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if int(num_workers) <= 1:
        return backfill_progress_delta(progress, instance_data, overwrite=overwrite)

    instance_index = build_instance_index(instance_data)
    updated = deepcopy(progress)
    rows: List[Dict[str, Any]] = []
    tasks = []

    for key, entry in updated.items():
        if not isinstance(entry, dict):
            rows.append({"key": key, "status": "invalid_entry"})
            continue
        if (not overwrite) and "expert_delta_j_norm" in entry:
            rows.append({"key": key, "status": "exists"})
            continue

        instance, matched_key = find_instance_for_progress_entry(key, entry, instance_index)
        if instance is None:
            rows.append({"key": key, "status": "missing_instance"})
            continue
        tasks.append((key, entry, instance, matched_key))

    if not tasks:
        return updated, rows

    finished = 0
    with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
        futures = [executor.submit(_backfill_one_delta_task, task) for task in tasks]
        for future in as_completed(futures):
            key, entry_out, row = future.result()
            if row.get("status") == "updated":
                updated[key] = entry_out
            rows.append(row)
            finished += 1
            if finished == 1 or finished % 200 == 0 or finished == len(tasks):
                print(
                    f"[DeltaBackfill] processed {finished}/{len(tasks)} "
                    f"with workers={int(num_workers)}",
                    flush=True,
                )

    return updated, rows


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: str):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def _write_summary(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill expert marginal delta_j fields into ALNS progress pickle."
    )
    parser.add_argument(
        "--buffer-dir",
        type=str,
        default="/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer",
    )
    parser.add_argument("--progress-name", type=str, default="buffer_progress.pkl")
    parser.add_argument("--instance-pickle-name", type=str, default="evrptw_50C_12R.pkl")
    parser.add_argument("--output-progress-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional path for a CSV summary. Defaults under buffer/progress.",
    )
    args = parser.parse_args()

    progress_path = os.path.join(args.buffer_dir, "progress", args.progress_name)
    instance_path = os.path.join(args.buffer_dir, "pickle", args.instance_pickle_name)
    output_name = args.output_progress_name or args.progress_name
    output_path = os.path.join(args.buffer_dir, "progress", output_name)

    progress = _load_pickle(progress_path)
    instance_data = _load_pickle(instance_path)
    if not isinstance(progress, dict):
        raise ValueError(f"Progress must be dict, got {type(progress)}")

    updated, rows = backfill_progress_delta_parallel(
        progress=progress,
        instance_data=instance_data,
        overwrite=bool(args.overwrite),
        num_workers=int(args.num_workers),
    )

    if output_path == progress_path and not args.no_backup:
        backup_path = progress_path + ".pre_delta.bak"
        if not os.path.exists(backup_path):
            shutil.copy2(progress_path, backup_path)
            print(f"[DeltaBackfill] backup -> {backup_path}")
        else:
            print(f"[DeltaBackfill] backup exists -> {backup_path}")

    _save_pickle(updated, output_path)

    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = os.path.join(args.buffer_dir, "progress", "delta_backfill_summary.csv")
    _write_summary(rows, summary_csv)

    counts = {}
    for row in rows:
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
    print(f"[DeltaBackfill] saved -> {output_path}")
    print(f"[DeltaBackfill] summary -> {summary_csv}")
    print(f"[DeltaBackfill] counts={counts}")


if __name__ == "__main__":
    main()
