#!/usr/bin/env python3
"""Convert generated Solomon-style txt instances back to the DRL eval pickle format."""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evrptw_gen.benchmarks.Greedy_Heuristic_Solver.utils.load_instances import (
    load_instance,
)


NAME_RE = re.compile(
    r"solomon_dataset_(?P<idx>\d+)_(?P<itype>[A-Za-z]+)_(?P<tw>narrow|wide)_(?P<stamp>\d+)\.txt$"
)


def _name_key(path: Path):
    match = NAME_RE.match(path.name)
    if match is None:
        return (10**12, path.name)
    return (int(match.group("idx")), path.name)


def _meta_float(meta, key, default):
    value = meta.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def convert_one(path: Path) -> dict:
    parsed = load_instance(str(path), drop_depot_station_duplicate=True)
    meta = parsed.get("meta", {})
    vehicle = parsed.get("vehicle", {})
    match = NAME_RE.match(path.name)

    depot_node = parsed["depot"]
    customers = parsed["customers"]
    stations = parsed["stations"]

    depot = np.asarray([[depot_node["x"], depot_node["y"]]], dtype=np.float32)
    cus_loc = np.asarray([[c["x"], c["y"]] for c in customers], dtype=np.float32)
    rs_loc = np.asarray([[s["x"], s["y"]] for s in stations], dtype=np.float32)
    demands = np.asarray([c["demand"] for c in customers], dtype=np.float32)
    tw = np.asarray([[c["ready"] * 60.0, c["due"] * 60.0] for c in customers], dtype=np.float32)
    service_time = np.asarray([c["service"] * 60.0 for c in customers], dtype=np.float32)

    cs_time_to_depot = parsed.get("cs2depot", None)
    if cs_time_to_depot is None:
        cs_time_to_depot = [0.0] * len(stations)
    cs_time_to_depot = np.asarray(cs_time_to_depot, dtype=np.float32)
    if cs_time_to_depot.shape[0] == len(stations) + 1:
        # Text files include a depot-co-located station line that load_instance drops.
        cs_time_to_depot = cs_time_to_depot[1:]

    instance_id = str(meta.get("instance_id", path.stem))
    file_name = str(meta.get("file", path.name))
    if not file_name.endswith(".txt"):
        file_name = path.name

    area_x = meta.get("instance_x_range", meta.get("instance x range", [0.0, 100.0]))
    area_y = meta.get("instance_y_range", meta.get("instance y range", [0.0, 100.0]))
    if isinstance(area_x, str):
        area_x = [0.0, 100.0]
    if isinstance(area_y, str):
        area_y = [0.0, 100.0]

    instance_type = str(meta.get("instance_type", match.group("itype") if match else "UNK"))
    time_window_type = str(match.group("tw") if match else "unknown")

    env = {
        "area_size": [list(area_x), list(area_y)],
        "dist_fun": "2-norm",
        "set_edge": False,
        "num_customers": int(len(customers)),
        "num_charging_stations": int(len(stations)),
        "num_cluster": int(float(meta.get("number_of_clusters", meta.get("number", 3)) or 3)),
        "speed": float(vehicle.get("v", 30.0)),
        "battery_capacity": float(vehicle.get("Q", 120.0)),
        "consumption_per_distance": float(vehicle.get("r", 0.5)),
        "loading_capacity": float(vehicle.get("C", 1.0)),
        "charging_speed": float(meta.get("gv", vehicle.get("gv", 1.0 / max(float(vehicle.get("g", 1.0)), 1e-8)))),
        "charging_efficiency": 1.0,
        "instance_startTime": 0.0,
        "instance_endTime": _meta_float(meta, "instance_endTime", 24.0) * 60.0,
        "working_startTime": _meta_float(meta, "working_startTime", 8.0) * 60.0,
        "working_endTime": _meta_float(meta, "working_endTime", 22.0) * 60.0,
        "instance_type": instance_type,
        "time_window_type": time_window_type,
        "service_time_type": str(meta.get("service_time_type", "unknown")),
        "demand_type": str(meta.get("demand_type", "unknown")),
        "cs_time_to_depot": cs_time_to_depot,
    }

    return {
        "env": env,
        "depot": depot,
        "customers": cus_loc,
        "charging_stations": rs_loc,
        "demands": demands,
        "tw": tw,
        "service_time": service_time,
        "id": instance_id,
        "file": file_name,
        "instance_id": instance_id,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solomon-dir", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    solomon_dir = Path(args.solomon_dir)
    files = sorted(solomon_dir.glob("*.txt"), key=_name_key)
    if int(args.limit) > 0:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"No .txt files found under {solomon_dir}")

    instances = [convert_one(path) for path in files]
    out = Path(args.output_pkl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(instances, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[SolomonToPickle] converted={len(instances)} -> {out}")
    first = instances[0]
    print(
        "[SolomonToPickle] first="
        f"{first['file']} | C={first['customers'].shape[0]} | "
        f"RS={first['charging_stations'].shape[0]} | "
        f"type={first['env'].get('instance_type')} | tw={first['env'].get('time_window_type')}"
    )


if __name__ == "__main__":
    main()
