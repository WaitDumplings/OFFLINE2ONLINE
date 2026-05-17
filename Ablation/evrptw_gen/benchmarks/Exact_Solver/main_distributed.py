# main_distributed.py
from data_parser import parse_file
from EVRP_Graph import Graph_EVRP_TW
from EVRP_Solver import EVRP_TW_Gurobi_Solver

import argparse
import os
import pandas as pd
from gurobipy import GRB


def build_capture_obj_at_times_callback(times_sec):
    """
    Record incumbent objective at specified time thresholds (first callback after crossing each).
    Gurobi callback signature: cb(model, where)

    times_sec: list like [60, 300]
    """
    times = sorted([float(t) for t in times_sec])

    cap = {
        t: {"obj": None, "captured_at": None, "solcnt": None, "mode": None}
        for t in times
    }
    recorded = {t: False for t in times}

    def cb(model, where):
        if where != GRB.Callback.MIP:
            return

        # fast exit if all recorded
        if all(recorded.values()):
            return

        try:
            runtime = float(model.cbGet(GRB.Callback.RUNTIME))
        except Exception:
            return

        # Only do work if we've passed at least one unrecorded threshold
        if runtime < times[0]:
            return

        # incumbent info
        try:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        except Exception:
            objbst = None
        try:
            solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
        except Exception:
            solcnt = None

        feasible_now = (
            solcnt is not None and solcnt > 0 and
            objbst is not None and objbst < GRB.INFINITY / 2
        )

        # record each threshold at first callback after crossing
        for t in times:
            if recorded[t]:
                continue
            if runtime >= t:
                if feasible_now:
                    cap[t]["obj"] = float(objbst)
                    cap[t]["mode"] = f"incumbent_at_{int(t)}s"
                else:
                    cap[t]["obj"] = None
                    cap[t]["mode"] = f"no_feasible_at_{int(t)}s"
                cap[t]["captured_at"] = runtime
                cap[t]["solcnt"] = int(solcnt) if solcnt is not None else None
                recorded[t] = True

    return cb, cap


def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    all_files = sorted([f for f in os.listdir(args.file_dir) if f.endswith(".txt")])
    start_idx = max(0, args.start_idx)
    end_idx = args.end_idx if args.end_idx != -1 else len(all_files)
    end_idx = min(end_idx, len(all_files))
    files = all_files[start_idx:end_idx]

    print(f"[INFO] Total .txt files: {len(all_files)}")
    print(f"[INFO] Processing slice: idx [{start_idx}:{end_idx}) -> {len(files)} files")
    print("[INFO] TimeLimit=15min (900s). Capture incumbent at 1min (60s) and 5min (300s). obj_15min=final incumbent.")

    time_limit = args.time_limit
    TIME_LIMIT_S = 15 * 60  # 900s
    T1 = 1 * 60             # 60s
    T2 = 5 * 60             # 300s

    rows = []

    for local_idx, file in enumerate(files):
        global_idx = start_idx + local_idx
        file_path = os.path.join(args.file_dir, file)

        try:
            Depot_nodes, Customer_nodes, RS_nodes, parameters = parse_file(file_path)

            Graph = Graph_EVRP_TW(
                Depot_nodes, Customer_nodes, RS_nodes, parameters,
                RS_dummy_count=args.dummy
            )

            Solver = EVRP_TW_Gurobi_Solver(Graph, time_limit=TIME_LIMIT_S)

            cb, cap = build_capture_obj_at_times_callback([T1, T2])
            Solver.model.optimize(cb)

            model = Solver.model

            # final incumbent at termination -> treat as 15min result
            obj_15min = None
            solcnt = None
            runtime = None
            try:
                solcnt = int(model.SolCount)
            except Exception:
                solcnt = None
            try:
                runtime = float(model.Runtime)
            except Exception:
                runtime = None

            if solcnt is not None and solcnt > 0:
                try:
                    obj_15min = float(model.ObjVal)
                except Exception:
                    obj_15min = None

            # 1min/5min from callback
            obj_1min = cap[float(T1)]["obj"]
            obj_5min = cap[float(T2)]["obj"]

            # If finished before 1min/5min, callback may not have fired -> fill with final incumbent if feasible
            if (obj_15min is not None) and (runtime is not None):
                if (runtime < T1) and (obj_1min is None):
                    obj_1min = obj_15min
                if (runtime < T2) and (obj_5min is None):
                    obj_5min = obj_15min

            rows.append({
                "file": file,
                "obj_1min": obj_1min,
                "obj_5min": obj_5min,
                "obj_15min": obj_15min,
            })

            if args.verbose:
                print(f"[OK] idx={global_idx} file={file} obj_1min={obj_1min} obj_5min={obj_5min} obj_15min={obj_15min}")

        except Exception as e:
            rows.append({
                "file": file,
                "obj_1min": None,
                "obj_5min": None,
                "obj_15min": None,
            })
            if args.verbose:
                print(f"[ERROR] idx={global_idx} file={file}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)

    tag = f"idx_{start_idx}_{end_idx}"
    out_csv = os.path.join(args.save_path, f"gurobi_anytime_{tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed EVRP-TW Gurobi runner: capture incumbent obj at 1min & 5min; obj_15min=final incumbent (TimeLimit=15min)."
    )
    parser.add_argument("--file_dir", required=False, default="../../../dataset/anchored/Cus_100/solomon/", type=str,
                        help="Directory containing .txt instances.")
    parser.add_argument("--dummy", type=int, default=10, help="Number of RS dummy nodes.")
    parser.add_argument("--save_path", type=str, default="./logs_gurobi_0_20", help="Where to save CSV outputs.")
    parser.add_argument("--time_limit", type=int, default=15, help="Time limit per instance in minutes.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (inclusive) in sorted file list.")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index (exclusive) in sorted file list.")
    parser.add_argument("--verbose", action="store_true", help="Print per-instance progress.")

    args = parser.parse_args()
    main(args)
