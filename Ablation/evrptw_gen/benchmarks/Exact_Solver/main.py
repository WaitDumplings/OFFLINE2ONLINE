from data_parser import parse_file
from EVRP_Graph import Graph_EVRP_TW
from EVRP_Solver import EVRP_TW_Gurobi_Solver

import argparse
import os
import time
import json
import traceback
import pandas as pd
from gurobipy import GRB


def safe_json(obj):
    """Best-effort JSON serialization."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)

def is_feasible(Solver):
    return True if Solver.model.Status != GRB.INFEASIBLE else False

def extract_instance_id(parameters, file):
    """
    Try to get instance id from parsed metadata; fallback to filename stem.
    Adjust this if your parser uses a different key.
    """
    for k in ["instance_id", "InstanceID", "id", "name"]:
        if isinstance(parameters, dict) and k in parameters:
            return parameters[k]
    return os.path.splitext(file)[0]


def extract_solver_outputs(Solver):
    """
    Best-effort extraction of objective, fleet size, and routes.
    You should customize these 3-5 lines if you know your solver fields.
    """
    objective = None
    routes = None
    status = None

    # status (optional)
    for attr in ["status", "solve_status", "model_status", "gurobi_status"]:
        if hasattr(Solver, attr):
            status = getattr(Solver, attr)
            break
    # sometimes stored in model.Status
    if status is None and hasattr(Solver, "model") and hasattr(Solver.model, "Status"):
        status = Solver.model.Status

    # objective
    for attr in ["obj_val", "objective_value", "objective", "Optimal_Value", "opt_val", "best_obj", "ObjVal"]:
        if hasattr(Solver, attr):
            try:
                objective = float(getattr(Solver, attr))
                break
            except Exception:
                objective = getattr(Solver, attr)
                break
    # common: Solver.model.ObjVal (Gurobi)
    if objective is None and hasattr(Solver, "model") and hasattr(Solver.model, "ObjVal"):
        try:
            objective = float(Solver.model.ObjVal)
        except Exception:
            objective = Solver.model.ObjVal


    # fleet size
    routes = Solver.get_routes() if is_feasible(Solver) else []
    fleet_size = len(routes)
    return objective, fleet_size, routes, status


def main(args):
    file_dir = args.file_dir
    os.makedirs(args.save_path, exist_ok=True)

    summary_records = []
    route_records = []

    for file in os.listdir(file_dir):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(file_dir, file)
        t0 = time.time()

        try:
            Depot_nodes, Customer_nodes, RS_nodes, parameters = parse_file(file_path)

            Graph = Graph_EVRP_TW(
                Depot_nodes, Customer_nodes, RS_nodes, parameters,
                RS_dummy_count=args.dummy
            )

            Solver = EVRP_TW_Gurobi_Solver(Graph)

            # solve
            Solver.solver()

            # optional printing (you can keep it)
            if args.print_results:
                Solver.print_results(Optimal_Value=True, DV_Info=False, Routes=True)

            t1 = time.time()

            instance_id = extract_instance_id(parameters, file)
            objective, fleet_size, routes, status = extract_solver_outputs(Solver)

            # summary row
            summary_records.append({
                "instance_id": instance_id,
                "file": file,
                "dummy": args.dummy,
                "status": status,
                "objective_value": objective,
                "fleet_size": fleet_size,
                "elapsed_time_s": float(t1 - t0),
                # keep a copy of routes in summary for quick grep; long table is routes.csv
                "routes_json": safe_json(routes) if routes is not None else None,
            })

            # routes long table
            if isinstance(routes, (list, tuple)):
                for ridx, route in enumerate(routes):
                    route_records.append({
                        "instance_id": instance_id,
                        "file": file,
                        "dummy": args.dummy,
                        "route_idx": ridx,
                        "route_json": safe_json(route),
                        "route_str": str(route),
                    })

        except Exception as e:
            t1 = time.time()
            # log failure as a summary row too
            summary_records.append({
                "instance_id": os.path.splitext(file)[0],
                "file": file,
                "dummy": args.dummy,
                "status": "ERROR",
                "objective_value": None,
                "fleet_size": None,
                "elapsed_time_s": float(t1 - t0),
                "routes_json": None,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc() if args.save_traceback else None,
            })
            if args.verbose:
                print(f"[ERROR] {file}: {e}")

    # save
    df_summary = pd.DataFrame(summary_records)
    df_routes = pd.DataFrame(route_records)

    summary_csv = os.path.join(args.save_path, "gurobi_summary.csv")
    routes_csv = os.path.join(args.save_path, "gurobi_routes.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_routes.to_csv(routes_csv, index=False)

    # parquet optional
    if args.save_parquet:
        try:
            df_summary.to_parquet(os.path.join(args.save_path, "gurobi_summary.parquet"), index=False)
            df_routes.to_parquet(os.path.join(args.save_path, "gurobi_routes.parquet"), index=False)
        except Exception as e:
            print(f"[WARN] Parquet save skipped: {e}")

    print(f"Saved summary -> {summary_csv}")
    print(f"Saved routes  -> {routes_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EVRP-TW solver (Gurobi) and collect results to Pandas.")

    parser.add_argument("--file_dir", required=False, default="../../../dataset/anchored/Cus_15/solomon/", type=str,
                        help="Directory containing .txt instances.")
    parser.add_argument("--dummy", type=int, default=3, help="Number of RS dummy nodes.")
    parser.add_argument("--save_path", type=str, default="./logs_gurobi", help="Where to save CSV/Parquet logs.")

    parser.add_argument("--print_results", action="store_true", help="Call Solver.print_results(...) for each instance.")
    parser.add_argument("--save_parquet", action="store_true", help="Also save parquet outputs (requires pyarrow/fastparquet).")
    parser.add_argument("--save_traceback", action="store_true", help="Save traceback into summary on failures.")
    parser.add_argument("--verbose", action="store_true", help="Print errors during run.")

    args = parser.parse_args()
    main(args)
