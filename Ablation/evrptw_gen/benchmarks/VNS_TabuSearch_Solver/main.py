import time
import argparse
import os
import json
import pandas as pd

from solver import VNSTSolver
from utils.helpers import plot_solution, set_random_seed
from utils.load_instances import load_instance


def _route_to_ids(route):
    """Route -> [node_id, ...]"""
    if hasattr(route, "nodes"):
        return [getattr(n, "id", str(n)) for n in route.nodes]
    if isinstance(route, (list, tuple)):
        return [getattr(x, "id", str(x)) for x in route]
    return [str(route)]


def _solution_to_routes_ids(solution):
    """solution: list[Route] -> list[list[node_id]]"""
    return [_route_to_ids(r) for r in solution]


def _served_all_customers(instance, solution):
    """
    served_all, unserved_count
    VNST: node.type == 'c' means customer
    """
    all_customers = {c.id for c in instance.customers}
    served = set()
    for route in solution:
        nodes = route.nodes if hasattr(route, "nodes") else route
        for n in nodes:
            if getattr(n, "type", None) == "c":
                served.add(n.id)
    unserved = all_customers - served
    return (served == all_customers), len(unserved)


def main():
    parser = argparse.ArgumentParser(description="Run VNSTSolver on a directory of VRPTW instances (VNS/TS) and save logs.")
    parser.add_argument(
        "--instance_dir",
        type=str,
        default="../../../dataset/anchored/Cus_5/solomon/",
        required=False,
        help="Path to the instance directory (contains .txt files)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)"
    )
    parser.add_argument(
        "--predefine_route_number",
        type=int,
        default=2,
        help="Predefined number of routes for VNSTSolver (default: 2)"
    )
    parser.add_argument(
        "--save_log_path",
        type=str,
        default="./logs_vnst",
        help="Directory to save csv/parquet logs"
    )
    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    instance_dir = args.instance_dir
    os.makedirs(args.save_log_path, exist_ok=True)

    summary_records = []
    route_records = []
    failed_files = []
    exception_files = []

    for file in os.listdir(instance_dir):
        if not file.endswith(".txt"):
            continue

        instance_path = os.path.join(instance_dir, file)

        try:
            # Load the instance
            instance = load_instance(instance_path)

            # Initialize solver
            solver = VNSTSolver(instance, predefine_route_number=args.predefine_route_number)

            # Print instance info
            print(f"Instance file: {file}")

            # Solve and measure runtime
            start_time = time.time()
            solution = solver.solve()
            end_time = time.time()

            # Basic stats
            fleet_size = int(len(solution))
            objective_value = float(getattr(solver, "global_value", float("nan")))
            elapsed_time = float(end_time - start_time)

            served_all, unserved_count = _served_all_customers(instance, solution)
            if not served_all:
                failed_files.append(file)

            # instance_id
            instance_id = getattr(instance, "instance_id", os.path.splitext(file)[0])

            # routes to JSON-friendly format
            routes_ids = _solution_to_routes_ids(solution)

            # Print solution details
            print(f"Optimal solution uses {fleet_size} vehicles")
            print(f"Total distance: {objective_value:.2f}")
            print(f"Elapsed time: {elapsed_time:.3f} s")
            print(f"served_all: {served_all}, unserved_count: {unserved_count}")

            # 1) Summary
            summary_records.append({
                "instance_id": instance_id,
                "file": file,
                "fleet_size": fleet_size,
                "objective_value": objective_value,
                "elapsed_time_s": elapsed_time,
                "served_all": bool(served_all),
                "unserved_count": int(unserved_count),
                "routes_ids_json": json.dumps(routes_ids, ensure_ascii=False),
            })

            # 2) Routes
            for r_idx, route in enumerate(solution):
                route_ids = _route_to_ids(route)
                route_records.append({
                    "instance_id": instance_id,
                    "file": file,
                    "route_idx": int(r_idx),
                    "route_len": int(len(route_ids)),
                    "route_ids_json": json.dumps(route_ids, ensure_ascii=False),
                    "route_str": " -> ".join(map(str, route_ids)),
                })

        except Exception as e:
            exception_files.append((file, repr(e)))
            print(f"[ERROR] {file}: {e}")

    # Save logs
    df_summary = pd.DataFrame(summary_records)
    df_routes = pd.DataFrame(route_records)

    summary_csv = os.path.join(args.save_log_path, "vnst_summary.csv")
    routes_csv  = os.path.join(args.save_log_path, "vnst_routes.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_routes.to_csv(routes_csv, index=False)

    # Optional parquet
    try:
        summary_parquet = os.path.join(args.save_log_path, "vnst_summary.parquet")
        routes_parquet  = os.path.join(args.save_log_path, "vnst_routes.parquet")
        df_summary.to_parquet(summary_parquet, index=False)
        df_routes.to_parquet(routes_parquet, index=False)
    except Exception as e:
        print(f"[WARN] Parquet save skipped: {e}")

    # Save failed list (not served all customers)
    failed_path = os.path.join(args.save_log_path, "unserved_instances.txt")
    with open(failed_path, "w") as f:
        for x in failed_files:
            f.write(x + "\n")

    # Save exception list
    exception_path = os.path.join(args.save_log_path, "exceptions.txt")
    with open(exception_path, "w") as f:
        for fname, err in exception_files:
            f.write(f"{fname}\t{err}\n")

    print(f"Saved summary -> {summary_csv}")
    print(f"Saved routes  -> {routes_csv}")
    print(f"Saved failed list -> {failed_path}")
    print(f"Saved exceptions  -> {exception_path}")

if __name__ == "__main__":
    main()
