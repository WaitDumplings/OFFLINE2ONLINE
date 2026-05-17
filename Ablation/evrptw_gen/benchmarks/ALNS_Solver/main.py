import time
import argparse
import os
import pickle
import json
import pandas as pd

from tqdm import tqdm

from solver import ALNS_Solver
from utils.helpers import set_random_seed
from utils.load_instances import load_instance

def get_file_sort_key(filename):
    stem = os.path.splitext(filename)[0]
    return int(stem.split("_")[2])

def load_progress(progress_path: str):
    if os.path.exists(progress_path):
        with open(progress_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Progress file at {progress_path} is not a dict.")
        return data
    return {}


def save_progress(progress_dict: dict, progress_path: str):
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(progress_dict, f)
    os.replace(tmp_path, progress_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run resumable ALNS solver on a directory of EVRPTW instances."
    )

    parser.add_argument(
        "--instance_path",
        type=str,
        required=False,
        default="../../../dataset/unanchored/Cus_50/solomon/",
        help="Path to the instance directory (.txt files)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--save_log_path",
        type=str,
        default="./logs_Cus50_200iter",
        help="Directory to save summary / route logs."
    )
    parser.add_argument(
        "--progress_path",
        type=str,
        default="./buffer_progress.pkl",
        help="Single pickle file storing progress dict for all instances."
    )
    parser.add_argument(
        "--delta_iters",
        type=int,
        default=200,
        help="How many additional ALNS iterations to run for each selected instance this time."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from progress_path if the instance already exists there."
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Optional limit on number of instances to process (for debugging)."
    )
    parser.add_argument(
        "--skip_frozen",
        action="store_true",
        help="Skip instances whose cur_iter already exceeded max_iters."
    )

    args = parser.parse_args()

    set_random_seed(args.seed)

    os.makedirs(args.save_log_path, exist_ok=True)
    progress_dir = os.path.dirname(args.progress_path)
    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)

    # load one global progress pickle
    progress_dict = load_progress(args.progress_path)

    summary_records = []
    route_records = []
    failed_files = []

    files = [f for f in os.listdir(args.instance_path) if f.endswith(".txt")]
    files.sort()

    if args.max_instances is not None:
        files = files[:args.max_instances]

    for file in tqdm(files, desc="Processing instances"):
        instance_path = os.path.join(args.instance_path, file)
        instance = load_instance(instance_path)

        instance_id = instance.get("instance_id", os.path.splitext(file)[0])
        prev_state = progress_dict.get(file, None)

        # optionally skip frozen instances
        if args.skip_frozen and prev_state is not None:
            prev_cur_iter = int(prev_state.get("cur_iter", 1))
            prev_max_iters = int(prev_state.get("max_iters", getattr(ALNS_Solver(instance), "max_iters", 10**9)))
            if prev_cur_iter > prev_max_iters:
                print(f"[Skip Frozen] {file} already reached max_iters ({prev_cur_iter-1}/{prev_max_iters})")
                continue

        checkpoint = None
        resume_flag = False

        if args.resume and prev_state is not None:
            checkpoint = prev_state
            resume_flag = True

        # slightly perturb seed on resumed runs for more search diversity
        solver_seed = args.seed
        if checkpoint is not None:
            solver_seed = args.seed + int(checkpoint.get("cur_iter", 1))

        solver = ALNS_Solver(instance, seed=solver_seed, checkpoint=checkpoint)

        print(f"Instance file: {file}")
        start_time = time.time()

        if resume_flag:
            solution = solver.solve(delta_iters=args.delta_iters, resume=True)
        else:
            solution = solver.solve(delta_iters=args.delta_iters, resume=False)

        end_time = time.time()
        elapsed_time = float(end_time - start_time)

        visited_all = bool(all(solver.visited))
        if not visited_all:
            failed_files.append(file)

        objective_value = float(getattr(solver, "global_value", float("nan")))
        cur_iter = int(getattr(solver, "cur_iter", -1))
        max_iters = int(getattr(solver, "max_iters", -1))
        is_frozen = bool(cur_iter > max_iters)

        # read previous metadata if exists
        prev_selected_count = int(prev_state.get("selected_count", 0)) if prev_state is not None else 0
        prev_best_obj = float(prev_state.get("global_value", float("inf"))) if prev_state is not None else float("inf")

        if prev_state is None or not (prev_best_obj < float("inf")):
            last_improvement = None
        else:
            if objective_value < prev_best_obj - 1e-9:
                last_improvement = float(prev_best_obj - objective_value)
            else:
                last_improvement = 0.0

        # checkpoint state from solver
        new_state = solver.get_checkpoint()

        # add buffer-level metadata
        new_state["instance_id"] = instance_id
        new_state["file"] = file
        new_state["selected_count"] = prev_selected_count + 1
        new_state["last_improvement"] = last_improvement
        new_state["elapsed_time_s_last_run"] = elapsed_time
        new_state["visited_all"] = visited_all
        new_state["is_frozen"] = is_frozen
        new_state["max_iters"] = max_iters

        progress_dict[file] = new_state

        fleet_size = len(solution)

        # summary row
        summary_records.append({
            "instance_id": instance_id,
            "file": file,
            "fleet_size": fleet_size,
            "objective_value": objective_value,
            "elapsed_time_s": elapsed_time,
            "visited_all": visited_all,
            "unvisited_count": int(len(solver.visited) - sum(bool(x) for x in solver.visited)),
            "cur_iter": cur_iter,
            "max_iters": max_iters,
            "selected_count": int(new_state["selected_count"]),
            "last_improvement": new_state["last_improvement"],
            "is_frozen": is_frozen,
            "routes_json": json.dumps(solution, ensure_ascii=False, default=str),
        })

        # route rows
        for r_idx, route in enumerate(solution):
            route_records.append({
                "instance_id": instance_id,
                "file": file,
                "route_idx": r_idx,
                "route_json": json.dumps(route, ensure_ascii=False, default=str),
                "route_str": str(route),
            })

        print(
            f"Elapsed time: {elapsed_time:.2f} seconds | "
            f"best_obj={objective_value:.6f} | "
            f"cur_iter={cur_iter} | "
            f"max_iters={max_iters} | "
            f"frozen={is_frozen}"
        )

    # save one global progress pickle
    save_progress(progress_dict, args.progress_path)

    # logs
    df_summary = pd.DataFrame(summary_records)
    df_routes = pd.DataFrame(route_records)

    summary_csv = os.path.join(args.save_log_path, "alns_summary.csv")
    routes_csv = os.path.join(args.save_log_path, "alns_routes.csv")

    df_summary.to_csv(summary_csv, index=False)
    df_routes.to_csv(routes_csv, index=False)

    try:
        summary_parquet = os.path.join(args.save_log_path, "alns_summary.parquet")
        routes_parquet = os.path.join(args.save_log_path, "alns_routes.parquet")
        df_summary.to_parquet(summary_parquet, index=False)
        df_routes.to_parquet(routes_parquet, index=False)
    except Exception as e:
        print(f"[WARN] Parquet save skipped: {e}")

    failed_path = os.path.join(args.save_log_path, "unvisited_instances.txt")
    with open(failed_path, "w") as f:
        for x in failed_files:
            f.write(x + "\n")

    print(f"Saved summary -> {summary_csv}")
    print(f"Saved routes  -> {routes_csv}")
    print(f"Saved failed list -> {failed_path}")
    print(f"Saved progress pickle -> {args.progress_path}")


if __name__ == "__main__":
    main()