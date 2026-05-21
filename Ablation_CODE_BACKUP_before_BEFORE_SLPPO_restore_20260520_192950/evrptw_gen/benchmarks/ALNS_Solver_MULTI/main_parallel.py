import time
import argparse
import os
import pickle
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from solver import ALNS_Solver
from offline_delta import attach_expert_delta_to_state
from utils.helpers import set_random_seed
from utils.load_instances import load_instance


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


def process_one_instance(task):
    file, args_dict, prev_state = task

    instance_path = os.path.join(args_dict["instance_path"], file)
    base_seed = int(args_dict["seed"])
    delta_iters = int(args_dict["delta_iters"])
    resume = bool(args_dict["resume"])
    skip_frozen = bool(args_dict["skip_frozen"])
    search_profile = str(args_dict.get("search_profile", "") or "")

    instance = load_instance(instance_path)
    instance_id = instance.get("instance_id", os.path.splitext(file)[0])

    # 每个 worker 自己设 seed
    set_random_seed(base_seed)

    # optionally skip frozen
    if skip_frozen and prev_state is not None:
        prev_cur_iter = int(prev_state.get("cur_iter", 1))
        tmp_solver = ALNS_Solver(instance, seed=base_seed, search_profile=search_profile)
        prev_max_iters = int(prev_state.get("max_iters", getattr(tmp_solver, "max_iters", 10**9)))
        if prev_cur_iter > prev_max_iters:
            return {
                "file": file,
                "skipped": True,
                "skip_reason": f"already reached max_iters ({prev_cur_iter-1}/{prev_max_iters})"
            }

    checkpoint = None
    resume_flag = False

    if resume and prev_state is not None:
        checkpoint = prev_state
        resume_flag = True

    solver_seed = base_seed
    if checkpoint is not None:
        solver_seed = base_seed + int(checkpoint.get("cur_iter", 1))

    solver = ALNS_Solver(
        instance,
        seed=solver_seed,
        checkpoint=checkpoint,
        search_profile=search_profile,
    )

    start_time = time.time()
    if resume_flag:
        solution = solver.solve(delta_iters=delta_iters, resume=True)
    else:
        solution = solver.solve(delta_iters=delta_iters, resume=False)
    end_time = time.time()

    elapsed_time = float(end_time - start_time)
    visited_all = bool(all(solver.visited))
    objective_value = float(getattr(solver, "global_value", float("nan")))
    cur_iter = int(getattr(solver, "cur_iter", -1))
    max_iters = int(getattr(solver, "max_iters", -1))
    is_frozen = bool(cur_iter > max_iters)

    prev_selected_count = int(prev_state.get("selected_count", 0)) if prev_state is not None else 0
    prev_best_obj = float(prev_state.get("global_value", float("inf"))) if prev_state is not None else float("inf")

    if prev_state is None or not (prev_best_obj < float("inf")):
        last_improvement = None
    else:
        if objective_value < prev_best_obj - 1e-9:
            last_improvement = float(prev_best_obj - objective_value)
        else:
            last_improvement = 0.0

    new_state = solver.get_checkpoint()
    attach_expert_delta_to_state(new_state, instance)
    new_state["instance_id"] = instance_id
    new_state["file"] = file
    new_state["selected_count"] = prev_selected_count + 1
    new_state["last_improvement"] = last_improvement
    new_state["elapsed_time_s_last_run"] = elapsed_time
    new_state["visited_all"] = visited_all
    new_state["is_frozen"] = is_frozen
    new_state["max_iters"] = max_iters

    fleet_size = len(solution)

    summary_row = {
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
    }

    route_rows = []
    for r_idx, route in enumerate(solution):
        route_rows.append({
            "instance_id": instance_id,
            "file": file,
            "route_idx": r_idx,
            "route_json": json.dumps(route, ensure_ascii=False, default=str),
            "route_str": str(route),
        })

    return {
        "file": file,
        "skipped": False,
        "visited_all": visited_all,
        "new_state": new_state,
        "summary_row": summary_row,
        "route_rows": route_rows,
        "elapsed_time": elapsed_time,
        "objective_value": objective_value,
        "cur_iter": cur_iter,
        "max_iters": max_iters,
        "is_frozen": is_frozen,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run resumable ALNS solver on a directory of EVRPTW instances."
    )

    parser.add_argument(
        "--instance_path",
        type=str,
        required=False,
        default="/data/Maojie/Github2/EVRP-TW-D-B/dataset4/unanchored/Cus_50/buffer/solomon",
        help="Path to the instance directory (.txt files)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--save_log_path",
        type=str,
        default="./logs_Cus50_3000_32mul_4k_instance",
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
        default=3000,
        help="How many additional ALNS iterations to run for each selected instance this time."
    )
    parser.add_argument(
        "--search_profile",
        type=str,
        default="",
        choices=["", "warm_1k", "online_warm_1k"],
        help="Optional ALNS search profile. warm_1k makes all operator families active within 1k iters."
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of worker processes."
    )
    parser.add_argument(
        "--progress_save_every",
        "--progress-save-every",
        type=int,
        default=10,
        help=(
            "Atomically save progress_path after this many completed instances. "
            "Use 1 for maximum crash safety; larger values reduce pickle write overhead."
        )
    )

    args = parser.parse_args()
    if args.progress_save_every <= 0:
        raise ValueError("--progress_save_every must be positive.")

    set_random_seed(args.seed)

    os.makedirs(args.save_log_path, exist_ok=True)
    progress_dir = os.path.dirname(args.progress_path)
    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)

    progress_dict = load_progress(args.progress_path)

    summary_records = []
    route_records = []
    failed_files = []

    files = [f for f in os.listdir(args.instance_path) if f.endswith(".txt")]
    files.sort()

    if args.max_instances is not None:
        files = files[:args.max_instances]

    args_dict = {
        "instance_path": args.instance_path,
        "seed": args.seed,
        "delta_iters": args.delta_iters,
        "resume": args.resume,
        "skip_frozen": args.skip_frozen,
        "search_profile": args.search_profile,
    }

    tasks = [(file, args_dict, progress_dict.get(file, None)) for file in files]

    completed_since_save = 0
    saved_updates = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_one_instance, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing instances"):
            result = future.result()

            file = result["file"]

            if result["skipped"]:
                print(f"[Skip Frozen] {file} {result['skip_reason']}")
                continue

            progress_dict[file] = result["new_state"]
            completed_since_save += 1
            summary_records.append(result["summary_row"])
            route_records.extend(result["route_rows"])

            if not result["visited_all"]:
                failed_files.append(file)

            print(
                f"Instance file: {file} | "
                f"Elapsed time: {result['elapsed_time']:.2f}s | "
                f"best_obj={result['objective_value']:.6f} | "
                f"cur_iter={result['cur_iter']} | "
                f"max_iters={result['max_iters']} | "
                f"frozen={result['is_frozen']}"
            )

            if completed_since_save >= args.progress_save_every:
                save_progress(progress_dict, args.progress_path)
                saved_updates += completed_since_save
                print(
                    f"[ProgressSave] saved {saved_updates} completed instance updates -> {args.progress_path}",
                    flush=True,
                )
                completed_since_save = 0

    save_progress(progress_dict, args.progress_path)

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
