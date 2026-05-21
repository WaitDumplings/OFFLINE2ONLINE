import time
import argparse
import os
import pickle
import json
import pandas as pd

from tqdm import tqdm

from solver import ALNS_Solver
from offline_delta import attach_expert_delta_to_state
from utils.helpers import set_random_seed
from utils.load_instances import load_instance


def load_progress(progress_path: str):
    if not os.path.exists(progress_path):
        raise FileNotFoundError(f"Progress file not found: {progress_path}")
    with open(progress_path, "rb") as f:
        progress = pickle.load(f)
    if not isinstance(progress, dict):
        raise ValueError(f"Progress file must be a dict, got {type(progress)}")
    return progress


def save_progress(progress: dict, progress_path: str):
    os.makedirs(os.path.dirname(progress_path), exist_ok=True) if os.path.dirname(progress_path) else None
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(progress, f)
    os.replace(tmp_path, progress_path)


def check_ckpt_match(file_name: str, ckpt_entry: dict):
    """
    Return (matched: bool, reason: str)
    """
    if not isinstance(ckpt_entry, dict):
        return False, "checkpoint entry is not a dict"

    ckpt_file = ckpt_entry.get("file", None)
    if ckpt_file is None:
        return False, "checkpoint missing key 'file'"

    if ckpt_file != file_name:
        return False, f"checkpoint file mismatch: ckpt={ckpt_file}, current={file_name}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Resume ALNS from existing progress.pkl and update each matched instance by delta_iters."
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=False,
        default="/data/Maojie/Github2/EVRP-TW-D-B2/dataset2/unanchored/Cus_50/buffer/solomon",
        help="Path to the instance dir",
    )
    parser.add_argument(
        "--progress_path",
        type=str,
        required=True,
        help="Path to the global progress pickle",
    )
    parser.add_argument(
        "--delta_iters",
        type=int,
        required=True,
        help="Additional iterations to run on top of current checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    parser.add_argument(
        "--save_log_path",
        type=str,
        default="./logs_resume",
        help="Path to save csv/parquet logs",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Optional: only process first N txt files",
    )
    parser.add_argument(
        "--skip_missing_ckpt",
        action="store_true",
        help="Skip files not found in progress.pkl instead of raising error",
    )
    parser.add_argument(
        "--skip_mismatch_ckpt",
        action="store_true",
        help="Skip mismatched checkpoint entries instead of raising error",
    )
    parser.add_argument(
        "--skip_frozen",
        action="store_true",
        help="Skip instances whose cur_iter already exceeds max_iters",
    )

    args = parser.parse_args()

    set_random_seed(args.seed)
    os.makedirs(args.save_log_path, exist_ok=True)

    progress = load_progress(args.progress_path)

    summary_records = []
    route_records = []
    failed_files = []
    skipped_files = []

    files = [f for f in os.listdir(args.instance_path) if f.endswith(".txt")]
    files.sort()

    if args.max_instances is not None:
        files = files[:args.max_instances]

    for file in tqdm(files, desc="Processing instances"):
        instance_path = os.path.join(args.instance_path, file)
        instance = load_instance(instance_path)

        if file not in progress:
            msg = f"[Missing CKPT] {file} not found in progress pickle"
            if args.skip_missing_ckpt:
                print(msg)
                skipped_files.append(file)
                continue
            raise KeyError(msg)

        ckpt_entry = progress[file]
        matched, reason = check_ckpt_match(file, ckpt_entry)
        if not matched:
            msg = f"[Mismatch CKPT] {file}: {reason}"
            if args.skip_mismatch_ckpt:
                print(msg)
                skipped_files.append(file)
                continue
            raise ValueError(msg)

        cur_iter = int(ckpt_entry.get("cur_iter", 1))
        max_iters = int(ckpt_entry.get("max_iters", 25000))

        if args.skip_frozen and cur_iter > max_iters:
            print(f"[Skip Frozen] {file}: cur_iter={cur_iter}, max_iters={max_iters}")
            skipped_files.append(file)
            continue

        solver_seed = args.seed + cur_iter
        solver = ALNS_Solver(instance, seed=solver_seed, checkpoint=ckpt_entry)

        print(f"Instance file: {file}")
        print(f"Resume from cur_iter={cur_iter}, delta_iters={args.delta_iters}")

        start_time = time.time()
        solution = solver.solve(delta_iters=args.delta_iters, resume=True)
        end_time = time.time()

        elapsed_time = float(end_time - start_time)
        visited_all = bool(all(solver.visited))

        if not visited_all:
            failed_files.append(file)

        # update checkpoint back into global pickle
        new_ckpt = solver.get_checkpoint()
        attach_expert_delta_to_state(new_ckpt, instance)
        new_ckpt["file"] = file
        new_ckpt["instance_id"] = instance.get("instance_id", os.path.splitext(file)[0])
        new_ckpt["elapsed_time_s_last_run"] = elapsed_time
        new_ckpt["visited_all"] = visited_all

        progress[file] = new_ckpt

        instance_id = instance.get("instance_id", os.path.splitext(file)[0])
        fleet_size = len(solution)
        objective_value = float(getattr(solver, "global_value", float("nan")))
        new_cur_iter = int(getattr(solver, "cur_iter", -1))
        new_max_iters = int(getattr(solver, "max_iters", -1))

        summary_records.append({
            "instance_id": instance_id,
            "file": file,
            "fleet_size": fleet_size,
            "objective_value": objective_value,
            "elapsed_time_s": elapsed_time,
            "visited_all": visited_all,
            "unvisited_count": int(len(solver.visited) - sum(bool(x) for x in solver.visited)),
            "cur_iter_before": cur_iter,
            "cur_iter_after": new_cur_iter,
            "max_iters": new_max_iters,
            "delta_iters": args.delta_iters,
            "routes_json": json.dumps(solution, ensure_ascii=False, default=str),
        })

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
            f"cur_iter: {cur_iter} -> {new_cur_iter}"
        )

    save_progress(progress, args.progress_path)

    df_summary = pd.DataFrame(summary_records)
    df_routes = pd.DataFrame(route_records)

    summary_csv = os.path.join(args.save_log_path, "resume_summary.csv")
    routes_csv = os.path.join(args.save_log_path, "resume_routes.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_routes.to_csv(routes_csv, index=False)

    try:
        summary_parquet = os.path.join(args.save_log_path, "resume_summary.parquet")
        routes_parquet = os.path.join(args.save_log_path, "resume_routes.parquet")
        df_summary.to_parquet(summary_parquet, index=False)
        df_routes.to_parquet(routes_parquet, index=False)
    except Exception as e:
        print(f"[WARN] Parquet save skipped: {e}")

    failed_path = os.path.join(args.save_log_path, "unvisited_instances.txt")
    with open(failed_path, "w") as f:
        for x in failed_files:
            f.write(x + "\n")

    skipped_path = os.path.join(args.save_log_path, "skipped_instances.txt")
    with open(skipped_path, "w") as f:
        for x in skipped_files:
            f.write(x + "\n")

    print(f"Saved summary -> {summary_csv}")
    print(f"Saved routes  -> {routes_csv}")
    print(f"Saved failed list -> {failed_path}")
    print(f"Saved skipped list -> {skipped_path}")
    print(f"Updated progress -> {args.progress_path}")


if __name__ == "__main__":
    main()
