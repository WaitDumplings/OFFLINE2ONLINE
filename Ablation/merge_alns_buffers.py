#!/usr/bin/env python3
"""
Merge ALNS buffer folders while keeping pickle / solomon / progress aligned.

Input buffer layout:
    buffer/
      pickle/evrptw_50C_12R.pkl
      progress/buffer_progress.pkl
      solomon/*.txt

The pickle file is treated as the source of valid instances. A record is merged
only when its pickle record, solomon text file, and progress solution all exist.
The output keeps the same internal layout and rewrites instance ids/files to a
new contiguous sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SOL_PATTERN = re.compile(
    r"^solomon_dataset_(?P<idx>\d+)_(?P<itype>[A-Za-z]+)_(?P<tw>narrow|wide)_(?P<stamp>\d+)\.txt$"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_buffer1 = root / "/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k"
    default_buffer2 = root / "dataset4/unanchored/Cus_50/buffer"
    default_output = root / "/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer"

    parser = argparse.ArgumentParser(
        description="Merge two or more ALNS buffers and reindex them consistently."
    )
    parser.add_argument(
        "--buffers",
        nargs="+",
        type=Path,
        default=[default_buffer1, default_buffer2],
        help="Input buffer directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output buffer directory.",
    )
    parser.add_argument(
        "--pickle-name",
        type=str,
        default="evrptw_50C_12R.pkl",
        help="Pickle filename under each buffer/pickle directory.",
    )
    parser.add_argument(
        "--progress-name",
        type=str,
        default="buffer_progress.pkl",
        help="Progress filename under each buffer/progress directory.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp suffix for output solomon filenames. Defaults to current time.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and print the merge plan without writing output files.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def file_stem(filename: str) -> str:
    return filename[:-4] if filename.endswith(".txt") else filename


def parse_solomon_name(filename: str) -> Optional[Dict[str, Any]]:
    match = SOL_PATTERN.match(filename)
    if match is None:
        return None
    return {
        "idx": int(match.group("idx")),
        "instance_type": match.group("itype"),
        "time_window_type": match.group("tw"),
        "timestamp": match.group("stamp"),
    }


def instance_type_from_record(record: Dict[str, Any], filename: str) -> str:
    env = record.get("env", {})
    value = env.get("instance_type")
    if value:
        return str(value)
    parsed = parse_solomon_name(filename)
    if parsed is not None:
        return str(parsed["instance_type"])
    return "UNK"


def time_window_type_from_record(record: Dict[str, Any], filename: str) -> str:
    env = record.get("env", {})
    value = env.get("time_window_type")
    if value:
        return str(value)
    parsed = parse_solomon_name(filename)
    if parsed is not None:
        return str(parsed["time_window_type"])
    return "unknown"


def sort_key_for_record(source_idx: int, record: Dict[str, Any]) -> Tuple[Any, ...]:
    filename = str(record.get("file") or "")
    parsed = parse_solomon_name(filename)
    if parsed is not None:
        return (
            int(parsed["timestamp"]),
            int(parsed["idx"]),
            source_idx,
            filename,
        )
    return (
        math.inf,
        math.inf,
        source_idx,
        str(record.get("instance_id") or record.get("id") or filename),
    )


def choose_progress(records: List[Tuple[int, Dict[str, Any]]]) -> Tuple[int, Dict[str, Any]]:
    """
    Pick the most useful solution record among duplicates.

    Prefer records with larger cur_iter / selected_count, then lower objective.
    This keeps exact duplicates harmless and preserves later/better progress if
    one buffer contains a refreshed solution for the same instance.
    """
    def key(item: Tuple[int, Dict[str, Any]]) -> Tuple[float, float, float, int]:
        source_idx, rec = item
        cur_iter = float(rec.get("cur_iter", 0) or 0)
        selected_count = float(rec.get("selected_count", 0) or 0)
        value = rec.get("global_value", float("inf"))
        try:
            objective_rank = -float(value)
        except Exception:
            objective_rank = float("-inf")
        return (cur_iter, selected_count, objective_rank, source_idx)

    return max(records, key=key)


def collect_buffer(
    buffer_dir: Path,
    source_idx: int,
    pickle_name: str,
    progress_name: str,
) -> Dict[str, Any]:
    pickle_path = buffer_dir / "pickle" / pickle_name
    progress_path = buffer_dir / "progress" / progress_name
    solomon_dir = buffer_dir / "solomon"

    if not pickle_path.exists():
        raise FileNotFoundError(f"Missing pickle file: {pickle_path}")
    if not progress_path.exists():
        raise FileNotFoundError(f"Missing progress file: {progress_path}")
    if not solomon_dir.exists():
        raise FileNotFoundError(f"Missing solomon directory: {solomon_dir}")

    instances = load_pickle(pickle_path)
    progress = load_pickle(progress_path)

    if not isinstance(instances, list):
        raise TypeError(f"{pickle_path} must contain a list, got {type(instances)}")
    if not isinstance(progress, dict):
        raise TypeError(f"{progress_path} must contain a dict, got {type(progress)}")

    solomon_files = {
        path.name: path for path in solomon_dir.glob("*.txt")
    }
    progress_by_file = {
        str(rec.get("file") or key): rec
        for key, rec in progress.items()
        if isinstance(rec, dict)
    }

    rows = []
    missing_progress = []
    missing_solomon = []
    duplicate_pickle_files = []
    seen_pickle_files = set()

    for local_idx, record in enumerate(instances):
        if not isinstance(record, dict):
            raise TypeError(
                f"{pickle_path} item {local_idx} must be dict, got {type(record)}"
            )

        old_file = str(record.get("file") or "")
        if not old_file:
            old_id = str(record.get("instance_id") or record.get("id") or "")
            old_file = old_id + ".txt" if old_id else ""

        if old_file in seen_pickle_files:
            duplicate_pickle_files.append(old_file)
            continue
        seen_pickle_files.add(old_file)

        progress_record = progress_by_file.get(old_file)
        solomon_path = solomon_files.get(old_file)

        if progress_record is None:
            missing_progress.append(old_file)
            continue
        if solomon_path is None:
            missing_solomon.append(old_file)
            continue

        rows.append(
            {
                "source_idx": source_idx,
                "buffer_dir": buffer_dir,
                "local_idx": local_idx,
                "old_file": old_file,
                "old_id": str(record.get("instance_id") or record.get("id") or file_stem(old_file)),
                "pickle_record": record,
                "progress_record": progress_record,
                "solomon_path": solomon_path,
                "sort_key": sort_key_for_record(source_idx, record),
            }
        )

    return {
        "rows": rows,
        "pickle_count": len(instances),
        "progress_count": len(progress),
        "solomon_count": len(solomon_files),
        "full_triple_count": len(rows),
        "missing_progress": missing_progress,
        "missing_solomon": missing_solomon,
        "progress_without_pickle": sorted(set(progress_by_file) - seen_pickle_files),
        "solomon_without_pickle": sorted(set(solomon_files) - seen_pickle_files),
        "duplicate_pickle_files": duplicate_pickle_files,
    }


def merge_rows(collected: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_file: Dict[str, List[Dict[str, Any]]] = {}

    for result in collected:
        for row in result["rows"]:
            by_file.setdefault(row["old_file"], []).append(row)

    merged_rows = []
    duplicate_files = {}

    for old_file, rows in by_file.items():
        if len(rows) == 1:
            merged_rows.append(rows[0])
            continue

        duplicate_files[old_file] = len(rows)
        progress_candidates = [
            (int(row["source_idx"]), row["progress_record"])
            for row in rows
        ]
        chosen_source_idx, chosen_progress = choose_progress(progress_candidates)
        chosen = rows[-1]
        for row in rows:
            if int(row["source_idx"]) == chosen_source_idx:
                chosen = row
                break
        chosen = dict(chosen)
        chosen["progress_record"] = chosen_progress
        merged_rows.append(chosen)

    merged_rows.sort(key=lambda row: row["sort_key"])

    stats = {
        "duplicate_full_triple_files": duplicate_files,
        "merged_full_triple_count": len(merged_rows),
    }
    return merged_rows, stats


def rewrite_records(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    pickle_name: str,
    progress_name: str,
    timestamp: str,
) -> Dict[str, Any]:
    pickle_dir = output_dir / "pickle"
    progress_dir = output_dir / "progress"
    solomon_dir = output_dir / "solomon"
    pickle_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    solomon_dir.mkdir(parents=True, exist_ok=True)

    merged_instances = []
    merged_progress = {}
    manifest_rows = []

    for new_idx, row in enumerate(rows):
        old_file = row["old_file"]
        instance_type = instance_type_from_record(row["pickle_record"], old_file)
        time_window_type = time_window_type_from_record(row["pickle_record"], old_file)
        new_stem = (
            f"solomon_dataset_{new_idx}_{instance_type}_{time_window_type}_{timestamp}"
        )
        new_file = new_stem + ".txt"

        new_pickle = deepcopy(row["pickle_record"])
        new_pickle["id"] = new_stem
        new_pickle["instance_id"] = new_stem
        new_pickle["file"] = new_file
        merged_instances.append(new_pickle)

        new_progress = deepcopy(row["progress_record"])
        new_progress["instance_id"] = new_stem
        new_progress["file"] = new_file
        merged_progress[new_file] = new_progress

        shutil.copy2(row["solomon_path"], solomon_dir / new_file)

        manifest_rows.append(
            {
                "new_idx": new_idx,
                "new_file": new_file,
                "new_instance_id": new_stem,
                "source_buffer": str(row["buffer_dir"]),
                "source_local_idx": row["local_idx"],
                "old_file": old_file,
                "old_instance_id": row["old_id"],
                "instance_type": instance_type,
                "time_window_type": time_window_type,
                "global_value": new_progress.get("global_value"),
            }
        )

    dump_pickle(merged_instances, pickle_dir / pickle_name)
    dump_pickle(merged_progress, progress_dir / progress_name)

    manifest_path = output_dir / "merge_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_idx",
                "new_file",
                "new_instance_id",
                "source_buffer",
                "source_local_idx",
                "old_file",
                "old_instance_id",
                "instance_type",
                "time_window_type",
                "global_value",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "pickle_output": str(pickle_dir / pickle_name),
        "progress_output": str(progress_dir / progress_name),
        "solomon_output": str(solomon_dir),
        "manifest_output": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or str(int(time.time()))

    output_dir = args.output
    if output_dir.exists() and not args.dry_run:
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    collected = []
    for source_idx, buffer_dir in enumerate(args.buffers):
        result = collect_buffer(
            buffer_dir=buffer_dir,
            source_idx=source_idx,
            pickle_name=args.pickle_name,
            progress_name=args.progress_name,
        )
        collected.append(result)
        print(f"\n[Inspect] buffer={buffer_dir}")
        print(
            f"  pickle={result['pickle_count']} | "
            f"progress={result['progress_count']} | "
            f"solomon={result['solomon_count']} | "
            f"full_triples={result['full_triple_count']}"
        )
        print(
            f"  missing_progress={len(result['missing_progress'])} | "
            f"missing_solomon={len(result['missing_solomon'])} | "
            f"progress_without_pickle={len(result['progress_without_pickle'])} | "
            f"solomon_without_pickle={len(result['solomon_without_pickle'])} | "
            f"duplicate_pickle_files={len(result['duplicate_pickle_files'])}"
        )

    rows, merge_stats = merge_rows(collected)
    print("\n[MergePlan]")
    print(f"  output={output_dir}")
    print(f"  timestamp={timestamp}")
    print(f"  merged_full_triples={len(rows)}")
    print(
        f"  duplicate_full_triple_files="
        f"{len(merge_stats['duplicate_full_triple_files'])}"
    )

    summary = {
        "buffers": [str(path) for path in args.buffers],
        "output": str(output_dir),
        "timestamp": timestamp,
        "input": [
            {
                "pickle_count": item["pickle_count"],
                "progress_count": item["progress_count"],
                "solomon_count": item["solomon_count"],
                "full_triple_count": item["full_triple_count"],
                "missing_progress_count": len(item["missing_progress"]),
                "missing_solomon_count": len(item["missing_solomon"]),
                "progress_without_pickle_count": len(item["progress_without_pickle"]),
                "solomon_without_pickle_count": len(item["solomon_without_pickle"]),
                "duplicate_pickle_files_count": len(item["duplicate_pickle_files"]),
                "progress_without_pickle_sample": item["progress_without_pickle"][:20],
                "solomon_without_pickle_sample": item["solomon_without_pickle"][:20],
            }
            for item in collected
        ],
        "merged_full_triple_count": len(rows),
        "duplicate_full_triple_files": merge_stats["duplicate_full_triple_files"],
    }

    if args.dry_run:
        print("\n[DryRun] no files written.")
        print(json.dumps(summary, indent=2))
        return

    outputs = rewrite_records(
        rows=rows,
        output_dir=output_dir,
        pickle_name=args.pickle_name,
        progress_name=args.progress_name,
        timestamp=timestamp,
    )
    summary.update(outputs)

    summary_path = output_dir / "merge_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n[Done]")
    print(f"  pickle:   {outputs['pickle_output']}")
    print(f"  progress: {outputs['progress_output']}")
    print(f"  solomon:  {outputs['solomon_output']}")
    print(f"  manifest: {outputs['manifest_output']}")
    print(f"  summary:  {summary_path}")


if __name__ == "__main__":
    main()
