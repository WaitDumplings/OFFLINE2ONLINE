#!/usr/bin/env python3
"""Train selector on actual buffer branch-help labels, then simulate eval allocation."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_selector_and_simulate import instance_features, read_obj_csv


def load_instances(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def obj_from_row(row):
    if "objective" in row:
        return float(row["objective"])
    return float(row["objective_value"])


def make_candidate(base_part_csv, off_part_csv):
    base = read_obj_csv(base_part_csv)
    off = read_obj_csv(off_part_csv)
    keys = sorted(set(base) & set(off), key=lambda k: int(k))
    return {k: min(obj_from_row(base[k]), obj_from_row(off[k])) for k in keys}


def train_model(x, y, seed):
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=seed
    )
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.035,
            l2_regularization=0.08,
            random_state=seed,
        ),
    }
    metrics = {}
    best_name, best_score, best_model = None, -np.inf, None
    for name, model in models.items():
        model.fit(x_train, y_train)
        p = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, p)
        ap = average_precision_score(y_val, p)
        metrics[name] = {"val_auc": float(auc), "val_ap": float(ap)}
        if ap > best_score:
            best_name, best_score, best_model = name, ap, model
    best_model.fit(x, y)
    return best_name, best_model, metrics


def simulate(scores, eval_base50_csv, eval_candidate_csv, output_csv, prefix):
    base = read_obj_csv(eval_base50_csv)
    cand = read_obj_csv(eval_candidate_csv)
    keys = sorted(set(base) & set(cand), key=lambda k: int(k))
    base_obj = np.asarray([obj_from_row(base[k]) for k in keys], dtype=np.float64)
    cand_obj = np.asarray([obj_from_row(cand[k]) for k in keys], dtype=np.float64)
    score_arr = np.asarray([scores[int(k)] for k in keys], dtype=np.float64)
    summaries = []
    for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]:
        threshold = float(np.quantile(score_arr, 1.0 - frac)) if frac < 1.0 else -np.inf
        use = score_arr >= threshold
        obj = np.where(use, cand_obj, base_obj)
        delta = obj - base_obj
        summaries.append(
            {
                "name": f"{prefix}_top{int(frac * 100)}",
                "activate": int(np.sum(use)),
                "avg_obj": float(np.mean(obj)),
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "wins": int(np.sum(delta < -1e-9)),
                "losses": int(np.sum(delta > 1e-9)),
                "ties": int(np.sum(np.abs(delta) <= 1e-9)),
                "threshold": threshold,
            }
        )
    best = min(summaries, key=lambda row: row["avg_obj"])
    use = score_arr >= best["threshold"]
    obj = np.where(use, cand_obj, base_obj)
    rows = []
    for i, key in enumerate(keys):
        row = dict(base[key])
        row["selector_score"] = float(score_arr[i])
        row["use_candidate"] = int(use[i])
        row["selected_objective_value"] = float(obj[i])
        row["base50_objective_value"] = float(base_obj[i])
        row["candidate_objective_value"] = float(cand_obj[i])
        rows.append(row)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summaries, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-pkl", required=True)
    parser.add_argument("--buffer-base50-csv", required=True)
    parser.add_argument("--buffer-base-part-csv", required=True)
    parser.add_argument("--buffer-off-part-csv", required=True)
    parser.add_argument("--eval-pkl", required=True)
    parser.add_argument("--eval-base50-csv", required=True)
    parser.add_argument("--eval-candidate-csv", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    buffer_instances = load_instances(args.buffer_pkl)
    base50 = read_obj_csv(args.buffer_base50_csv)
    cand_train = make_candidate(args.buffer_base_part_csv, args.buffer_off_part_csv)
    keys = sorted(set(base50) & set(cand_train), key=lambda k: int(k))
    x = np.vstack([instance_features(buffer_instances[int(k)]) for k in keys])
    base_obj = np.asarray([obj_from_row(base50[k]) for k in keys], dtype=np.float64)
    cand_obj = np.asarray([cand_train[k] for k in keys], dtype=np.float64)
    delta = cand_obj - base_obj
    y = (delta < -float(args.margin)).astype(np.int64)
    best_name, model, metrics = train_model(x, y, args.seed)

    eval_instances = load_instances(args.eval_pkl)
    eval_x = np.vstack([instance_features(inst) for inst in eval_instances])
    scores = model.predict_proba(eval_x)[:, 1]

    all_summaries = {}
    best_by_candidate = {}
    for name, csv_path in args.eval_candidate_csv:
        summaries, best = simulate(
            scores,
            args.eval_base50_csv,
            csv_path,
            str(Path(args.output_dir) / f"{name}_help_selected.csv"),
            name,
        )
        all_summaries[name] = summaries
        best_by_candidate[name] = best

    summary = {
        "train_instances": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "train_candidate_avg_delta": float(np.mean(delta)),
        "train_candidate_win_rate": float(np.mean(delta < -1e-9)),
        "best_model": best_name,
        "val_metrics": metrics,
        "eval_score_mean": float(np.mean(scores)),
        "eval_score_std": float(np.std(scores)),
        "best_by_candidate": best_by_candidate,
        "all_summaries": all_summaries,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[HelpSelectorSummary] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
