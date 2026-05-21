#!/usr/bin/env python3
"""Train static missing-mode selector on buffer labels and simulate eval allocation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NAME_RE = re.compile(r"solomon_dataset_(?P<idx>\d+)_(?P<itype>[A-Za-z]+)_(?P<tw>narrow|wide)_")


def _stem(value):
    if value is None:
        return ""
    return os.path.splitext(os.path.basename(str(value)))[0]


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_obj_csv(path):
    out = {}
    for row in read_csv_rows(path):
        out[str(row["index"])] = row
    return out


def match_type_tw(inst):
    file_name = str(inst.get("file", ""))
    m = NAME_RE.search(file_name)
    itype = str(inst.get("env", {}).get("instance_type", "UNK"))
    tw = str(inst.get("env", {}).get("time_window_type", "unknown"))
    if m:
        itype = m.group("itype")
        tw = m.group("tw")
    return itype, tw


def instance_features(inst):
    env = inst.get("env", {})
    depot = np.asarray(inst["depot"], dtype=np.float64).reshape(-1, 2)
    cus = np.asarray(inst["customers"], dtype=np.float64).reshape(-1, 2)
    rs = np.asarray(inst["charging_stations"], dtype=np.float64).reshape(-1, 2)
    dem = np.asarray(inst.get("demands", []), dtype=np.float64).reshape(-1)
    tw = np.asarray(inst.get("tw", []), dtype=np.float64).reshape(-1, 2)
    service = np.asarray(inst.get("service_time", []), dtype=np.float64).reshape(-1)
    all_nodes = np.concatenate([depot, cus, rs], axis=0)
    depot_dist = np.linalg.norm(cus - depot[0], axis=1) if len(cus) else np.zeros(1)
    if len(rs):
        nearest_rs = np.min(np.linalg.norm(cus[:, None, :] - rs[None, :, :], axis=-1), axis=1)
    else:
        nearest_rs = np.zeros(len(cus))
    if len(cus) > 1:
        center = np.mean(cus, axis=0)
        dispersion = np.linalg.norm(cus - center, axis=1)
        pair_sample = np.linalg.norm(cus[:, None, :] - cus[None, :, :], axis=-1)
        pair_mean = float(np.mean(pair_sample))
    else:
        dispersion = np.zeros(1)
        pair_mean = 0.0
    tw_width = (tw[:, 1] - tw[:, 0]) if tw.size else np.zeros(1)
    tw_start = tw[:, 0] if tw.size else np.zeros(1)
    tw_end = tw[:, 1] if tw.size else np.zeros(1)
    itype, tw_type = match_type_tw(inst)
    bucket_feats = [
        float(itype == "R"),
        float(itype == "C"),
        float(itype == "RC"),
        float(tw_type == "narrow"),
        float(tw_type == "wide"),
    ]
    capacity = float(env.get("loading_capacity", 1.0) or 1.0)
    battery = float(env.get("battery_capacity", 1.0) or 1.0)
    cons = float(env.get("consumption_per_distance", 1.0) or 1.0)
    horizon = float(env.get("instance_endTime", 1.0) or 1.0)
    span = np.ptp(all_nodes, axis=0) if len(all_nodes) else np.zeros(2)
    raw = [
        len(cus),
        len(rs),
        float(np.mean(dem)) if dem.size else 0.0,
        float(np.std(dem)) if dem.size else 0.0,
        float(np.sum(dem) / max(capacity, 1e-8)),
        float(np.mean(service) / max(horizon, 1e-8)) if service.size else 0.0,
        float(np.mean(tw_width) / max(horizon, 1e-8)),
        float(np.std(tw_width) / max(horizon, 1e-8)),
        float(np.min(tw_width) / max(horizon, 1e-8)),
        float(np.mean(tw_start) / max(horizon, 1e-8)),
        float(np.mean(tw_end) / max(horizon, 1e-8)),
        float(np.mean(depot_dist)),
        float(np.std(depot_dist)),
        float(np.max(depot_dist)),
        float(np.mean(nearest_rs)) if len(nearest_rs) else 0.0,
        float(np.std(nearest_rs)) if len(nearest_rs) else 0.0,
        float(np.mean(dispersion)),
        float(np.std(dispersion)),
        pair_mean,
        float(span[0]),
        float(span[1]),
        float(cons / max(battery, 1e-8)),
        float(len(rs) / max(len(cus), 1)),
    ]
    return np.asarray(bucket_feats + raw, dtype=np.float64)


def build_instance_index(instances):
    out = {}
    for i, inst in enumerate(instances):
        for key in [inst.get("file"), inst.get("instance_id"), inst.get("id"), i]:
            if key is not None:
                out[str(key)] = i
                out[_stem(key)] = i
    return out


def build_train_data(buffer_pkl, missing_csv):
    with open(buffer_pkl, "rb") as f:
        instances = pickle.load(f)
    index = build_instance_index(instances)
    rows = read_csv_rows(missing_csv)
    xs, ys = [], []
    for row in rows:
        key = row.get("file") or row.get("instance_id")
        idx = index.get(str(key), index.get(_stem(key)))
        if idx is None:
            continue
        xs.append(instance_features(instances[idx]))
        ys.append(int(row.get("selected", 0)))
    return np.vstack(xs), np.asarray(ys, dtype=np.int64)


def build_eval_features(eval_pkl):
    with open(eval_pkl, "rb") as f:
        instances = pickle.load(f)
    xs = np.vstack([instance_features(inst) for inst in instances])
    return xs


def train_models(x, y, seed):
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=seed,
    )
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            l2_regularization=0.05,
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
        score = ap
        if score > best_score:
            best_name, best_score, best_model = name, score, model
    best_model.fit(x, y)
    return best_name, best_model, metrics


def simulate(scores, base50_csv, candidate_csv, output_csv, prefix):
    base = read_obj_csv(base50_csv)
    cand = read_obj_csv(candidate_csv)
    keys = sorted(set(base) & set(cand), key=lambda k: int(k))
    base_obj = np.asarray([float(base[k]["objective_value"]) for k in keys], dtype=np.float64)
    cand_obj = np.asarray([float(cand[k]["objective"]) if "objective" in cand[k] else float(cand[k]["objective_value"]) for k in keys], dtype=np.float64)
    score_arr = np.asarray([scores[int(k)] for k in keys], dtype=np.float64)
    rows = []
    summaries = []
    prior = None
    for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        threshold = float(np.quantile(score_arr, 1.0 - frac))
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
    best = min(summaries, key=lambda s: s["avg_obj"])
    threshold = best["threshold"]
    use = score_arr >= threshold
    obj = np.where(use, cand_obj, base_obj)
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
    parser.add_argument("--missing-csv", required=True)
    parser.add_argument("--eval-pkl", required=True)
    parser.add_argument("--base50-csv", required=True)
    parser.add_argument("--candidate-csv", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    x, y = build_train_data(args.buffer_pkl, args.missing_csv)
    best_name, model, metrics = train_models(x, y, args.seed)
    eval_x = build_eval_features(args.eval_pkl)
    scores = model.predict_proba(eval_x)[:, 1]
    all_summaries = {}
    best_by_candidate = {}
    for name, csv_path in args.candidate_csv:
        summaries, best = simulate(
            scores,
            args.base50_csv,
            csv_path,
            str(Path(args.output_dir) / f"{name}_selected.csv"),
            name,
        )
        all_summaries[name] = summaries
        best_by_candidate[name] = best
    summary = {
        "train_instances": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "best_model": best_name,
        "val_metrics": metrics,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "best_by_candidate": best_by_candidate,
        "all_summaries": all_summaries,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[SelectorSummary] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
