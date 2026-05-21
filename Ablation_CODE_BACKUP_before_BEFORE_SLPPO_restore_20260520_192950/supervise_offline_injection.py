#!/usr/bin/env python3
import json
import os
import pickle
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/Maojie/Github2/EVRP-TW-D-B_Weekend")
PYTHON = "/home/npg/miniconda3/envs/maojie/bin/python"

PHASE1_LOG_DIR = ROOT / "LOGS_NEW" / "RGAE_1000_seed3000"
PHASE1_IMG_DIR = ROOT / "IMGS_NEW" / "RGAE_1000_seed3000"
PHASE2_LOG_DIR = ROOT / "LOGS_NEW" / "RGAE_POMO50_300_seed3000"
PHASE2_IMG_DIR = ROOT / "IMGS_NEW" / "RGAE_POMO50_300_seed3000"
SUP_DIR = ROOT / "LOGS_NEW" / "OFFLINE_INJECTION_SUPERVISION"
ALNS_LOG_DIR = ROOT / "LOGS_NEW" / "ALNS_5K_resume_32"
ALNS_PROGRESS = ROOT / "dataset" / "unanchored" / "Cus_50" / "buffer" / "progress" / "buffer_progress.pkl"

FIRST_PHASE_PIDS = [21922, 21169, 21170, 21171]
PLOT_WATCHER_PID = 22219
ALNS_PID = 28990

EVAL_FREQ = 10
SUCCESS_GAP_OBJ = 30.0
WEAK_GAP_OBJ = 15.0
CHECK_INTERVAL = 600

PHASE1_SERIES = {
    "Vanilla": PHASE1_LOG_DIR / "gpu0_vanilla_ppo_2head_seed3000_u1000.txt",
    "Exp1_sampler": PHASE1_LOG_DIR / "gpu1_rgae_exp1_sampler_seed3000_u1000.txt",
    "Exp2_sampler_group": PHASE1_LOG_DIR / "gpu2_rgae_exp2_sampler_group_seed3000_u1000.txt",
    "Exp3_sampler_group_ref": PHASE1_LOG_DIR / "gpu3_rgae_exp3_sampler_group_ref_seed3000_u1000.txt",
}

PHASE2_SERIES = {
    "Vanilla": PHASE2_LOG_DIR / "gpu0_vanilla_ppo_2head_seed3000_u300.txt",
    "POMO50_Exp1_sampler": PHASE2_LOG_DIR / "gpu1_pomo50_exp1_sampler_seed3000_u300.txt",
    "POMO50_Exp2_group": PHASE2_LOG_DIR / "gpu2_pomo50_exp2_group_seed3000_u300.txt",
    "POMO50_Exp3_group_ref": PHASE2_LOG_DIR / "gpu3_pomo50_exp3_group_ref_seed3000_u300.txt",
}

EVAL_RE = re.compile(
    r"\[EvalSummary\] mode=(\S+) episodes=(\d+) avg_reward=([-0-9.]+).*?"
    r"avg_cs=([-0-9.]+) solved_rate=([-0-9.]+) aggregation=(\S+)"
)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    SUP_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with (SUP_DIR / "supervisor_events.log").open("a") as f:
        f.write(line + "\n")


def is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_pids(pids, label):
    alive = [pid for pid in pids if is_alive(pid)]
    if not alive:
        log(f"{label}: no live pids to stop")
        return
    log(f"{label}: sending SIGTERM to {alive}")
    for pid in alive:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(20)
    still = [pid for pid in alive if is_alive(pid)]
    if still:
        log(f"{label}: sending SIGKILL to {still}")
        for pid in still:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def parse_evals(path):
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    rows = []
    best_idx = 0
    for m in EVAL_RE.finditer(text):
        mode, episodes, reward, avg_cs, solved, agg = m.groups()
        if agg == "best_of_8":
            best_idx += 1
            update = best_idx * EVAL_FREQ
        else:
            update = best_idx * EVAL_FREQ
        rows.append(
            {
                "update": update,
                "mode": mode,
                "episodes": int(episodes),
                "aggregation": agg,
                "reward": float(reward),
                "objective": -100.0 * float(reward),
                "solved_rate": float(solved),
                "avg_cs": float(avg_cs),
            }
        )
    return rows


def phase_snapshot(series):
    snap = {}
    for name, path in series.items():
        rows = parse_evals(path)
        best = [r for r in rows if r["aggregation"] == "best_of_8"]
        avg = [r for r in rows if r["aggregation"] == "trajectory_avg"]
        snap[name] = {
            "path": str(path),
            "latest_best": best[-1] if best else None,
            "latest_avg": avg[-1] if avg else None,
            "best_best": min(best, key=lambda r: r["objective"]) if best else None,
            "eval_count": len(best),
        }
    return snap


def min_latest_update(snap):
    vals = [v["latest_best"]["update"] for v in snap.values() if v["latest_best"]]
    return min(vals) if vals else 0


def _row_at_or_before(item, aggregation, target_update):
    rows = [
        r for r in parse_evals(Path(item["path"]))
        if r["aggregation"] == aggregation and r["update"] <= target_update
    ]
    if not rows:
        return None
    return max(rows, key=lambda r: r["update"])


def evaluate_offline_success(snap, min_update, relaxed=False):
    # Compare all methods at the same evaluation update. GPU speeds differ, so
    # latest-vs-latest can unfairly favor faster runs.
    target_update = min_update
    vanilla = snap.get("Vanilla", {})
    vb = _row_at_or_before(vanilla, "best_of_8", target_update) if vanilla else None
    va = _row_at_or_before(vanilla, "trajectory_avg", target_update) if vanilla else None
    if not vb or not va or vb["update"] < target_update or va["update"] < target_update:
        return {"success": False, "reason": "missing synced vanilla eval", "winner": None, "deltas": {}, "target_update": target_update}
    threshold = WEAK_GAP_OBJ if relaxed else SUCCESS_GAP_OBJ
    deltas = {}
    winner = None
    best_delta = -1e9
    for name, item in snap.items():
        if name == "Vanilla":
            continue
        b = _row_at_or_before(item, "best_of_8", target_update)
        a = _row_at_or_before(item, "trajectory_avg", target_update)
        if not b or not a or b["update"] < target_update or a["update"] < target_update:
            continue
        delta_best = vb["objective"] - b["objective"]
        delta_avg = va["objective"] - a["objective"]
        ok = (
            b["solved_rate"] >= 0.99
            and delta_best >= threshold
            and delta_avg >= threshold
        )
        deltas[name] = {
            "update": target_update,
            "delta_best_obj": delta_best,
            "delta_avg_obj": delta_avg,
            "offline_best_obj": b["objective"],
            "offline_avg_obj": a["objective"],
            "success": ok,
        }
        if delta_best > best_delta:
            best_delta = delta_best
            winner = name
    success = any(v["success"] for v in deltas.values())
    return {
        "success": success,
        "threshold": threshold,
        "target_update": target_update,
        "winner": winner,
        "deltas": deltas,
        "vanilla_best_obj": vb["objective"],
        "vanilla_avg_obj": va["objective"],
    }


def dump_json(name, obj):
    SUP_DIR.mkdir(parents=True, exist_ok=True)
    path = SUP_DIR / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return path


def maybe_plot(series, out_dir, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"plot skipped: {exc}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [("objective", "Objective lower is better"), ("solved_rate", "Solved Rate"), ("avg_cs", "Average CS")]
    for agg in ["best_of_8", "trajectory_avg"]:
        for metric, ylabel in metrics:
            curves = []
            for name, path in series.items():
                rows = [r for r in parse_evals(path) if r["aggregation"] == agg]
                if rows:
                    curves.append((name, rows))
            if not curves:
                continue
            if metric == "objective":
                fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.35, 1.0]})
                main_ax, zoom_ax = axes
                for name, rows in curves:
                    xs = [r["update"] for r in rows]
                    ys = [r[metric] for r in rows]
                    main_ax.plot(xs, ys, marker="o", label=name)
                    zoom_ax.plot(xs, ys, marker="o", label=name)
                main_ax.set_title(f"{title} | {agg} | {metric}")
                main_ax.set_xlabel("Update")
                main_ax.set_ylabel(ylabel)
                main_ax.grid(True, alpha=0.25)
                main_ax.legend()
                zoom_ax.set_title("Objective zoom: 900-1050")
                zoom_ax.set_xlabel("Update")
                zoom_ax.set_ylabel("Objective")
                zoom_ax.set_ylim(900, 1050)
                zoom_ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(out_dir / f"{agg}_{metric}.png", dpi=160)
                zoom_fig, zoom_only_ax = plt.subplots(figsize=(9, 5))
                for name, rows in curves:
                    zoom_only_ax.plot([r["update"] for r in rows], [r[metric] for r in rows], marker="o", label=name)
                zoom_only_ax.set_title(f"{title} | {agg} | objective zoom: 900-1050")
                zoom_only_ax.set_xlabel("Update")
                zoom_only_ax.set_ylabel("Objective")
                zoom_only_ax.set_ylim(900, 1050)
                zoom_only_ax.grid(True, alpha=0.25)
                zoom_only_ax.legend()
                zoom_fig.tight_layout()
                zoom_fig.savefig(out_dir / f"{agg}_{metric}_zoom.png", dpi=160)
                plt.close(zoom_fig)
                plt.close(fig)
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            for name, rows in curves:
                ax.plot([r["update"] for r in rows], [r[metric] for r in rows], marker="o", label=name)
            ax.set_title(f"{title} | {agg} | {metric}")
            ax.set_xlabel("Update")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{agg}_{metric}.png", dpi=160)
            plt.close(fig)


def progress_stats():
    if not ALNS_PROGRESS.exists():
        return {"exists": False}
    with ALNS_PROGRESS.open("rb") as f:
        data = pickle.load(f)
    records = data.values() if isinstance(data, dict) else data
    cur_vals = []
    max_vals = []
    frozen = 0
    total = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        total += 1
        if "cur_iter" in rec:
            cur_vals.append(int(rec["cur_iter"]))
        if "max_iters" in rec:
            max_vals.append(int(rec["max_iters"]))
        if rec.get("is_frozen"):
            frozen += 1
    return {
        "exists": True,
        "records": total,
        "cur_iter_min": min(cur_vals) if cur_vals else None,
        "cur_iter_max": max(cur_vals) if cur_vals else None,
        "max_iters_min": min(max_vals) if max_vals else None,
        "max_iters_max": max(max_vals) if max_vals else None,
        "frozen": frozen,
        "done_5k": bool(cur_vals) and min(cur_vals) >= 5000,
    }


def wait_for_alns_5k():
    while True:
        stats = progress_stats()
        dump_json("alns_5k_progress_latest.json", stats)
        if stats.get("done_5k"):
            log(f"ALNS 5k progress complete: {stats}")
            return stats
        if not is_alive(ALNS_PID):
            log(f"ALNS pid {ALNS_PID} not alive; current stats={stats}")
            if stats.get("cur_iter_min", 0) and stats.get("cur_iter_min", 0) >= 5000:
                return stats
        else:
            log(f"Waiting for ALNS 5k buffer update: {stats}")
        time.sleep(CHECK_INTERVAL)


def launch(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    f = log_path.open("w")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    log(f"launched pid={proc.pid} log={log_path.name}: {' '.join(cmd)}")
    return proc.pid


def common_train_args(exp_name, cuda_id, updates):
    return [
        "--exp-name", exp_name,
        "--cuda-id", str(cuda_id),
        "--seed", "3000",
        "--cuda", "true",
        "--torch-deterministic", "true",
        "--save-dir", str(ROOT / "checkpoint" / "RGAE_POMO50"),
        "--eval-data-path", str(ROOT / "dataset" / "unanchored" / "Cus_50" / "pickle" / "evrptw_50C_12R.pkl"),
        "--eval-freq", "10",
        "--eval-batch-size", "1000",
        "--num-updates", str(updates),
        "--num-envs", "128",
        "--num-steps", "75",
        "--n-traj", "50",
        "--test-agent", "8",
        "--num-minibatches", "128",
        "--accum-steps", "16",
        "--eval-decode-mode", "sampling",
        "--train-config-schedule", "batch_cycle",
        "--train-config-stratify-keys", "instance_type,time_window_policy",
        "--train-config-fixed-overrides", "service_time_policy=cargoweight,cluster_number_policy=random",
        "--train-config-use-online-counter", "true",
        "--train-env-seed-by-update", "true",
        "--use-direct-progress-pbrs", "true",
        "--progress-pbrs-coef", "2.0",
        "--progress-pbrs-beta", "0.5",
        "--use-repair-fail-reward", "true",
        "--repair-fail-coef", "1.0",
        "--repair-progress-coef", "1.0",
        "--repair-success-bonus", "1.0",
        "--use-decomposed-reward-adv", "true",
        "--decomposed-reward-mode", "objective_progress",
        "--adv-objective-weight", "0.5",
        "--adv-progress-weight", "0.5",
        "--debug", "true",
        "--debug-test", "true",
    ]


def launch_phase2():
    PHASE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_IMG_DIR.mkdir(parents=True, exist_ok=True)
    pids = {}
    vanilla_cmd = [PYTHON, "train.py"] + common_train_args("gpu0_vanilla_ppo_2head_seed3000_u300", 0, 300) + [
        "--use-alns-teacher", "false",
        "--use-alns-bc", "false",
        "--use-alns-preference", "false",
    ]
    pids["Vanilla"] = launch(vanilla_cmd, PHASE2_SERIES["Vanilla"])

    base_inc = [
        "--adv-weight-schedule", "fixed",
        "--alns-buffer-dir", str(ROOT / "dataset" / "unanchored" / "Cus_50" / "buffer"),
        "--incumbent-ratio", "0.20",
        "--buffer-normal-frac", "1.0",
        "--buffer-temp-frac", "0.0",
        "--buffer-prefix-frac", "0.0",
        "--online-normal-frac", "1.0",
        "--online-temp-frac", "0.0",
        "--temperature-sampling", "1.0",
        "--use-regret-aware-sampler", "true",
        "--use-pomo50-regret-state", "true",
        "--pomo50-stable-beat-rate", "0.20",
        "--pomo50-low-beat-rate", "0.05",
        "--pomo50-regret-margin-rel", "0.01",
        "--buffer-regret-relative", "true",
        "--buffer-regret-kappa", "25.0",
        "--buffer-regret-margin", "0.0",
        "--buffer-unknown-weight", "1.0",
        "--buffer-uncertain-weight", "0.5",
        "--buffer-ppo-win-weight", "0.05",
        "--buffer-lucky-ppo-weight", "0.35",
        "--buffer-alns-win-base-weight", "1.0",
        "--regret-recompute-freq", "10",
        "--regret-probe-size", "256",
        "--use-route-preference", "false",
        "--use-selective-bc", "false",
        "--use-self-generated-buffer", "false",
        "--cmp-adv-coef", "0.0",
        "--adv-diag", "true",
        "--grad-cos-diag", "true",
        "--grad-cos-freq", "10",
        "--reference-adv-alns-win-only", "true",
    ]
    exp1_cmd = [PYTHON, "train_incumbent.py"] + common_train_args("gpu1_pomo50_exp1_sampler_seed3000_u300", 1, 300) + base_inc + [
        "--group-adv-coef", "0.0",
        "--reference-adv-coef", "0.0",
    ]
    exp2_cmd = [PYTHON, "train_incumbent.py"] + common_train_args("gpu2_pomo50_exp2_group_seed3000_u300", 2, 300) + base_inc + [
        "--group-adv-coef", "0.30",
        "--group-adv-clip", "3.0",
        "--reference-adv-coef", "0.0",
    ]
    exp3_cmd = [PYTHON, "train_incumbent.py"] + common_train_args("gpu3_pomo50_exp3_group_ref_seed3000_u300", 3, 300) + base_inc + [
        "--group-adv-coef", "0.30",
        "--group-adv-clip", "3.0",
        "--reference-adv-coef", "0.10",
        "--reference-adv-rho", "0.05",
        "--reference-adv-clip", "2.0",
    ]
    pids["POMO50_Exp1_sampler"] = launch(exp1_cmd, PHASE2_SERIES["POMO50_Exp1_sampler"])
    pids["POMO50_Exp2_group"] = launch(exp2_cmd, PHASE2_SERIES["POMO50_Exp2_group"])
    pids["POMO50_Exp3_group_ref"] = launch(exp3_cmd, PHASE2_SERIES["POMO50_Exp3_group_ref"])
    dump_json("phase2_pids.json", pids)
    return pids


def wait_phase(series, target_update, label, pids=None):
    while True:
        snap = phase_snapshot(series)
        dump_json(f"{label}_snapshot_latest.json", snap)
        maybe_plot(series, PHASE2_IMG_DIR if "phase2" in label else PHASE1_IMG_DIR, label)
        reached = min_latest_update(snap)
        log(f"{label}: min_latest_update={reached}/{target_update}")
        if reached >= target_update:
            return snap
        if pids and not any(is_alive(pid) for pid in pids.values()):
            log(f"{label}: all launched pids exited before target")
            return snap
        time.sleep(CHECK_INTERVAL)


def write_report(phase1_snap, phase1_decision, phase2_snap=None, alns_stats=None):
    lines = []
    lines.append("# Offline Injection Supervision Report")
    lines.append("")
    lines.append(f"Generated: {now()}")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("Validate whether offline ALNS archive injection improves the final optimization objective, not just feasibility.")
    lines.append("")
    lines.append("## Phase 1: RGAE 20% Archive Injection")
    lines.append("")
    vb = phase1_snap["Vanilla"]["latest_best"]
    va = phase1_snap["Vanilla"]["latest_avg"]
    lines.append(f"Vanilla latest: update={vb['update']} best_obj={vb['objective']:.3f}, avg_obj={va['objective']:.3f}, solved={vb['solved_rate']:.3f}")
    for name, item in phase1_snap.items():
        if name == "Vanilla":
            continue
        b = item["latest_best"]
        a = item["latest_avg"]
        db = vb["objective"] - b["objective"]
        da = va["objective"] - a["objective"]
        lines.append(f"- {name}: update={b['update']} best_obj={b['objective']:.3f} ({db:+.3f} vs Vanilla), avg_obj={a['objective']:.3f} ({da:+.3f}), solved={b['solved_rate']:.3f}")
    lines.append("")
    lines.append(f"Decision: {json.dumps(phase1_decision, ensure_ascii=False)}")
    lines.append("")
    if alns_stats:
        lines.append("## ALNS Buffer Update")
        lines.append("")
        lines.append(f"Progress stats: `{json.dumps(alns_stats, ensure_ascii=False)}`")
        lines.append("")
    if phase2_snap:
        lines.append("## Phase 2: POMO-50 Regret-State Archive Injection")
        lines.append("")
        vb2 = phase2_snap["Vanilla"]["latest_best"]
        va2 = phase2_snap["Vanilla"]["latest_avg"]
        lines.append(f"Vanilla latest: update={vb2['update']} best_obj={vb2['objective']:.3f}, avg_obj={va2['objective']:.3f}, solved={vb2['solved_rate']:.3f}")
        for name, item in phase2_snap.items():
            if name == "Vanilla":
                continue
            b = item["latest_best"]
            a = item["latest_avg"]
            db = vb2["objective"] - b["objective"]
            da = va2["objective"] - a["objective"]
            lines.append(f"- {name}: update={b['update']} best_obj={b['objective']:.3f} ({db:+.3f} vs Vanilla), avg_obj={a['objective']:.3f} ({da:+.3f}), solved={b['solved_rate']:.3f}")
        lines.append("")
    lines.append("## Current Interpretation")
    lines.append("")
    lines.append("- Treat ALNS as an archive/search-region oracle, not a step-level expert policy.")
    lines.append("- POMO-50 regret-state is used in training to estimate current policy search potential before applying ALNS-guided sampling/reference pressure.")
    lines.append("- A valid improvement should show lower objective under the same eval data, with solved rate near 1.0 and trajectory-average also improving, not only a single lucky best trajectory.")
    lines.append("")
    lines.append("## Next Experiments")
    lines.append("")
    lines.append("1. If POMO-50 Exp2/Exp3 remains ahead, run 3 seeds and compare confidence intervals at 300-1000 updates.")
    lines.append("2. Add an explicit structural-prior sampler only for stable ALNS-win instances, with behavior log-prob computed under the guided policy.")
    lines.append("3. Separate exploration quality from distillation: report greedy, sample mean, and best-of-n; if best improves but mean/greedy do not, add archive-best distillation without BC on raw ALNS steps.")
    lines.append("4. Keep progress reward as feasibility scaffold; for late-stage route quality, prioritize objective/group/reference advantage.")
    path = SUP_DIR / "offline_injection_experiment_report.md"
    path.write_text("\n".join(lines))
    log(f"report written: {path}")


def main():
    SUP_DIR.mkdir(parents=True, exist_ok=True)
    log("supervisor started")
    first150 = wait_phase(PHASE1_SERIES, 150, "phase1_to_150")
    decision150 = evaluate_offline_success(first150, 150, relaxed=False)
    dump_json("phase1_decision_150.json", {"snapshot": first150, "decision": decision150})
    log(f"phase1 decision @150: {decision150}")

    if decision150["success"]:
        phase1_final = first150
        phase1_decision = {"stage": 150, **decision150}
    else:
        log("phase1 unclear at 150; continuing to 300")
        phase1_300 = wait_phase(PHASE1_SERIES, 300, "phase1_to_300")
        decision300 = evaluate_offline_success(phase1_300, 300, relaxed=True)
        dump_json("phase1_decision_300.json", {"snapshot": phase1_300, "decision": decision300})
        log(f"phase1 decision @300: {decision300}")
        phase1_final = phase1_300
        phase1_decision = {"stage": 300, **decision300}

    stop_pids(FIRST_PHASE_PIDS + [PLOT_WATCHER_PID], "phase1 cleanup")
    maybe_plot(PHASE1_SERIES, PHASE1_IMG_DIR, "phase1_final")

    if not phase1_decision["success"]:
        write_report(phase1_final, phase1_decision, phase2_snap=None, alns_stats=progress_stats())
        log("offline injection was not strong enough by decision threshold; supervisor complete without phase2 launch")
        return

    alns_stats = wait_for_alns_5k()
    phase2_pids = launch_phase2()
    phase2_snap = wait_phase(PHASE2_SERIES, 300, "phase2_to_300", phase2_pids)
    maybe_plot(PHASE2_SERIES, PHASE2_IMG_DIR, "phase2_final")
    dump_json("phase2_final_snapshot.json", phase2_snap)
    write_report(phase1_final, phase1_decision, phase2_snap=phase2_snap, alns_stats=alns_stats)
    log("supervisor complete")


if __name__ == "__main__":
    main()
