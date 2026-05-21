import os
import argparse
from distutils.util import strtobool

from evrptw_gen.benchmarks.DRL_Solver.train import train


def str2bool(x):
    if isinstance(x, bool):
        return x
    return bool(strtobool(str(x)))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO-based DRL solver for EVRPTW with "
            "periodic ALNS teacher."
        )
    )

    # =========================================================
    # Runtime / Experiment
    # =========================================================
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--cuda", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--torch-deterministic", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--save-dir", type=str, default="./checkpoint")

    parser.add_argument(
        "--init-ckpt-path",
        type=str,
        default=None,
        help="Optional checkpoint used to initialize the student policy.",
    )

    # =========================================================
    # Environment / Problem
    # =========================================================
    parser.add_argument("--problem", type=str, default="evrptw")
    parser.add_argument("--env-id", type=str, default="evrptw-v0")
    parser.add_argument(
        "--env-entry-point",
        type=str,
        default="evrptw_gen.benchmarks.DRL_Solver.envs.evrp_vector_env:EVRPTWVectorEnv",
    )

    parser.add_argument("--train-cus-num", type=int, default=50)
    parser.add_argument("--train-cs-num", type=int, default=12)
    parser.add_argument(
        "--use-candidate-dynamic-embedding",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help=(
            "If True, inject candidate-level dynamic EVRPTW features "
            "(arrival, waiting, slack, load/battery proxies) into decoder logits."
        ),
    )

    parser.add_argument("--config-path", type=str, default="./evrptw_gen/configs/config.yaml")
    parser.add_argument("--perturb-dict-path", type=str, default="./evrptw_gen/configs/perturb_config.yaml")
    parser.add_argument(
        "--train-config-schedule",
        type=str,
        default="batch_cycle",
        choices=["mixed", "cycle", "random", "batch_cycle", "batch_random"],
        help=(
            "Generated train env config schedule. 'mixed' keeps the YAML score "
            "sampler; 'cycle'/'random' fix one positive-score policy combo per "
            "PPO update; 'batch_cycle'/'batch_random' mix combos inside each "
            "PPO rollout batch."
        ),
    )
    parser.add_argument(
        "--train-config-stratify-keys",
        type=str,
        default="instance_type,time_window_policy",
        help=(
            "Comma-separated config policy keys used when train-config-schedule is not mixed."
        ),
    )
    parser.add_argument(
        "--train-config-cycle-offset",
        type=int,
        default=0,
        help="Offset into the scheduled config combo list.",
    )
    parser.add_argument(
        "--train-config-fixed-overrides",
        type=str,
        default="service_time_policy=cargoweight,cluster_number_policy=random",
        help=(
            "Comma-separated key=value choices applied to generated train configs, "
            "for example service_time_policy=cargoweight,cluster_number_policy=random."
        ),
    )
    parser.add_argument(
        "--train-config-use-online-counter",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, cycle config groups by generated-online updates only, "
            "so ALNS teacher updates do not skip a group."
        ),
    )
    parser.add_argument(
        "--train-config-shuffle-combos",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="If True, deterministically shuffle config combos before cycling.",
    )
    parser.add_argument(
        "--train-config-shuffle-seed",
        type=int,
        default=17,
        help="Seed offset used when train-config-shuffle-combos=True.",
    )
    parser.add_argument(
        "--train-config-log",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="If True, log the generated-train config combo used by each scheduled update.",
    )
    parser.add_argument(
        "--train-config-log-freq",
        type=int,
        default=10,
        help="Log every N scheduled generated-train updates when train-config-log=True.",
    )
    parser.add_argument(
        "--train-env-seed-by-update",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="If True, offset generated train env seeds by update to avoid repeated instances.",
    )

    parser.add_argument(
        "--eval-data-path",
        type=str,
        default="./dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl",
    )
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=1000)

    # =========================================================
    # PPO Training
    # =========================================================
    parser.add_argument("--num-updates", type=int, default=4000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=100)

    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--anneal-lr", type=str2bool, default=True, nargs="?", const=True)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)

    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--accum-steps", type=int, default=1)

    parser.add_argument("--norm-adv", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=str2bool, default=True, nargs="?", const=True)

    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--vf-coef", type=float, default=0.5)

    parser.add_argument("--max-grad-norm-backbone", type=float, default=2.0)
    parser.add_argument("--max-grad-norm-critic", type=float, default=10.0)
    parser.add_argument("--target-kl", type=float, default=0.01)

    # =========================================================
    # Model
    # =========================================================
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--n-encode-layers", type=int, default=2)
    parser.add_argument(
        "--max-route-events",
        type=int,
        default=16,
        help=(
            "Number of recent current-route event tokens exposed to the decoder. "
            "Repeated RS visits are represented as separate route events."
        ),
    )

    parser.add_argument(
        "--reward-mode",
        type=str,
        default="vanilla",
        choices=["legacy", "vanilla"],
        help=(
            "legacy keeps the existing reward stack. vanilla keeps only "
            "-distance, served-customer PBRS, and terminal success/failure."
        ),
    )
    parser.add_argument(
        "--use-direct-progress-pbrs",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, replace the old served PBRS with a direct served-ratio "
            "potential: coef * (Phi(s') - Phi(s)), where "
            "Phi=1-(1-served/n)^beta."
        ),
    )
    parser.add_argument("--progress-pbrs-coef", type=float, default=1.0)
    parser.add_argument("--progress-pbrs-beta", type=float, default=0.5)
    parser.add_argument(
        "--use-repair-fail-reward",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, terminal failure reward is a normalized repair-distance "
            "proxy added to the progress component; terminal success defaults to 0."
        ),
    )
    parser.add_argument("--repair-fail-coef", type=float, default=1.0)
    parser.add_argument("--repair-progress-coef", type=float, default=1.0)
    parser.add_argument("--repair-success-bonus", type=float, default=1.0)

    # =========================================================
    # Trajectory / Inference
    # =========================================================
    parser.add_argument("--n-traj", type=int, default=70)
    parser.add_argument("--test-agent", type=int, default=1)
    parser.add_argument(
        "--eval-decode-mode",
        type=str,
        default="greedy",
        choices=["greedy", "sampling"],
        help="Decode mode for the primary evaluation curve.",
    )
    parser.add_argument(
        "--eval-greedy-too",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, run an additional greedy evaluation with one trajectory "
            "on the same eval instances and log it as mode=greedy."
        ),
    )

    # =========================================================
    # Failure penalty / dual update
    # =========================================================
    parser.add_argument("--lambda-fail-init", type=float, default=10.0)
    parser.add_argument("--target-success", type=float, default=0.95)
    parser.add_argument("--lambda-max", type=float, default=50.0)
    parser.add_argument("--lambda-lr-up", type=float, default=1.0)
    parser.add_argument("--lambda-lr-down", type=float, default=2.0)
    parser.add_argument("--lambda-tolerance", type=float, default=0.01)

    # =========================================================
    # Bootstrapped Reward Shaping (BSRS)
    # =========================================================
    parser.add_argument(
        "--use-bsrs",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, add bootstrapped potential-based shaping during PPO "
            "rollouts: r <- r + eta * (gamma * V(s') - V(s)). "
            "The potential is computed from the current critic with no grad."
        ),
    )
    parser.add_argument(
        "--bsrs-eta",
        type=float,
        default=0.05,
        help="BSRS shape scale eta. Small positive values are recommended first.",
    )
    parser.add_argument(
        "--bsrs-clip",
        type=float,
        default=1.0,
        help=(
            "Absolute clamp for the per-step BSRS reward. Use <=0 to disable "
            "clipping."
        ),
    )
    parser.add_argument(
        "--bsrs-warmup-updates",
        type=int,
        default=0,
        help="Number of PPO updates to wait before enabling BSRS.",
    )
    parser.add_argument(
        "--bsrs-value-mode",
        type=str,
        default="weighted",
        choices=[
            "weighted",
            "sum",
            "task",
            "objective",
            "progress",
            "terminal",
            "teacher",
        ],
        help=(
            "How to scalarize a decomposed critic for BSRS. Ignored for a "
            "single-head critic."
        ),
    )
    parser.add_argument(
        "--use-split-bsrs",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, route BSRS by action type: customer actions use the "
            "customer value mode, RS actions use the RS value mode, and depot "
            "actions receive no BSRS."
        ),
    )
    parser.add_argument(
        "--bsrs-customer-eta",
        type=float,
        default=None,
        help=(
            "Customer-action BSRS scale when split BSRS is enabled. Defaults "
            "to --bsrs-eta."
        ),
    )
    parser.add_argument(
        "--bsrs-customer-value-mode",
        type=str,
        default="progress",
        choices=[
            "weighted",
            "sum",
            "task",
            "objective",
            "progress",
            "terminal",
            "teacher",
        ],
        help="Critic head/scalarization used for customer-action split BSRS.",
    )
    parser.add_argument(
        "--bsrs-rs-eta",
        type=float,
        default=0.0,
        help="RS-action BSRS scale when split BSRS is enabled.",
    )
    parser.add_argument(
        "--bsrs-rs-value-mode",
        type=str,
        default="terminal",
        choices=[
            "weighted",
            "sum",
            "task",
            "objective",
            "progress",
            "terminal",
            "teacher",
        ],
        help="Critic head/scalarization used for RS-action split BSRS.",
    )
    parser.add_argument(
        "--bsrs-rs-negative-only",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, keep only negative RS-action split BSRS. This avoids "
            "rewarding station loops when the terminal value head is noisy."
        ),
    )

    # =========================================================
    # ALNS Teacher
    # =========================================================
    parser.add_argument(
        "--use-alns-teacher",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
    )

    parser.add_argument("--alns-buffer-dir", type=str, default="./dataset/unanchored/Cus_50/buffer/")
    parser.add_argument("--alns-progress-name", type=str, default="buffer_progress.pkl")
    parser.add_argument("--alns-instance-pickle-name", type=str, default="evrptw_50C_12R.pkl")
    parser.add_argument("--alns-teacher-freq", type=int, default=30)
    parser.add_argument("--alns-teacher-batch-size", type=int, default=256)
    parser.add_argument("--alns-obj-scale", type=float, default=100.0)
    parser.add_argument(
        "--alns-teacher-filter-better",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help=(
            "If True, probe the current policy on sampled ALNS instances and "
            "only keep records where ALNS is better than the current policy."
        ),
    )
    parser.add_argument(
        "--alns-teacher-filter-margin",
        type=float,
        default=0.0,
        help=(
            "Absolute objective margin for ALNS filtering. With margin m, "
            "ALNS is kept only when alns_obj + m < rl_obj."
        ),
    )
    parser.add_argument(
        "--use-alns-bc",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help=(
            "If True, add a light behavior-cloning update from ALNS best_routes "
            "on ALNS-better offline instances."
        ),
    )
    parser.add_argument("--alns-bc-coef", type=float, default=0.2)
    parser.add_argument("--alns-bc-batch-size", type=int, default=64)
    parser.add_argument("--alns-bc-max-steps", type=int, default=100)
    parser.add_argument(
        "--use-alns-preference",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, add a filtered route-level preference update that ranks "
            "ALNS trajectories above the current student's probe trajectories."
        ),
    )
    parser.add_argument("--alns-pref-coef", type=float, default=0.05)
    parser.add_argument("--alns-pref-beta", type=float, default=2.0)
    parser.add_argument("--alns-pref-batch-size", type=int, default=64)
    parser.add_argument("--alns-pref-max-steps", type=int, default=100)
    parser.add_argument(
        "--alns-pref-length-norm",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="If True, compare average route log-probability per action.",
    )
    parser.add_argument(
        "--use-alns-soft-kl",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, add a student-state soft target update. The target "
            "distribution is built from feasible upcoming ALNS customers."
        ),
    )
    parser.add_argument("--alns-soft-kl-coef", type=float, default=0.05)
    parser.add_argument("--alns-soft-kl-batch-size", type=int, default=64)
    parser.add_argument("--alns-soft-kl-max-steps", type=int, default=100)
    parser.add_argument("--alns-soft-kl-lookahead", type=int, default=8)
    parser.add_argument("--alns-soft-kl-tau", type=float, default=2.0)
    parser.add_argument(
        "--use-alns-gap-gate",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, weight ALNS teacher reward/BC by the current-policy "
            "advantage gap observed during the ALNS probe."
        ),
    )
    parser.add_argument("--alns-gap-gate-temp", type=float, default=0.10)
    parser.add_argument("--alns-gap-gate-margin", type=float, default=0.0)
    parser.add_argument(
        "--use-alns-dense-teacher",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, pass ALNS best-route action sequences into the env and "
            "add a gated dense route-progress teacher reward during ALNS updates."
        ),
    )
    parser.add_argument("--alns-dense-teacher-coef", type=float, default=0.05)
    parser.add_argument(
        "--alns-dense-teacher-mode",
        type=str,
        default="exact",
        choices=["exact", "lookahead", "soft"],
        help=(
            "exact requires the next ALNS action; lookahead/soft allows local "
            "customer reordering within a small future window."
        ),
    )
    parser.add_argument("--alns-dense-teacher-lookahead", type=int, default=1)
    parser.add_argument("--alns-dense-teacher-rank-tau", type=float, default=2.0)
    parser.add_argument(
        "--use-alns-route-potential",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, use the ALNS route as a route-level reference potential "
            "over matched teacher edges instead of strict step-by-step BC."
        ),
    )
    parser.add_argument("--alns-route-potential-coef", type=float, default=0.50)
    parser.add_argument("--alns-route-potential-clip", type=float, default=0.05)
    parser.add_argument(
        "--use-alns-return-redistribution",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, redistribute ALNS teacher return/gap over route-level "
            "customer decisions after rollout and before GAE."
        ),
    )
    parser.add_argument(
        "--alns-rr-signal-source",
        type=str,
        default="hybrid",
        choices=["env_teacher", "gap", "hybrid"],
        help=(
            "env_teacher redistributes the sparse teacher component already "
            "emitted by the env; gap adds a clipped log objective gap; hybrid "
            "does both while replacing the original sparse teacher component."
        ),
    )
    parser.add_argument("--alns-rr-coef", type=float, default=0.50)
    parser.add_argument("--alns-rr-clip", type=float, default=0.25)
    parser.add_argument("--alns-rr-temperature", type=float, default=0.20)
    parser.add_argument("--alns-rr-fail-value", type=float, default=0.25)
    parser.add_argument(
        "--alns-rr-credit-mode",
        type=str,
        default="route_coord",
        choices=["route_coord", "uniform"],
        help=(
            "route_coord assigns more negative credit to customers whose "
            "within-route position deviates from ALNS; uniform only spreads "
            "the signal over customer visits."
        ),
    )
    parser.add_argument(
        "--alns-rr-normalize-route-distance",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
    )

    # =========================================================
    # Decomposed reward advantage / multi-head critic
    # =========================================================
    parser.add_argument(
        "--use-decomposed-reward-adv",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, train a multi-head critic for decomposed reward "
            "components and combine their normalized advantages for PPO."
        ),
    )
    parser.add_argument(
        "--decomposed-reward-mode",
        type=str,
        default="objective_progress_terminal_teacher",
        choices=[
            "objective_progress_terminal_teacher",
            "objective_progress",
            "legacy",
            "task_terminal_teacher",
        ],
        help=(
            "objective_progress_terminal_teacher uses four heads: distance "
            "objective, PBRS/progress, terminal success/failure, and teacher. "
            "legacy/task_terminal_teacher keeps the old three-head split."
        ),
    )
    parser.add_argument("--adv-objective-weight", type=float, default=None)
    parser.add_argument("--adv-progress-weight", type=float, default=None)
    parser.add_argument("--adv-task-weight", type=float, default=1.0)
    parser.add_argument("--adv-terminal-weight", type=float, default=1.0)
    parser.add_argument("--adv-teacher-weight", type=float, default=1.0)
    parser.add_argument("--use-adaptive-adv-weights", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--adv-adapt-success-threshold", type=float, default=0.85)
    parser.add_argument("--adv-adapt-success-temperature", type=float, default=0.05)
    parser.add_argument("--adv-objective-early-weight", type=float, default=0.20)
    parser.add_argument("--adv-progress-early-weight", type=float, default=0.50)
    parser.add_argument("--adv-task-early-weight", type=float, default=0.50)
    parser.add_argument("--adv-terminal-early-weight", type=float, default=0.20)
    parser.add_argument("--adv-teacher-early-weight", type=float, default=0.10)
    parser.add_argument("--adv-objective-late-weight", type=float, default=0.65)
    parser.add_argument("--adv-progress-late-weight", type=float, default=0.20)
    parser.add_argument("--adv-task-late-weight", type=float, default=0.20)
    parser.add_argument("--adv-terminal-late-weight", type=float, default=0.10)
    parser.add_argument("--adv-teacher-late-weight", type=float, default=0.05)

    # =========================================================
    # Teacher Reward Mode
    # =========================================================
    parser.add_argument(
        "--teacher-reward-mode",
        type=str,
        default="success_coef",
        choices=["success_coef", "scaled_success", "additive", "none"],
    )

    parser.add_argument(
        "--plain-success-bonus-coef",
        type=float,
        default=1.0,
        help=(
            "Success bonus coefficient when no valid teacher is available. "
            "For strict teacher-only terminal success reward, set this to 0.0."
        ),
    )

    # Old additive teacher reward parameters.
    parser.add_argument("--teacher-penalty-ratio", type=float, default=0.50)
    parser.add_argument("--teacher-bonus-ratio", type=float, default=0.20)
    parser.add_argument("--teacher-failure-ratio", type=float, default=0.50)
    parser.add_argument("--teacher-reward-temp", type=float, default=0.20)
    parser.add_argument("--teacher-reward-margin", type=float, default=0.01)
    parser.add_argument("--teacher-closure-failure-min-ratio", type=float, default=0.10)

    # =========================================================
    # Teacher-scaled Success Bonus
    # =========================================================
    parser.add_argument(
        "--teacher-success-coef-type",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear", "saturation", "convex"],
    )

    parser.add_argument("--teacher-success-worse-floor", type=float, default=0.50)
    parser.add_argument("--teacher-success-equal-coef", type=float, default=1.0)
    parser.add_argument("--teacher-success-max-coef", type=float, default=1.5)
    parser.add_argument("--teacher-success-improve-scale", type=float, default=0.10)
    parser.add_argument("--teacher-success-target-improve", type=float, default=0.20)
    parser.add_argument("--teacher-success-power", type=float, default=1.5)
    parser.add_argument("--teacher-success-worse-temp", type=float, default=0.20)
    parser.add_argument(
        "--teacher-gap-worse-only",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "If True, teacher-calibrated success only penalizes routes worse "
            "than teacher and never pulls better-than-teacher routes back."
        ),
    )

    parser.add_argument(
        "--teacher-scaled-failure",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
    )

    # =========================================================
    # Debug
    # =========================================================
    parser.add_argument("--debug", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--debug-test", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument(
        "--quiet-teacher-diagnostics",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="Suppress offline/teacher diagnostic blocks in training logs.",
    )

    args = parser.parse_args()

    # =========================================================
    # Safety checks
    # =========================================================
    if args.num_envs % args.num_minibatches != 0:
        raise ValueError(
            f"num_envs={args.num_envs} must be divisible by "
            f"num_minibatches={args.num_minibatches}."
        )

    if args.adv_adapt_success_temperature <= 0:
        raise ValueError("adv_adapt_success_temperature must be positive.")

    if args.train_config_log_freq <= 0:
        raise ValueError("train_config_log_freq must be positive.")

    for name in [
        "adv_objective_early_weight",
        "adv_progress_early_weight",
        "adv_task_early_weight",
        "adv_terminal_early_weight",
        "adv_teacher_early_weight",
        "adv_objective_late_weight",
        "adv_progress_late_weight",
        "adv_task_late_weight",
        "adv_terminal_late_weight",
        "adv_teacher_late_weight",
    ]:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative.")

    if args.alns_teacher_batch_size <= 0:
        raise ValueError("alns_teacher_batch_size must be positive.")

    if args.alns_teacher_batch_size % args.num_minibatches != 0:
        raise ValueError(
            f"alns_teacher_batch_size={args.alns_teacher_batch_size} must be divisible by "
            f"num_minibatches={args.num_minibatches}."
        )

    if args.use_alns_teacher and args.alns_teacher_freq <= 0:
        raise ValueError("use_alns_teacher=True requires alns_teacher_freq > 0.")

    if args.alns_obj_scale <= 0:
        raise ValueError("alns_obj_scale must be positive.")

    if args.alns_teacher_filter_margin < 0:
        raise ValueError("alns_teacher_filter_margin must be non-negative.")

    if args.alns_bc_coef < 0:
        raise ValueError("alns_bc_coef must be non-negative.")

    if args.alns_bc_batch_size <= 0:
        raise ValueError("alns_bc_batch_size must be positive.")

    if args.alns_bc_max_steps <= 0:
        raise ValueError("alns_bc_max_steps must be positive.")
    if args.alns_pref_coef < 0:
        raise ValueError("alns_pref_coef must be non-negative.")
    if args.alns_pref_beta <= 0:
        raise ValueError("alns_pref_beta must be positive.")
    if args.alns_pref_batch_size <= 0:
        raise ValueError("alns_pref_batch_size must be positive.")
    if args.alns_pref_max_steps <= 0:
        raise ValueError("alns_pref_max_steps must be positive.")
    if args.alns_soft_kl_coef < 0:
        raise ValueError("alns_soft_kl_coef must be non-negative.")
    if args.alns_soft_kl_batch_size <= 0:
        raise ValueError("alns_soft_kl_batch_size must be positive.")
    if args.alns_soft_kl_max_steps <= 0:
        raise ValueError("alns_soft_kl_max_steps must be positive.")
    if args.alns_soft_kl_lookahead <= 0:
        raise ValueError("alns_soft_kl_lookahead must be positive.")
    if args.alns_soft_kl_tau <= 0:
        raise ValueError("alns_soft_kl_tau must be positive.")
    if args.progress_pbrs_coef < 0:
        raise ValueError("progress_pbrs_coef must be non-negative.")
    if args.progress_pbrs_beta <= 0:
        raise ValueError("progress_pbrs_beta must be positive.")
    if args.repair_fail_coef < 0:
        raise ValueError("repair_fail_coef must be non-negative.")
    if args.repair_progress_coef < 0:
        raise ValueError("repair_progress_coef must be non-negative.")
    if args.repair_success_bonus < 0:
        raise ValueError("repair_success_bonus must be non-negative.")
    if args.alns_gap_gate_temp <= 0:
        raise ValueError("alns_gap_gate_temp must be positive.")
    if args.alns_dense_teacher_coef < 0:
        raise ValueError("alns_dense_teacher_coef must be non-negative.")
    if args.alns_dense_teacher_lookahead <= 0:
        raise ValueError("alns_dense_teacher_lookahead must be positive.")
    if args.alns_dense_teacher_rank_tau <= 0:
        raise ValueError("alns_dense_teacher_rank_tau must be positive.")
    if args.alns_rr_coef < 0:
        raise ValueError("alns_rr_coef must be non-negative.")
    if args.alns_rr_clip <= 0:
        raise ValueError("alns_rr_clip must be positive.")
    if args.alns_rr_temperature <= 0:
        raise ValueError("alns_rr_temperature must be positive.")
    if args.alns_rr_fail_value < 0:
        raise ValueError("alns_rr_fail_value must be non-negative.")

    if args.teacher_reward_mode == "none":
        args.use_alns_teacher = False

    # Backward-compatible CLI aliases used by auto_run.sh.
    if args.teacher_reward_mode == "success_coef":
        args.teacher_reward_mode = "scaled_success"

    if args.reward_mode == "vanilla":
        args.use_bsrs = False
        args.use_split_bsrs = False
        args.use_alns_teacher = False
        args.use_alns_bc = False
        args.use_alns_preference = False
        args.use_alns_soft_kl = False
        args.use_alns_dense_teacher = False
        args.use_alns_route_potential = False
        args.use_alns_gap_gate = False
        args.teacher_reward_mode = "none"

    if args.teacher_success_coef_type == "nonlinear":
        args.teacher_success_coef_type = "saturation"

    if args.teacher_success_worse_floor < 0:
        raise ValueError("teacher_success_worse_floor must be non-negative.")

    if args.teacher_success_equal_coef < 0:
        raise ValueError("teacher_success_equal_coef must be non-negative.")

    if args.teacher_success_max_coef < args.teacher_success_equal_coef:
        raise ValueError(
            "teacher_success_max_coef should be >= teacher_success_equal_coef."
        )

    if args.teacher_success_improve_scale <= 0:
        raise ValueError("teacher_success_improve_scale must be positive.")

    if args.teacher_success_target_improve <= 0:
        raise ValueError("teacher_success_target_improve must be positive.")

    if args.teacher_success_power <= 0:
        raise ValueError("teacher_success_power must be positive.")

    if args.teacher_success_worse_temp <= 0:
        raise ValueError("teacher_success_worse_temp must be positive.")

    if args.plain_success_bonus_coef < 0:
        raise ValueError("plain_success_bonus_coef must be non-negative.")
    if args.bsrs_clip < 0:
        # Negative clipping is ambiguous; use 0 to mean unclipped.
        args.bsrs_clip = 0.0
    if args.bsrs_warmup_updates < 0:
        raise ValueError("bsrs_warmup_updates must be non-negative.")
    if args.bsrs_customer_eta is None:
        args.bsrs_customer_eta = args.bsrs_eta
    if args.bsrs_customer_eta < 0:
        raise ValueError("bsrs_customer_eta must be non-negative.")
    if args.bsrs_rs_eta < 0:
        raise ValueError("bsrs_rs_eta must be non-negative.")

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
