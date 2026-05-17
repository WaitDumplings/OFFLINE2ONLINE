import argparse
import json
import sys

import gym
import numpy as np

from evrptw_gen.benchmarks.DRL_Solver.DRL_train import parse_args as parse_train_args
from evrptw_gen.benchmarks.DRL_Solver.train import (
    _alns_record_to_action_sequence,
    _load_alns_buffer_from_dir,
    _make_alns_train_envs,
)
from evrptw_gen.configs.load_config import Config


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alns-buffer-dir", default="./dataset/unanchored/Cus_50/buffer/")
    parser.add_argument("--alns-progress-name", default="buffer_progress.pkl")
    parser.add_argument("--alns-instance-pickle-name", default="evrptw_50C_12R.pkl")
    parser.add_argument("--config-path", default="./evrptw_gen/configs/config.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-json", default="./logs_result_online_alns_dpo/expert_route_mask_validity.json")
    return parser.parse_args()


def make_args(cli):
    old_argv = sys.argv
    try:
        sys.argv = ["check_expert_route_mask_validity.py"]
        args = parse_train_args()
    finally:
        sys.argv = old_argv

    args.seed = int(cli.seed)
    args.train_cus_num = 50
    args.train_cs_num = 12
    args.n_traj = 1
    args.num_steps = int(cli.max_steps)
    args.num_envs = int(cli.batch_size)
    args.config_path = cli.config_path
    args.alns_buffer_dir = cli.alns_buffer_dir
    args.alns_progress_name = cli.alns_progress_name
    args.alns_instance_pickle_name = cli.alns_instance_pickle_name
    args.reward_mode = "vanilla"
    args.lambda_fail_init = 10.0
    args.debug = False
    args.debug_test = False
    return args


def register_env(args):
    try:
        gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    except Exception:
        pass


def strip_leading_depots(seq):
    out = [int(x) for x in list(seq or [])]
    while len(out) > 0 and out[0] == 0:
        out = out[1:]
    return out


def check_batch(args, config, batch, global_start, max_steps):
    num_nodes = int(args.train_cus_num + args.train_cs_num + 1)
    sequences = []
    empty_records = []
    for offset, record in enumerate(batch):
        seq = record.get("teacher_action_sequence", None)
        if seq is None:
            seq = _alns_record_to_action_sequence(
                record,
                num_customers=int(args.train_cus_num),
                num_nodes=num_nodes,
            )
        seq = strip_leading_depots(seq)
        record["teacher_action_sequence"] = seq
        sequences.append(seq)
        if len(seq) == 0:
            empty_records.append(global_start + offset)

    envs = None
    invalids = []
    finished_by_sequence = np.zeros(len(batch), dtype=np.bool_)
    done_by_env = np.zeros(len(batch), dtype=np.bool_)
    steps_checked = np.zeros(len(batch), dtype=np.int32)

    try:
        envs, sampled = _make_alns_train_envs(
            args=args,
            config=config,
            update_step=0,
            sampled_records=batch,
            use_teacher_reward=False,
            seed_offset=830000 + global_start,
        )
        obs = envs.reset()
        active = np.ones(len(batch), dtype=np.bool_)

        for step in range(int(max_steps)):
            target = np.zeros((len(batch), 1), dtype=np.int64)
            has_target = np.zeros(len(batch), dtype=np.bool_)
            for i, seq in enumerate(sequences):
                if active[i] and step < len(seq):
                    target[i, 0] = int(seq[step])
                    has_target[i] = True
                elif active[i]:
                    finished_by_sequence[i] = True

            if not np.any(has_target):
                break

            action_mask = np.asarray(obs["action_mask"], dtype=np.bool_)
            feasible = action_mask[np.arange(len(batch)), 0, target[:, 0]]
            bad = has_target & active & (~feasible)
            for local_idx in np.where(bad)[0]:
                invalids.append(
                    {
                        "record_index": int(global_start + local_idx),
                        "local_index": int(local_idx),
                        "step": int(step),
                        "action": int(target[local_idx, 0]),
                        "seq_head": sequences[local_idx][:20],
                        "seq_len": int(len(sequences[local_idx])),
                        "allowed_count": int(action_mask[local_idx, 0].sum()),
                        "allowed_head": [
                            int(x)
                            for x in np.where(action_mask[local_idx, 0])[0][:20].tolist()
                        ],
                    }
                )

            valid = has_target & active & feasible
            steps_checked[valid] += 1

            forced_action = target.copy()
            forced_action[~valid, 0] = 0
            obs, _, done, _ = envs.step(forced_action)
            done_flat = np.asarray(done).reshape(-1)
            done_by_env |= done_flat
            active = active & valid & (~done_flat)
            if not np.any(active):
                break
    finally:
        if envs is not None:
            envs.close()

    return {
        "invalids": invalids,
        "empty_records": empty_records,
        "steps_checked": steps_checked.tolist(),
        "done_by_env": done_by_env.tolist(),
        "finished_by_sequence": finished_by_sequence.tolist(),
        "sequence_lengths": [len(seq) for seq in sequences],
        "starts_with_depot": [bool(len(seq) > 0 and seq[0] == 0) for seq in sequences],
    }


def main():
    cli = parse_cli()
    args = make_args(cli)
    register_env(args)
    config = Config(cli.config_path)
    records = _load_alns_buffer_from_dir(
        args.alns_buffer_dir,
        progress_name=args.alns_progress_name,
        instance_pickle_name=args.alns_instance_pickle_name,
    )
    if records is None:
        raise RuntimeError("failed to load ALNS buffer")

    records = list(records)
    if int(cli.max_records) > 0:
        records = records[: int(cli.max_records)]

    all_invalids = []
    empty_records = []
    starts_with_depot = 0
    total_steps = 0
    total_seq_len = 0
    done_count = 0

    for start in range(0, len(records), int(cli.batch_size)):
        batch = records[start : start + int(cli.batch_size)]
        result = check_batch(args, config, batch, start, int(cli.max_steps))
        all_invalids.extend(result["invalids"])
        empty_records.extend(result["empty_records"])
        starts_with_depot += int(sum(result["starts_with_depot"]))
        total_steps += int(sum(result["steps_checked"]))
        total_seq_len += int(sum(result["sequence_lengths"]))
        done_count += int(sum(result["done_by_env"]))

        print(
            "[ExpertMaskCheck] "
            + json.dumps(
                {
                    "checked": min(start + len(batch), len(records)),
                    "total": len(records),
                    "invalid_so_far": len(all_invalids),
                    "empty_so_far": len(empty_records),
                },
                sort_keys=True,
            )
        )

    summary = {
        "buffer_info": {
            "buffer_dir": args.alns_buffer_dir,
            "progress_name": args.alns_progress_name,
            "instance_pickle_name": args.alns_instance_pickle_name,
        },
        "records_checked": len(records),
        "invalid_records": len({x["record_index"] for x in all_invalids}),
        "invalid_actions": len(all_invalids),
        "empty_records": len(empty_records),
        "starts_with_depot_after_conversion": starts_with_depot,
        "done_count": done_count,
        "total_steps_checked": total_steps,
        "total_sequence_length": total_seq_len,
        "first_invalids": all_invalids[:20],
        "first_empty_records": empty_records[:20],
    }
    with open(cli.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[ExpertMaskCheckSummary] " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
