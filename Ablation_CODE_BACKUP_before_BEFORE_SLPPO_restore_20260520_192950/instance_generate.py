from evrptw_gen import InstanceGenerator
from evrptw_gen.configs.load_config import Config
from evrptw_gen.utils.nodes_generator_scheduler import NodesGeneratorScheduler
import copy
import os
import pickle


def _generate_balanced_eval(args):
    """Generate a fixed C/R/RC x loose/strain evaluation set."""
    if args.eval_instances_per_type % 2 != 0:
        raise ValueError("--eval_instances_per_type must be even.")

    config = Config(args.config_path)
    gen = InstanceGenerator(
        args.config_path,
        config=config,
        save_path=None,
        num_instances=1,
        plot_instances=False,
        seed=args.eval_seed,
    )

    perturb_dict = {}
    if args.add_perturb:
        perturb_cfg = Config(args.perturb_config_path).setup_env_parameters()
        perturb_dict = perturb_cfg.get("perturb", {})

    base_env = copy.deepcopy(gen.env)
    per_tw = args.eval_instances_per_type // 2
    tw_splits = [("wide", "loose"), ("narrow", "strain")]
    instances = []
    group_counts = {}

    for instance_type in ("C", "R", "RC"):
        for tw_policy, tw_alias in tw_splits:
            group_name = f"{instance_type}_{tw_alias}"
            group_counts[group_name] = 0

            for _ in range(per_tw):
                env = copy.deepcopy(base_env)
                env["num_customers"] = int(args.eval_customer_num)
                env["num_charging_stations"] = int(args.eval_cs_num)
                env["test_instance_type"] = instance_type
                env["time_window_policy"] = tw_policy
                env["eval_group"] = group_name
                env["eval_tw_alias"] = tw_alias

                inst = gen.generate_tensors(
                    env=env,
                    perturb_dict=perturb_dict,
                    num_customers=int(args.eval_customer_num),
                    num_charging_stations=int(args.eval_cs_num),
                )
                inst["id"] = len(instances)
                inst["eval_group"] = group_name
                inst["env"]["eval_group"] = group_name
                inst["env"]["eval_tw_alias"] = tw_alias
                instances.append(inst)
                group_counts[group_name] += 1

    pickle_dir = os.path.join(args.save_path, "pickle")
    os.makedirs(pickle_dir, exist_ok=True)

    output_name = args.eval_output_name
    if output_name is None:
        output_name = (
            f"evrptw_{args.eval_customer_num}C_{args.eval_cs_num}R_"
            f"CRC_tw{len(instances)}.pkl"
        )

    output_path = os.path.join(pickle_dir, output_name)
    with open(output_path, "wb") as f:
        pickle.dump(instances, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[BalancedEval] saved {len(instances)} instances -> {output_path}")
    for group_name in sorted(group_counts):
        print(f"[BalancedEval] {group_name}: {group_counts[group_name]}")
    return output_path

def main(args):
    if getattr(args, "balanced_eval", False):
        return _generate_balanced_eval(args)

    config_path = args.config_path
    save_path = args.save_path
    num_instances = args.num_instances
    plot_instances = args.plot_instances
    customer_range = args.customer_range
    node_generate_policy = args.node_generate_policy
    cus_per_cs = args.cus_per_cs
    perturb_config_path = args.perturb_config_path
    save_format = args.save_format

    min_customer_num, max_customer_num = customer_range

    if save_format == "all":
        save_template_solomon = True
        save_template_pickle = True
    elif save_format == "solomon":
        save_template_solomon = True
        save_template_pickle = False
    elif save_format == "pickle":
        save_template_solomon = False
        save_template_pickle = True
    else:
        raise ValueError(f"Unknown save_format: {save_format}")

    gen = InstanceGenerator(
        config_path,
        save_path=save_path,
        num_instances=num_instances,
        plot_instances=plot_instances,
        seed=args.seed,
    )
    nodes_generator_scheduler = NodesGeneratorScheduler(
        min_customer_num=min_customer_num,
        max_customer_num=max_customer_num,
        cus_per_cs=cus_per_cs,
        seed=args.seed,
    )
    perturb_dict = Config(perturb_config_path).setup_env_parameters()

    gen.generate(perturb_dict=perturb_dict['perturb'],
                save_template_solomon=save_template_solomon,
                save_template_pickle=save_template_pickle,
                node_generater_scheduler=nodes_generator_scheduler,
                node_generate_policy=node_generate_policy) 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate EVRPTW instances based on a configuration file.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./evrptw_gen/configs/config.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./dataset/unanchored/Cus_50/buffer",
        help="Directory to save generated instances."
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=1024,
        help="Number of instances to generate."
    )
    parser.add_argument(
        "--plot_instances",
        action='store_true',
        help="Whether to plot the generated instances."
    )
    parser.add_argument(
        "--customer_range",
        type=int,
        nargs=2,
        default=[50, 50],
        help="Range of number of customers (min max)."
    )
    parser.add_argument(
        "--node_generate_policy",
        type=str,
        default="fixed",
        help="Node generation policy: 'fixed' or 'linear'."
    )
    parser.add_argument(
        "--cus_per_cs",
        type=int,
        default=4,
        help="Number of customers per charging station (used if node_generate_policy is 'fixed_cus_per_cs')."
    )
    parser.add_argument(
        "--add_perturb",
        action='store_true',
        help="Whether to add perturbations to the instances."
    )
    parser.add_argument(
        "--perturb_config_path",
        type=str,
        default="./evrptw_gen/configs/perturb_config.yaml",
        help="Path to the perturbation configuration file."
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="all",
        help="Format to save instances: 'all', 'solomon', 'pickle'."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base RNG seed. With a seed, each generated instance uses a "
            "deterministic derived seed; without one, runs use OS entropy."
        ),
    )
    parser.add_argument(
        "--balanced_eval",
        action="store_true",
        help="Generate fixed 200 C/R/RC eval sets split half loose and half strain TW.",
    )
    parser.add_argument(
        "--eval_instances_per_type",
        type=int,
        default=200,
        help="Number of eval instances for each C/R/RC type.",
    )
    parser.add_argument(
        "--eval_customer_num",
        type=int,
        default=50,
        help="Customer count for balanced eval generation.",
    )
    parser.add_argument(
        "--eval_cs_num",
        type=int,
        default=12,
        help="Charging station count for balanced eval generation.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=2025,
        help="RNG seed for balanced eval generation.",
    )
    parser.add_argument(
        "--eval_output_name",
        type=str,
        default=None,
        help="Optional pickle filename for balanced eval generation.",
    )


    args = parser.parse_args()
    main(args)
