# EVRP-TW-D-B

**EVRP-TW-D-B** is an industrial-scale, unit-consistent **dataset-and-benchmark suite** for the **Electric Vehicle Routing Problem with Time Windows (EVRP-TW)**. It provides a configurable **instance generator (EVRP-TW-D)** and unified **evaluation utilities (EVRP-TW-B)** to enable reproducible, apples-to-apples comparisons under consistent operational semantics (**explicit physical units** for distance/energy/charging and **protocol-consistent** feasibility checks).

The complete dataset is available at [URL](https://zenodo.org/records/18529191).

> **Tagline:** *Industrial-scale EVRP-TW generator + benchmark utilities with unit-consistent physical semantics.*

---

## Paper status (under review)

**Title:** *From Theory to Practice: An Industrial-Scale Benchmark for Electric Vehicle Routing with Time Windows*

This repository is prepared for **single-blind review**. To preserve anonymity, the README currently omits author metadata and external project links. Citation metadata will be added after the review stage.

---

## What this repository provides

- **EVRP-TW-D (Dataset / Generator):** a policy-driven instance generator with modular components for spatial layouts, time-window regimes, demand/service-time sampling, and physical parameter instantiation.
- **EVRP-TW-B (Benchmark utilities):** evaluator and utilities for solver-agnostic comparison under unified feasibility and objective semantics. Convenience scripts (`train.py`, `eval.py`) are included to streamline DRL workflows for benchmark experiments.
- **Configuration-driven workflows:** `configs/` specifies scale, time-window distributions, charging/vehicle profiles, and environment-like metadata for controlled dataset generation.

---

## Repository structure

Current tree:

```text
.
├── eval.py                # convenience script for evaluation (esp. DRL workflows)
├── docs/                  # documentation (formats, benchmark usage, release notes)
│   └── FORMAT.md           # Instance format specification (nodes / parameters / metadata)
├── evrptw_gen             # main package (core implementation)
│   ├── __init__.py
│   ├── benchmarks         # benchmark utilities & solver integration helpers
│   ├── configs            # configuration references for dataset instantiation
│   ├── generator.py       # core instance generation pipeline (InstanceGenerator)
│   ├── policies           # modular generation policies (spatial / TW / attribute regimes)
│   └── utils              # shared helpers (I/O, geometry, seeding, validation, etc.)
├── instance_generate.py   # user-facing entry point (CLI wrapper around the generator)
├── LICENSE                # repository license
├── README.md              # project overview and usage
└── train.py               # convenience script for training (DRL solver experiments)
```

## ⚠️ NumPy Version Requirement

**Important:** This project **requires NumPy version `1.26.4`**.

Both the **dataset generator** and the **DRL solvers** depend on NumPy `1.26.4`.  
Using other NumPy versions may lead to **runtime errors, incompatible pickle files, or unexpected numerical behavior**, especially when loading serialized datasets or running evaluation scripts.

### Recommended setup

```bash
pip install numpy==1.26.4
```

## Generating data (EVRP-TW-D)

### (1) Generate a fixed scale (e.g., **N = 100**)

```shell
python instance_generate.py \
  --config_path ./evrptw_gen/configs/config.yaml \
  --save_path ./eval/Cus_100/ \
  --num_instances 1000 \
  --customer_range 100 100 \
  --cus_per_cs 5 \
  --node_generate_policy fixed \
  --save_format all
```

### (2) Generate a customer range (e.g., **N ∈ [80, 120]**)

```shell
python instance_generate.py \
  --config_path ./evrptw_gen/configs/config.yaml \
  --save_path ./eval/Cus_80_120/ \
  --num_instances 1000 \
  --customer_range 80 120 \
  --cus_per_cs 5 \
  --node_generate_policy fixed \
  --save_format all
```
### Script arguments (reference)

`instance_generate.py` supports the following CLI flags:

| Argument | Type | Default | Description |
|---|---:|---|---|
| `--config_path` | `str` | `./evrptw_gen/configs/config.yaml` | Path to the main generator config. |
| `--save_path` | `str` | `./dataset/Cus_100/` | Output directory for generated instances. |
| `--num_instances` | `int` | `1000` | Number of instances to generate. |
| `--plot_instances` | `flag` | `False` | If set, plot generated instances. |
| `--customer_range` | `int int` | `100 100` | Customer count range: `MIN MAX`. |
| `--node_generate_policy` | `str` | `fixed` | Node-generation policy (e.g., `fixed`, `linear`, etc.). |
| `--cus_per_cs` | `int` | `5` | Customers per charging station (used by the node scheduler). |
| `--perturb_config_path` | `str` | `./evrptw_gen/configs/perturb_config.yaml` | Path to the perturbation config. |
| `--save_format` | `str` | `all` | Output format: `all` \| `solomon` \| `pickle`. |
| `--add_perturb` | `flag` | `False` | Parsed but currently **not used** (see note below). |

> **Implementation note:** `--add_perturb` is currently parsed but **not used for fixed-scale instance generation**. We always load the perturbation config and pass it to `gen.generate(...)` because **batch-parallel training** requires instances within the same batch to share the **same numbers of customers and charging stations**, which improves training efficiency and stability. In other words, perturbations are intended to be applied **within a fixed-scale batch** during training, rather than toggled at the dataset generation stage.
---

### Output artifacts

`instance_generate.py` can save instances in one or both formats via `--save_format`:

- **Solomon-style text templates** (`--save_format solomon`): compatible with classic VRPTW-style tooling.
- **Pickle** (`--save_format pickle`): Python-native format for fast loading in training/evaluation pipelines.
- **Both** (`--save_format all`, default).

See the [`docs/FORMAT.md`](docs/FORMAT.md) file for details.

> The exact filenames and directory layout are defined by your generator/config settings.  

## Training and evaluation utilities (EVRP-TW-B)
For detailed usage instructions, experiment workflows, and solver-specific
configurations, please refer to:
- **[`evrptw_gen/benchmarks/README.md`](evrptw_gen/benchmarks/README.md)**


## License

This project is released under the **Apache License 2.0**.
See the [LICENSE](LICENSE) file for details.
