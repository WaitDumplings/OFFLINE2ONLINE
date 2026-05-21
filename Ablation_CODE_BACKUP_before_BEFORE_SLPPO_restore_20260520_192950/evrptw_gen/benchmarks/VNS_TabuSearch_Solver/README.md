### Reference / Origin (VNS/TS)

The VNS/Tabu Search baseline implemented in this folder follows a **classical
VNS+TS meta-heuristic framework** for vehicle routing, adapted to EVRP-TW with
energy and time-window constraints.

The design is based on the algorithmic ideas presented in:
- *A Variable Neighborhood Search Heuristic for the Vehicle Routing Problem*
  (JSTOR: https://www.jstor.org/stable/43666939)

# VNS_TabuSearch_Solver (VNST) — EVRP-TW-B

This folder provides a **VNS/Tabu Search (VNST)** meta-heuristic baseline for EVRP-TW
under the **EVRP-TW-B** unified semantics. It is intended for:

- stronger **budgeted incumbents** than simple greedy heuristics,
- feasibility-first search with subsequent solution refinement,
- producing route-level logs for analysis and debugging.

---

## What it does

For each Solomon-style `.txt` instance in a directory, the script:

1. loads the instance via `utils.load_instances.load_instance`,
2. runs `VNSTSolver(instance, predefine_route_number=...)` to produce a **list of routes**,
3. checks whether **all customers are served** (based on node `type == 'c'`),
4. writes logs to disk:
   - a **summary table** (one row per instance),
   - a **route table** (one row per route),
   - a list of instances with unserved customers,
   - and a list of exceptions (parse/solve failures).

---

## Requirements

- Python
- `pandas`
- Internal modules:
  - `solver.VNSTSolver`
  - `utils.load_instances.load_instance`
  - `utils.helpers.set_random_seed`
  - (optional) `utils.helpers.plot_solution` (imported but not used by default)

---

## Input format

- Expects a directory containing **Solomon-style** instance files with suffix `.txt`:
  - `--instance_dir <DIR>`
- Each file is parsed by `load_instance(instance_path)` into an instance object with:
  - `instance.customers` iterable (customers have `id` and `type` fields),
  - optional `instance.instance_id` attribute.

---

## Output artifacts

All outputs are written under `--save_log_path`.

### 1) `vnst_summary.csv` (one row per instance)

Columns:

- `instance_id`: `instance.instance_id` if available, otherwise filename stem
- `file`: original `.txt` filename
- `fleet_size`: number of routes returned by the solver
- `objective_value`: `solver.global_value` (if present) as float
- `elapsed_time_s`: wall-clock runtime (seconds)
- `served_all`: whether all customers are served (customers have `type == 'c'`)
- `unserved_count`: number of unserved customers
- `routes_ids_json`: JSON string of routes as node-id sequences (`list[list[node_id]]`)

### 2) `vnst_routes.csv` (one row per route)

Columns:

- `instance_id`, `file`
- `route_idx`: route index within the solution
- `route_len`: number of nodes in the route
- `route_ids_json`: JSON string for the node-id sequence
- `route_str`: human-readable route string joined by ` -> `

### 3) `unserved_instances.txt`

A plain text list of `.txt` filenames for which `served_all == False`.

### 4) `exceptions.txt`

A tab-separated list of failures with error strings:

- `<filename>\t<repr(exception)>`

### Optional: Parquet outputs

The script attempts to write:

- `vnst_summary.parquet`
- `vnst_routes.parquet`

If `pyarrow` / `fastparquet` is not installed, it prints a warning and skips.

---

## Usage

## VNS/TS hyperparameters

The VNS/TS meta-heuristic baseline uses the following default hyperparameters:

| Parameter | Value |
|---|---:|
| Tabu tenure (list length) | 30 |
| Tabu iterations per call | 100 |
| Max neighborhood index \(k_{\max}\) | 15 |
| Feasibility-phase iteration budget \(\eta_{\mathrm{feas}}\) | 700 |
| Feasible improvement budget \(\eta_{\mathrm{dist}}\) | 100 |
| Initial penalty weights \((\alpha,\beta,\gamma)\) | (10.0, 10.0, 10.0) |
| Penalty lower bounds \((\alpha_{\min},\beta_{\min},\gamma_{\min})\) | (0.5, 0.75, 1.0) |
| Penalty upper bounds \((\alpha_{\max},\beta_{\max},\gamma_{\max})\) | (5000, 5000, 5000) |
| Penalty update factor \(\delta\) | 1.2 |
| Penalty update interval \(\tau_{\mathrm{pen}}\) | 2 |
| SA cooling parameter \(\delta_{\mathrm{sa}}\) | 0.08 |
| Predefined route number (init) | 3 |

### Quickstart

```bash
python main.py \
  --instance_dir ../../../dataset/anchored/Cus_5/solomon/ \
  --seed 1234 \
  --predefine_route_number 2 \
  --save_log_path ./logs_vnst
```

## CLI arguments
| Argument                   | Default                        | Description                                   |
| -------------------------- | ------------------------------ | --------------------------------------------- |
| `--instance_dir`           | `../../../dataset/anchored/Cus_5/solomon/` | Directory containing `.txt` instances         |
| `--seed`                   | `1234`                         | Random seed for reproducibility               |
| `--predefine_route_number` | `2`                            | Initial number of routes used by `VNSTSolver` |
| `--save_log_path`          | `./logs_vnst`                  | Output directory for logs/tables              |

## Notes
- **Serve-all check:** the script verifies coverage by checking whether every customer (`type == 'c'`)
  is served by at least one route. This is a deliverability diagnostic; full feasibility should still
  be validated by the unified evaluator.

- **Objective meaning:** `objective_value` is taken from `solver.global_value` (if present). Ensure this
  matches the benchmark objective (e.g., total travel distance) before reporting it directly.

- **Route serialization:** routes are converted to node-id sequences for stable logging via
  `_route_to_ids(...)` / `_solution_to_routes_ids(...)`.