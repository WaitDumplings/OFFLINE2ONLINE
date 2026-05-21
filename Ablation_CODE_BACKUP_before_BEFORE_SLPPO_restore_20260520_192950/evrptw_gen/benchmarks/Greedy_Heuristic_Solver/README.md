# Greedy_Heuristic_Solver — EVRP-TW-B

This folder provides a **fast constructive heuristic baseline** for EVRP-TW under
the **EVRP-TW-B** unified semantics. The greedy solver is intended for:

- quick **deliverability sanity checks** (coverage / route construction),
- lightweight baselines under strict budgets,
- generating routes/logs for debugging parsers and evaluators.

> **Note (CLI string):** the argparse description currently mentions “VNSTSolver”,
> but the script instantiates `GreedySolver`. This README documents the **actual**
> behavior of the code as provided.

---

## Greedy baseline algorithm
We include a **deterministic greedy heuristic** as a lightweight baseline and as the **anchor solver** used in RCR-style evaluation. The goal of this heuristic is **speed and robustness**, not high solution quality.

### High-level idea

Starting from the depot, the heuristic repeatedly selects the **nearest feasible unserved customer** and visits it. A customer is considered *feasible* if:

1. **Time window & capacity feasibility:** the vehicle can reach and serve the customer within its time window, and the remaining capacity suffices.
2. **Return feasibility:** after serving the customer, the vehicle can still return to the depot **either directly or via charging-station hops** under the **full-charging** assumption.

If no feasible customer exists for the current vehicle, the vehicle returns to the depot and a **new vehicle** is dispatched. The procedure stops when all customers are served, or when even a newly dispatched vehicle cannot serve any remaining customer feasibly.

### Charging-station hops (stop-graph)

To handle cases where a vehicle cannot directly reach a customer, the heuristic allows **multi-hop travel via charging stations**:

- Build a *stop-graph* over `{Depot} ∪ {Charging Stations}`.
- Precompute all-pairs shortest paths on this graph **once** (edge costs account for travel time, and when arriving at a charging station, the time to **recharge to full**).
- During construction, reaching a customer may follow a sequence of charging-station hops and then a final leg to the customer; after service, the heuristic also verifies that a feasible return path to the depot exists.

### Pseudocode (informal)

- Precompute shortest paths on the stop-graph.
- While there are unserved customers:
  - Start a new route at the depot with full battery.
  - Repeatedly:
    - Sort unserved customers by distance from current location.
    - Pick the **first feasible** customer in that order (time window, capacity, reachability with CS hops, and return-to-depot feasibility).
    - Travel (possibly via CS hops), wait if early, serve, and update time/battery/capacity.
  - Return to depot; if no customer was served in this route, terminate.

> This greedy baseline is intentionally simple and fast. It is useful as a sanity-check solver and as a consistent, deterministic reference point in benchmark workflows.

## What it does

For each Solomon-style `.txt` instance in a directory, the script:

1. loads the instance via `utils.load_instances.load_instance`,
2. runs `GreedySolver(instance).solve()` to produce a **list of routes**,
3. records runtime and summary statistics,
4. writes logs to disk:
   - a **summary table** (one row per instance),
   - a **route table** (one row per route),
   - and a list of instances with unvisited customers.

---

## Requirements

- Python
- `pandas`
- Internal modules:
  - `solver.GreedySolver`
  - `utils.load_instances.load_instance`
  - `utils.helpers.set_random_seed`
  - (optional) `utils.helpers.plot_solution` (imported but not used by default)

---

## Input format

- Expects a directory containing **Solomon-style** instance files with suffix `.txt`:
  - `--instance_path <DIR>`
- Each file is parsed by `load_instance(instance_path)` into an in-memory dict-like instance.

---

## Output artifacts

All outputs are written under `--save_log_path`.

### 1) `greedy_summary.csv` (one row per instance)

Columns:

- `instance_id`: from `instance["instance_id"]` if present, otherwise filename stem
- `file`: original `.txt` filename
- `fleet_size`: number of routes returned by the solver
- `objective_value`: `solver.global_value` (if present) as float
- `elapsed_time_s`: wall-clock runtime (seconds)
- `visited_all`: whether all customers were visited (`all(solver.visited)`)
- `unvisited_count`: number of unvisited customers
- `routes_json`: JSON string of the full solution (list of routes)

### 2) `greedy_routes.csv` (one row per route)

Columns:

- `instance_id`, `file`
- `route_idx`: route index within the solution
- `route_json`: JSON string for the route
- `route_str`: human-readable string fallback

### 3) `unvisited_instances.txt`

A plain text list of `.txt` filenames for which `visited_all == False`.

### Optional: Parquet outputs

The script attempts to write:

- `greedy_summary.parquet`
- `greedy_routes.parquet`

If `pyarrow` / `fastparquet` is not installed, it prints a warning and skips.

---

## Usage

### Quickstart

```bash
python main.py \
  --instance_path ../../../dataset/anchored/Cus_15/solomon \
  --seed 1234 \
  --save_log_path ./logs
```

## CLI arguments
| Argument          | Default                        | Description                           |
| ----------------- | ------------------------------ | ------------------------------------- |
| `--instance_path` | `../../../dataset/anchored/Cus_15/solomon` | Directory containing `.txt` instances |
| `--seed`          | `1234`                         | Random seed for reproducibility       |
| `--save_log_path` | `./logs`                       | Output directory for logs/tables      |

## Notes

- **Feasibility proxy:** the script checks `visited_all = all(solver.visited)` to detect coverage gaps.  
  This is a quick deliverability diagnostic; full feasibility should still be validated by the unified evaluator.

- **Objective meaning:** `objective_value` is taken from `solver.global_value` (if present).  
  Ensure this aligns with the benchmark’s objective (e.g., total travel distance) before reporting it directly.

- **Plotting:** `plot_solution` is imported but not used in the current script; keep it optional for debugging.

