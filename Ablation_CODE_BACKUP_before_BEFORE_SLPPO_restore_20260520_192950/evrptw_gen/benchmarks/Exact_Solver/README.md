# Exact_Solver (Gurobi / MILP) — EVRP-TW-B

This folder provides an **exact optimization baseline** for EVRP-TW using **Gurobi (MILP)**.
It is primarily used to produce **time-limited incumbents** for benchmarking and diagnostics
under the **EVRP-TW-B** unified semantics.

> **Anytime reporting:** the runner records the best incumbent objective at **1 min** and **5 min**,
> and reports the final incumbent at **15 min** (TimeLimit).

---

## What it does

`main_distributed.py` runs Gurobi on a batch of **Solomon-style `.txt` instances** and writes a CSV:

- `obj_1min`: best incumbent objective at (or right after) **60s**
- `obj_5min`: best incumbent objective at (or right after) **300s**
- `obj_15min`: final incumbent objective at termination (TimeLimit=**900s**)  
  (If no feasible solution exists by a cutoff, the corresponding value is `None`.)

The run supports **distributed slicing** via `--start_idx/--end_idx` over the sorted file list,
so multiple workers can process disjoint subsets.

---

## Requirements

- Python environment with:
  - `gurobipy`
  - `pandas`
- A valid **Gurobi license** on the machine.

---

## Input format

- Instances must be **Solomon-style text** files (`.txt`).
- The parser is invoked as:
  - `Depot_nodes, Customer_nodes, RS_nodes, parameters = parse_file(file_path)`

---

## Output

A CSV file is generated under `--save_path`:

- `gurobi_anytime_idx_<start>_<end>.csv`

Columns:
- `file`
- `obj_1min`
- `obj_5min`
- `obj_15min`

---

## Usage

### Quickstart (single worker)

```bash
python main_distributed.py \
  --file_dir ../../../dataset/anchored/Cus_100/solomon/ \
  --save_path ./logs_gurobi_0_20 \
  --dummy 10 \
  --time_limit 15
```

## Distributed slicing (multiple workers)
for example, #worker 0,1,2,3
```shell
python main_distributed.py --start_idx 0   --end_idx 50  --save_path ./logs_gurobi_0_50
python main_distributed.py --start_idx 50  --end_idx 100 --save_path ./logs_gurobi_50_100
python main_distributed.py --start_idx 100 --end_idx 150 --save_path ./logs_gurobi_100_150
python main_distributed.py --start_idx 150 --end_idx 200 --save_path ./logs_gurobi_150_200

```

## CLI arguments
| Argument       | Default                          | Description                                         |
| -------------- | -------------------------------- | --------------------------------------------------- |
| `--file_dir`   | `../../../dataset/anchored/Cus_100/solomon/` | Directory containing `.txt` instances               |
| `--dummy`      | `10`                             | Number of RS dummy nodes used in graph construction |
| `--save_path`  | `./logs_gurobi_0_20`             | Output directory for CSV logs                       |
| `--time_limit` | `15`                             | Time limit per instance (**minutes**)               |
| `--start_idx`  | `0`                              | Start index (inclusive) in sorted file list         |
| `--end_idx`    | `-1`                             | End index (exclusive); `-1` means “to the end”      |
| `--verbose`    | `False`                          | Print per-instance progress                         |

## Notes on anytime capture
- The runner uses a Gurobi MIP callback to capture incumbent objective values at the first callback after crossing each threshold (60s, 300s).
- If an instance finishes before a threshold and the callback does not fire, the script fills obj_1min/obj_5min with the final incumbent (if feasible).
