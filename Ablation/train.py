"""
Unified entry point for EVRPTW benchmark training.
Can be executed as:
    python run_benchmark.py
No need for -m or changing working directory.
"""

import os
import sys

# -------------------------------------------------------------------
# 1) load evrptw_gen to sys.path
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR   
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# -------------------------------------------------------------------
# 2) Load benchmarks/train.py 
# -------------------------------------------------------------------
from evrptw_gen.benchmarks.DRL_Solver.train import train
from evrptw_gen.benchmarks.DRL_Solver.DRL_train import parse_args


# -------------------------------------------------------------------
# 3) Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
