"""
Root entry point for strict incumbent-buffer PPO.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from evrptw_gen.benchmarks.DRL_Solver.incumbent_train import main


if __name__ == "__main__":
    main()
