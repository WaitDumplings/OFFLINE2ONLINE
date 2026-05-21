import time
import argparse
import os
from solver import GreedySolver
from utils.helpers import plot_solution, set_random_seed
from utils.load_instances import load_instance

import json
import pandas as pd
import pickle

data = "../../../dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl"

instance = pickle.load(open(data, "rb"))[0]
solver = GreedySolver(instance, format="tensor")
solution = solver.solve()
breakpoint()