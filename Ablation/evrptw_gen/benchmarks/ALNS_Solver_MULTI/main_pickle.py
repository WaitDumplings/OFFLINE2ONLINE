import pickle
import time
import argparse
import os
from solver import ALNS_Solver
from utils.helpers import set_random_seed
from utils.load_instances import load_instance
from tqdm import tqdm

import json
import pandas as pd


data = "../../../dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl"

instance = pickle.load(open(data, "rb"))[0]
solver = ALNS_Solver(instance, format="tensor")
solution = solver.solve()
breakpoint()