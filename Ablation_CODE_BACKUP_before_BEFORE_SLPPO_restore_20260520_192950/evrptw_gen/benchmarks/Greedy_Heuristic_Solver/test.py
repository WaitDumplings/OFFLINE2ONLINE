import time
import argparse
import os
from solver import GreedySolver
from utils.helpers import plot_solution, set_random_seed
from utils.load_instances import load_instance
import pickle
import json
import pandas as pd
import numpy as np

def compare_solver(solver1, solver2):
    print("\nComparing two solvers...")
    print((solver1.customers == solver2.customers).all())
    print((solver1.depot == solver2.depot).all())
    print((solver1.css == solver2.css).all())
    print((solver1.nodes == solver2.nodes).all())
    print((solver1.id_strs == solver2.id_strs))
    print(solver1.num_customers == solver2.num_customers)
    print(solver1.num_cs == solver2.num_cs)
    print(solver1.num_nodes == solver2.num_nodes)
    print(solver1.depot_node == solver2.depot_node)
    print(solver1.cs_start == solver2.cs_start)
    print(solver1.cs_end == solver2.cs_end)
    print(solver1.css_idx == solver2.css_idx)
    print(solver1.velocity == solver2.velocity)
    print(solver1.consume_rate == solver2.consume_rate)
    print(solver1.fuel_cap == solver2.fuel_cap)
    print(solver1.load_cap == solver2.load_cap)
    print((solver1.distance_matrix == solver2.distance_matrix).all())
    print(abs(np.sum(solver1.demands - solver2.demands)< 1e-6))
    print(solver1.working_start == solver2.working_start)

    # 真实值 vs 预处理值
    print(solver1.charging_power == solver2.charging_power)

    # hour vs minutes
    
    # print(solver1.working_end - solver2.working_end)
    # print(solver1.instance_end_time - solver2.instance_end_time)
    # print(abs(np.sum(solver1.service_time - solver2.service_time)) < 1e-6)
    # print(abs(np.sum(solver1.time_windows - solver2.time_windows)) < 1e-6)
    # breakpoint()
    # print(solver1.time == solver2.time)

    for key, value in solver1.nxt.items():
        if key not in solver2.nxt:
            print(f"Key {key} not in solver2.nxt")
        elif not np.array_equal(value, solver2.nxt[key]):
            breakpoint()
            print(f"Value mismatch for key {key}")
            
    for key, value in solver2.nxt.items():
        if key not in solver1.nxt:
            print(f"Key {key} not in solver1.nxt")
        elif not np.array_equal(value, solver1.nxt[key]):
            breakpoint()
            print(f"Value mismatch for key {key}")

    return True 
    

summary_records = []
route_records = []
failed_files = []

breakpoint()
instance_txt = load_instance("/data/Maojie/Github2/EVRP-TW-D-B/dataset/unanchored/Cus_50/solomon/test.txt")

# [34, 16, 6, 19, 1, 44, 15, 33, 5, 23, 0, 17, 48, 40, 12, 32, 18, 0, 21, 4, 37, 35, 10, 49, 9, 30, 24, 42, 2, 43, 0, 3, 11, 28, 38, 25, 36, 7, 0, 14, 47, 31, 0, 39, 27, 50, 29, 46, 26, 20, 13, 56, 0, 8, 0, 22, 45, 41, 0]

# tensor 版本
# instance_id = "17"
# eval_data = pickle.load(open("/data/Maojie/Github2/EVRP-TW-D-B/dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl", "rb"))
# for i in range(len(eval_data)):
#     instance = eval_data[i]
#     if instance["id"] == int(instance_id):
#         break
breakpoint()
solver = GreedySolver(instance_txt)
solution = solver.solve()   # solution: list of routes
visited_all = bool(all(solver.visited))

fleet_size = len(solution)
objective_value = float(getattr(solver, "global_value", float("nan")))

# 2) Routes
for r_idx, route in enumerate(solution):
    print(route)
breakpoint()
solver2 = GreedySolver(instance, format="tensor")
# compare_solver(solver, solver2)

print("\nTesting tensor format...")
solution2 = solver2.solve()   # solution: list of routes
visited_all2 = bool(all(solver2.visited))
if not visited_all2:
    print(f"Failed to visit all customers in tensor format for instance {instance_id}")

breakpoint()
# 2) Routes
for r_idx, route in enumerate(solution2):
    print(route)

