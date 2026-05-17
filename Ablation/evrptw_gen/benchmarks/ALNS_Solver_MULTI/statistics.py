import pandas as pd

csv = pd.read_csv("/data/Maojie/Github2/EVRP-TW-D-B/evrptw_gen/benchmarks/ALNS_Solver_MULTI/logs_Cus50_3000_32mul_1k_instance/alns_summary.csv")
obj = csv["objective_value"]
breakpoint()
print(obj.mean())
