import pandas as pd

csv = pd.read_csv("/data/Maojie/Github2/EVRP-TW-D-B/evrptw_gen/benchmarks/ALNS_Solver/logs_Cus50_resume100/resume_summary.csv")
obj = csv["objective_value"]
breakpoint()
print(obj.mean())
