import pandas as pd
import re

file1 = "rl_results.csv"
file2 = "/data/Maojie/Github2/EVRP-TW-D-B/evrptw_gen/benchmarks/ALNS_Solver_MULTI/logs_Cus50_3000_32mul_1k_instance/alns_summary.csv"

output_file = "file2_wins_instances.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)


def extract_idx(s):
    s = str(s)
    m = re.search(r"solomon_dataset_(\d+)", s, re.IGNORECASE)
    return int(m.group(1)) if m else None


# Extract matching id from file name
df1["match_id"] = df1["file"].apply(extract_idx)
df2["match_id"] = df2["file"].apply(extract_idx)

# Remove rows that cannot be matched
df1 = df1.dropna(subset=["match_id"]).copy()
df2 = df2.dropna(subset=["match_id"]).copy()

df1["match_id"] = df1["match_id"].astype(int)
df2["match_id"] = df2["match_id"].astype(int)

# Rename columns before merge to avoid file_x / file_y ambiguity
df1 = df1.rename(columns={
    "file": "file_1",
    "instance_id": "instance_id_1",
    "objective_value": "objective_value_1"
})

df2 = df2.rename(columns={
    "file": "file_2",
    "instance_id": "instance_id_2",
    "objective_value": "objective_value_2"
})

# Merge two result files by match_id
merged = pd.merge(df1, df2, on="match_id", how="inner")

# Scale objective_value_1 if needed
# 当前没有 scale，所以直接复制
merged["objective_value_1_scaled"] = merged["objective_value_1"]

# Difference: positive means file2 is better because objective is smaller
merged["diff"] = merged["objective_value_1_scaled"] - merged["objective_value_2"]
merged["abs_diff"] = merged["diff"].abs()

# Winner: smaller objective value wins
merged["winner"] = merged.apply(
    lambda row: "file1"
    if row["objective_value_1_scaled"] < row["objective_value_2"]
    else (
        "file2"
        if row["objective_value_2"] < row["objective_value_1_scaled"]
        else "tie"
    ),
    axis=1
)

# Count wins
win_counts = merged["winner"].value_counts()

file1_wins = win_counts.get("file1", 0)
file2_wins = win_counts.get("file2", 0)
ties = win_counts.get("tie", 0)

print(f"Matched instances: {len(merged)}")
print(f"file1 wins: {file1_wins}")
print(f"file2 wins: {file2_wins}")
print(f"ties: {ties}\n")

print("Total Abs Diff Stats:")
print(f"Mean Abs Diff: {merged['abs_diff'].mean()}")
print(f"Std Abs Diff: {merged['abs_diff'].std()}\n")

print("Signed Diff Stats:")
print(f"Mean Diff: {merged['diff'].mean()}")
print(f"Std Diff: {merged['diff'].std()}\n")

# Save file2 winning instances
file2_win_df = merged[merged["winner"] == "file2"].copy()

file2_win_df = file2_win_df[[
    "match_id",
    "file_2",
    "objective_value_1",
    "objective_value_2",
    "diff",
    "abs_diff"
]]

file2_win_df.to_csv(output_file, index=False)

print(f"Saved file2 wins to: {output_file}")
print(f"Saved count: {len(file2_win_df)}")