from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

project_dir = Path(r"D:\UIUC\hydraulic_condition_monitoring")
data_dir = project_dir / "Data"
modelling_dir = project_dir / "Modelling"

features = pd.read_csv(modelling_dir / "cycle_features.csv")

profile = pd.read_csv(data_dir / "profile.txt", sep=r"\s+", header=None)
profile.columns = [
    "cooler_condition_pct",
    "valve_condition_pct",
    "pump_leakage_level",
    "accumulator_pressure_bar",
    "stable_flag",
]

stable_profile = profile.loc[profile["stable_flag"] == 0].reset_index(drop=True)

if len(features) != len(stable_profile):
    raise ValueError("Feature rows and stable profile rows do not match.")

X = features.drop(columns=["pump_leakage_level"])
y = features["pump_leakage_level"].astype(int)

condition_columns = [
    "cooler_condition_pct",
    "valve_condition_pct",
    "pump_leakage_level",
    "accumulator_pressure_bar",
]
groups = stable_profile[condition_columns].astype(str).agg("|".join, axis=1)

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

print("Training rows:", len(train_idx))
print("Test rows:", len(test_idx))
print("Training labels:")
print(y.iloc[train_idx].value_counts().sort_index())
print("\nTest labels:")
print(y.iloc[test_idx].value_counts().sort_index())

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
print("\nCondition groups appearing in both sets:", len(train_groups & test_groups))
