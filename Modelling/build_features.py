from pathlib import Path

import numpy as np
import pandas as pd

project_dir = Path(r"D:\UIUC\hydraulic_condition_monitoring")
data_dir = project_dir / "Data"
output_file = project_dir / "Modelling" / "cycle_features.csv"

sensor_files = [
    "PS1.txt",
    "PS2.txt",
    "PS3.txt",
    "PS4.txt",
    "PS5.txt",
    "PS6.txt",
    "EPS1.txt",
    "TS1.txt",
    "TS2.txt",
    "TS3.txt",
    "TS4.txt",
]

profile = pd.read_csv(data_dir / "profile.txt", sep=r"\s+", header=None)
profile.columns = [
    "cooler_condition_pct",
    "valve_condition_pct",
    "pump_leakage_level",
    "accumulator_pressure_bar",
    "stable_flag",
]

features = pd.DataFrame(index=range(len(profile)))

for filename in sensor_files:
    print(f"Reading {filename}...")
    values = np.loadtxt(data_dir / filename)
    sensor = Path(filename).stem

    if values.shape[0] != len(profile):
        raise ValueError(
            f"{filename} has {values.shape[0]} cycles; expected {len(profile)}"
        )

    features[f"{sensor}_mean"] = values.mean(axis=1)
    features[f"{sensor}_std"] = values.std(axis=1)
    features[f"{sensor}_min"] = values.min(axis=1)
    features[f"{sensor}_max"] = values.max(axis=1)

    del values

stable = profile["stable_flag"] == 0
features = features.loc[stable].copy()
features["pump_leakage_level"] = profile.loc[stable, "pump_leakage_level"].to_numpy()

features.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")
print("Feature table shape:", features.shape)
print(features.head())
