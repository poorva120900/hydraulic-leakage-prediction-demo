from pathlib import Path
import pandas as pd

data_dir = Path(r"D:\UIUC\hydraulic_condition_monitoring\Data")

profile = pd.read_csv(data_dir / "profile.txt", sep=r"\s+", header=None)

profile.columns = [
    "cooler_condition_pct",
    "valve_condition_pct",
    "pump_leakage_level",
    "accumulator_pressure_bar",
    "stable_flag",
]

stable_profile = profile[profile["stable_flag"] == 0]

print("\nSettled cycles:", len(stable_profile))
print("\nPump leakage labels in settled cycles:")
print(stable_profile["pump_leakage_level"].value_counts().sort_index())
print("\nProfile labels:")
print(profile.head())

for column in profile.columns:
    print(f"\n{column} counts:")
    print(profile[column].value_counts().sort_index())

print("profile shape:", profile.shape)
print(profile.head())

for name in ["PS1.txt", "EPS1.txt", "TS1.txt"]:
    sample = pd.read_csv(data_dir / name, sep=r"\s+", header=None, nrows=2)
    print(f"{name}: first two rows have shape {sample.shape}")
