from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_dir = Path(r"D:\UIUC\hydraulic_condition_monitoring")
modelling_dir = project_dir / "Modelling"

features = pd.read_csv(modelling_dir / "cycle_features.csv")

plot_features = ["EPS1_mean", "PS3_mean", "PS1_max"]
classes = [0, 1, 2]
class_names = ["No leakage", "Weak leakage", "Severe leakage"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, feature in zip(axes, plot_features):
    values_by_class = [
        features.loc[features["pump_leakage_level"] == class_id, feature].dropna()
        for class_id in classes
    ]

    ax.boxplot(values_by_class)
    ax.set_xticks([1, 2, 3], class_names)
    ax.set_title(feature)
    ax.set_ylabel("Feature value")
    ax.tick_params(axis="x", rotation=20)

fig.suptitle("Sensor summaries by pump-leakage level")
fig.tight_layout()

output_file = modelling_dir / "sensor_patterns_by_leakage.png"
fig.savefig(output_file, dpi=150)
plt.show()

print(f"Saved chart: {output_file}")
