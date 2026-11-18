from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold


project_dir = Path(r"D:\UIUC\hydraulic_condition_monitoring")
data_dir = project_dir / "Data"
modelling_dir = project_dir / "Modelling"

# Load the cycle-level sensor features.
features = pd.read_csv(modelling_dir / "cycle_features.csv")

# Load and name the cycle condition labels.
profile = pd.read_csv(data_dir / "profile.txt", sep=r"\s+", header=None)
profile.columns = [
    "cooler_condition_pct",
    "valve_condition_pct",
    "pump_leakage_level",
    "accumulator_pressure_bar",
    "stable_flag",
]

# Keep only cycles marked stable.
stable_profile = profile.loc[profile["stable_flag"] == 0].reset_index(drop=True)

if len(features) != len(stable_profile):
    raise ValueError("Feature rows and stable profile rows do not match.")

# Sensor summaries are the model inputs.
X = features.drop(columns=["pump_leakage_level"])

# Pump leakage level is the value the model predicts.
y = features["pump_leakage_level"].astype(int)

# Group cycles with the same component-condition profile.
# These labels are used only for the split, not as model inputs.
condition_columns = [
    "cooler_condition_pct",
    "valve_condition_pct",
    "pump_leakage_level",
    "accumulator_pressure_bar",
]
groups = stable_profile[condition_columns].astype(str).agg("|".join, axis=1)

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

balanced_accuracies = []
macro_f1_scores = []
importances = []
prediction_examples = None

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups),
    start=1,
):
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = len(train_groups & test_groups)

    if overlap != 0:
        raise ValueError(f"Fold {fold} has {overlap} condition groups in both sets.")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])

    balanced_acc = balanced_accuracy_score(
        y.iloc[test_idx],
        predictions,
    )
    macro_f1 = f1_score(
        y.iloc[test_idx],
        predictions,
        average="macro",
    )

    balanced_accuracies.append(balanced_acc)
    macro_f1_scores.append(macro_f1)
    importances.append(model.feature_importances_)

    print(
        f"Fold {fold}: "
        f"balanced accuracy={balanced_acc:.3f}, "
        f"macro F1={macro_f1:.3f}, "
        f"group overlap={overlap}"
    )

    # Keep individual test predictions from the first fold.
    if fold == 1:
        probabilities = model.predict_proba(X.iloc[test_idx])
        class_positions = {
            int(label): position for position, label in enumerate(model.classes_)
        }

        input_summaries = X.iloc[test_idx][
            ["EPS1_mean", "PS3_mean", "PS1_max"]
        ].reset_index(drop=True)

        fold_predictions = pd.DataFrame(
            {
                "actual": y.iloc[test_idx].to_numpy(),
                "predicted": predictions,
                "confidence": probabilities.max(axis=1),
                "P_no_leakage": probabilities[:, class_positions[0]],
                "P_weak_leakage": probabilities[:, class_positions[1]],
                "P_severe_leakage": probabilities[:, class_positions[2]],
            }
        )

        fold_predictions = pd.concat(
            [input_summaries, fold_predictions],
            axis=1,
        )

        prediction_examples = fold_predictions.sample(
            n=5,
            random_state=42,
        )

        prediction_examples = fold_predictions.sample(
            n=5,
            random_state=42,
        )

print(
    "\nMean balanced accuracy:",
    round(mean(balanced_accuracies), 3),
    "+/-",
    round(pstdev(balanced_accuracies), 3),
)
print(
    "Mean macro F1:",
    round(mean(macro_f1_scores), 3),
    "+/-",
    round(pstdev(macro_f1_scores), 3),
)

print("\nFive individual predictions from held-out Fold 1:")
print(prediction_examples.round(3).to_string(index=False))

# Average feature importance across the five fold models.
importance_by_fold = pd.DataFrame(importances, columns=X.columns)
mean_importance = importance_by_fold.mean().sort_values(ascending=False)

print("\nTop 10 sensor features by average Random Forest importance:")
print(mean_importance.head(10))
