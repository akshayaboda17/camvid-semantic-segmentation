
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "results" / "experiments"
OUTPUT = ROOT / "results" / "experiment_summary.csv"

METRIC_KEYS = [
    "mean_iou",
    "mean_dice",
    "mean_precision",
    "mean_recall",
    "pixel_accuracy",
    "mean_boundary_f1",
]


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_model(folder):
    if folder.startswith("unetplusplus"):
        return "U-Net++"
    if folder.startswith("pspnet"):
        return "PSPNet"
    if folder.startswith("random_forest"):
        return "Random Forest"
    return "Unknown"


def get_seed(folder):
    for part in folder.split("_"):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return ""


def get_experiment_type(folder):
    if folder == "random_forest_test":
        return "Final test"

    if folder.endswith("_test"):
        return "Final test"

    if "agent_evidence" in folder:
        return "Agent search evidence"

    if "class_aware" in folder:
        return "Crop ablation"

    if "composite" in folder:
        return "Composite loss"

    if any(loss in folder for loss in
           ["_ce_", "_dice_", "_focal_", "_lovasz_", "_boundary_"]):
        return "Individual loss"

    return "Other"


def get_loss(folder):
    if "composite" in folder:
        return "CE + Dice + Lovasz"
    if "class_aware" in folder:
        return "CE + Dice + Lovasz"
    for loss in ["boundary", "lovasz", "focal", "dice", "ce"]:
        if f"_{loss}_" in folder:
            return loss.upper()
    if "random_forest" in folder:
        return "Not applicable"
    return ""


def extract_metrics(data):
    # Random Forest stores metrics inside "test_metrics".
    # Deep-learning results store them at the top level.
    if "test_metrics" in data:
        data = data["test_metrics"]

    return {
        key: data.get(key, "")
        for key in METRIC_KEYS
    }


rows = []

for folder in sorted(EXPERIMENTS.iterdir()):
    if not folder.is_dir():
        continue

    name = folder.name

    # Agent evidence contains trial records, not final evaluation metrics.
    if "agent_evidence" in name:
        continue

    # Choose the appropriate evaluation file for each experiment.
    if name == "random_forest_test":
        metric_file = folder / "test_summary.json"
        evaluation = "Test"
    elif name.endswith("_test"):
        metric_file = folder / "standard_metrics.json"
        evaluation = "Test"
    else:
        metric_file = folder / "best_validation_metrics.json"
        evaluation = "Validation"

    if not metric_file.exists():
        continue

    data = read_json(metric_file)

    row = {
        "experiment_folder": name,
        "model": get_model(name),
        "experiment_type": get_experiment_type(name),
        "loss": get_loss(name),
        "seed": get_seed(name),
        "evaluation_split": evaluation,
        **extract_metrics(data),
    }

    rows.append(row)


columns = [
    "experiment_folder",
    "model",
    "experiment_type",
    "loss",
    "seed",
    "evaluation_split",
    *METRIC_KEYS,
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCreated: {OUTPUT}")
print(f"Total experiment rows: {len(rows)}")

print("\nRows by experiment type:")
for category in sorted(set(row["experiment_type"] for row in rows)):
    count = sum(row["experiment_type"] == category for row in rows)
    print(f"{category}: {count}")

print("\nRows by evaluation split:")
for split in ["Validation", "Test"]:
    count = sum(row["evaluation_split"] == split for row in rows)
    print(f"{split}: {count}")
