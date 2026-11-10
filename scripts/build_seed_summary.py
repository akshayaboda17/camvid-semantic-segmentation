
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "results" / "experiments"
OUTPUT = ROOT / "results" / "three_seed_summary.csv"

MODELS = {
    "U-Net++": "unetplusplus",
    "PSPNet": "pspnet",
}

METRICS = [
    "mean_iou",
    "mean_dice",
    "mean_precision",
    "mean_recall",
    "pixel_accuracy",
    "mean_boundary_f1",
]

rows = []

for display_name, prefix in MODELS.items():
    seed_results = []

    for seed in [7, 21, 42]:
        folder = EXPERIMENTS / f"{prefix}_composite_seed{seed}"
        metrics_file = folder / "best_validation_metrics.json"

        if not metrics_file.exists():
            raise FileNotFoundError(
                f"Missing result file: {metrics_file}"
            )

        with metrics_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        seed_results.append({
            "seed": seed,
            **{metric: data[metric] for metric in METRICS},
        })

    for metric in METRICS:
        values = [result[metric] for result in seed_results]

        rows.append({
            "model": display_name,
            "metric": metric,
            "seed_7": values[0],
            "seed_21": values[1],
            "seed_42": values[2],
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "mean_percent": statistics.mean(values) * 100,
            "sample_std_percentage_points":
                statistics.stdev(values) * 100,
        })

columns = [
    "model",
    "metric",
    "seed_7",
    "seed_21",
    "seed_42",
    "mean",
    "sample_std",
    "mean_percent",
    "sample_std_percentage_points",
]

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"Created: {OUTPUT}")
print(f"Total statistical records: {len(rows)}")

print("\nThree-seed mIoU summary:")
for row in rows:
    if row["metric"] == "mean_iou":
        print(
            f'{row["model"]}: '
            f'{row["mean_percent"]:.2f}% '
            f'+/- {row["sample_std_percentage_points"]:.2f} '
            f'percentage points'
        )
