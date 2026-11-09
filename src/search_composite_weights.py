"""Agent-based search for non-negative CE/Dice/Lovasz composite weights.

The agent evaluates a fixed, reproducible set of weight configurations using
only the CamVid training and validation splits accessed by src.train. It never
loads the test split.

Example:
    python -m src.search_composite_weights --model unetplusplus
    python -m src.search_composite_weights --model pspnet
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("cross_entropy", "dice", "lovasz")
DEFAULT_SEED = 42
DEFAULT_TRIALS = 5
DEFAULT_SEARCH_EPOCHS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search composite CE/Dice/Lovasz loss weights using validation mIoU."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("unetplusplus", "pspnet"),
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--search-epochs", type=int, default=DEFAULT_SEARCH_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    if args.trials < 2:
        parser.error("--trials must be at least 2 (equal-weight baseline + search trial).")
    if args.search_epochs <= 0:
        parser.error("--search-epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    return args


def normalize_and_round(values: np.ndarray) -> dict[str, float]:
    values = np.maximum(values.astype(np.float64), 0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("A candidate weight vector cannot sum to zero.")
    values = values / total
    rounded = np.round(values, 6)
    # Make the sum exactly 1 after rounding, without changing the first two.
    rounded[-1] = round(1.0 - float(rounded[:-1].sum()), 6)
    return {name: float(value) for name, value in zip(COMPONENTS, rounded)}


def make_candidates(number: int, seed: int) -> list[dict[str, float]]:
    """One equal-weight reference plus reproducible Dirichlet random trials."""
    candidates = [normalize_and_round(np.array([1.0, 1.0, 1.0]))]
    rng = np.random.default_rng(seed)
    seen = {tuple(candidates[0][name] for name in COMPONENTS)}

    # Positive weights on the simplex; Dirichlet(1.5,1.5,1.5) avoids
    # concentrating every draw too close to a simplex corner.
    while len(candidates) < number:
        proposal = normalize_and_round(rng.dirichlet(np.array([1.5, 1.5, 1.5])))
        key = tuple(proposal[name] for name in COMPONENTS)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(proposal)

    return candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "trial",
        "weights_json",
        "best_validation_miou",
        "best_validation_miou_percent",
        "search_epochs",
        "seed",
        "status",
        "trial_output_dir",
        "training_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_trial(
    *,
    model: str,
    weights: dict[str, float],
    trial_number: int,
    output_dir: Path,
    search_epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    dataset_root: str | None,
) -> tuple[float, Path]:
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--model",
        model,
        "--loss",
        "composite",
        "--loss-weights",
        json.dumps(weights, separators=(",", ":")),
        "--epochs",
        str(search_epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if dataset_root:
        command.extend(["--dataset-root", dataset_root])

    log_path = output_dir / "training.log"
    output_dir.mkdir(parents=True, exist_ok=False)

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if process.returncode != 0:
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-50:])
        except OSError:
            tail = "Could not read trial log."
        raise RuntimeError(
            f"Agent trial {trial_number} failed with exit code "
            f"{process.returncode}. Last log lines:\n{tail}"
        )

    summary_path = output_dir / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Trial {trial_number} did not produce {summary_path}."
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    score = float(summary["best_validation_miou"])
    if not np.isfinite(score):
        raise RuntimeError(f"Trial {trial_number} produced a non-finite validation mIoU.")
    return score, log_path


def copy_selected_trial_to_final(
    selected_trial_dir: Path,
    final_dir: Path,
    agent_dir: Path,
) -> None:
    """Prepare a clean 10-epoch run initialized from selected trial epoch N."""
    if final_dir.exists() and any(final_dir.iterdir()):
        raise FileExistsError(
            f"Final composite directory already contains files: {final_dir}. "
            "Move it aside or choose a different output-root before rerunning the agent."
        )

    checkpoint_source = selected_trial_dir / "checkpoints"
    checkpoint_target = final_dir / "checkpoints"
    checkpoint_target.mkdir(parents=True, exist_ok=True)

    for filename in ("last_checkpoint.pth", "best_model.pth"):
        source = checkpoint_source / filename
        if not source.is_file():
            raise FileNotFoundError(f"Selected trial is missing {source}.")
        shutil.copy2(source, checkpoint_target / filename)

    copy_names = (
        "history.csv",
        "best_validation_metrics.json",
        "best_class_metrics.csv",
        "best_confusion_matrix.csv",
    )
    for filename in copy_names:
        source = selected_trial_dir / filename
        if source.is_file():
            shutil.copy2(source, final_dir / filename)

    # Keep the complete agent evidence beside the final composite run.
    for filename in ("agent_trials.csv", "selected_weights.json", "agent_summary.json"):
        source = agent_dir / filename
        if source.is_file():
            shutil.copy2(source, final_dir / filename)


def main() -> None:
    args = parse_args()
    default_root = (
        PROJECT_ROOT
        / "results"
        / "agent_search"
        / f"{args.model}_composite_seed{args.seed}"
    )
    agent_dir = Path(args.output_root) if args.output_root else default_root
    agent_dir = agent_dir.resolve()

    if agent_dir.exists() and any(agent_dir.iterdir()):
        raise FileExistsError(
            f"Agent output directory already exists and is not empty: {agent_dir}. "
            "Choose a new --output-root to avoid overwriting previous search logs."
        )
    agent_dir.mkdir(parents=True, exist_ok=True)

    final_dir = (
        PROJECT_ROOT
        / "results"
        / "training"
        / f"{args.model}_composite_seed{args.seed}"
    )
    if final_dir.exists() and any(final_dir.iterdir()):
        raise FileExistsError(
            f"Final composite output directory already contains files: {final_dir}. "
            "Move it aside before running a new search."
        )

    candidates = make_candidates(args.trials, args.seed)
    rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []

    print("=" * 72)
    print("COMPOSITE LOSS WEIGHT SEARCH AGENT")
    print("=" * 72)
    print("Model:", args.model)
    print("Objective: maximize validation mIoU")
    print("Components:", ", ".join(COMPONENTS))
    print("Search strategy: equal-weight reference + seeded Dirichlet random search")
    print(f"Compute budget: {args.trials} trials x {args.search_epochs} epochs = "
          f"{args.trials * args.search_epochs} model-epochs")
    print("Stopping rule: stop when the fixed trial budget is exhausted.")
    print("Dataset access: training and validation only; test split is not loaded.")
    print("Agent output:", agent_dir)

    for index, weights in enumerate(candidates, start=1):
        trial_name = f"trial_{index:02d}"
        trial_dir = agent_dir / trial_name
        print("\n" + "-" * 72)
        print(f"TRIAL {index}/{args.trials}")
        print("Weights:", weights)
        print(f"Training for {args.search_epochs} epochs...")

        try:
            score, log_path = run_trial(
                model=args.model,
                weights=weights,
                trial_number=index,
                output_dir=trial_dir,
                search_epochs=args.search_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                dataset_root=args.dataset_root,
            )
        except Exception as exc:
            failed_log = trial_dir / "training.log"
            rows.append({
                "trial": index,
                "weights_json": json.dumps(weights, sort_keys=True),
                "best_validation_miou": "",
                "best_validation_miou_percent": "",
                "search_epochs": args.search_epochs,
                "seed": args.seed,
                "status": f"failed: {exc}",
                "trial_output_dir": str(trial_dir),
                "training_log": str(failed_log),
            })
            write_csv(agent_dir / "agent_trials.csv", rows)
            raise

        row = {
            "trial": index,
            "weights_json": json.dumps(weights, sort_keys=True),
            "best_validation_miou": score,
            "best_validation_miou_percent": round(score * 100.0, 4),
            "search_epochs": args.search_epochs,
            "seed": args.seed,
            "status": "completed",
            "trial_output_dir": str(trial_dir),
            "training_log": str(log_path),
        }
        rows.append(row)
        completed.append({"trial": index, "weights": weights, "score": score, "directory": trial_dir})
        write_csv(agent_dir / "agent_trials.csv", rows)
        print(f"Trial {index} best validation mIoU: {score * 100:.2f}%")
        print("Trial log:", log_path)

    # The selected candidate is the highest validation mIoU among the fixed trials.
    selected = max(completed, key=lambda item: item["score"])
    selected_weights = selected["weights"]

    selected_weights_path = agent_dir / "selected_weights.json"
    with selected_weights_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "loss": "composite",
                "components": list(COMPONENTS),
                "weights": selected_weights,
                "selection_metric": "validation_mean_iou",
                "selected_trial": selected["trial"],
                "selected_validation_miou": selected["score"],
            },
            handle,
            indent=2,
        )

    search_summary = {
        "agent_name": "CompositeLossWeightSearchAgent",
        "model": args.model,
        "objective": "maximize validation mean IoU (mIoU)",
        "search_space": {
            "components": list(COMPONENTS),
            "constraints": "each lambda >= 0 and lambda_CE + lambda_Dice + lambda_Lovasz = 1",
            "sampling_distribution": "Dirichlet(alpha=[1.5, 1.5, 1.5])",
            "reference_trial": "equal weights (1/3, 1/3, 1/3)",
        },
        "strategy": "fixed-seed random search over the weight simplex",
        "compute_budget": {
            "trials": args.trials,
            "epochs_per_trial": args.search_epochs,
            "total_search_model_epochs": args.trials * args.search_epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "stopping_rule": "stop after the predeclared number of trials has completed",
        "data_protocol": {
            "training_split_used": True,
            "validation_split_used_for_selection": True,
            "test_split_accessed": False,
        },
        "selected_trial": selected["trial"],
        "selected_weights": selected_weights,
        "selected_validation_miou": selected["score"],
        "selected_validation_miou_percent": selected["score"] * 100.0,
        "trial_log": "agent_trials.csv",
    }
    summary_path = agent_dir / "agent_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(search_summary, handle, indent=2)

    copy_selected_trial_to_final(
        selected_trial_dir=selected["directory"],
        final_dir=final_dir,
        agent_dir=agent_dir,
    )

    print("\n" + "=" * 72)
    print("AGENT SEARCH COMPLETED")
    print("=" * 72)
    print("Selected trial:", selected["trial"])
    print("Selected weights:", selected_weights)
    print(f"Selected validation mIoU: {selected['score'] * 100:.2f}%")
    print("Trial log:", agent_dir / "agent_trials.csv")
    print("Agent summary:", summary_path)
    print("Prepared final training directory:", final_dir)
    print("\nNext, continue the selected checkpoint to the full 10-epoch budget:")
    print(
        f'python -m src.train --model {args.model} --loss composite '
        f'--loss-weights \'{json.dumps(selected_weights, separators=(",", ":"))}\' '
        f'--epochs 10 --batch-size {args.batch_size} --num-workers {args.num_workers} '
        f'--seed {args.seed} --output-dir "{final_dir.relative_to(PROJECT_ROOT)}" '
        f'--resume "{(final_dir / "checkpoints" / "last_checkpoint.pth").relative_to(PROJECT_ROOT)}"'
    )
    print("Test split was not accessed.")


if __name__ == "__main__":
    main()
