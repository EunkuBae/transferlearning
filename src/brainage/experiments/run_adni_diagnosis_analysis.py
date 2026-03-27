"""Analyze ADNI MMSE predictions by diagnosis subgroup."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from brainage.data.adni_mmse import normalize_adni_diagnosis
from brainage.utils.metrics import regression_metrics

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("Running ADNI diagnosis analysis requires PyYAML.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ADNI MMSE transfer predictions by diagnosis subgroup.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/adni_diagnosis_analysis.yaml"),
        help="Path to ADNI diagnosis analysis config YAML.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_predictions(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "subject_id": str(row["subject_id"]),
                "target_mmse": float(row["target_mmse"]),
                "predicted_mmse": float(row["predicted_mmse"]),
            }
            for row in reader
        ]


def load_diagnosis_lookup(path: Path, subject_id_column: str, diagnosis_column: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(row.get(subject_id_column, "")).strip()
            diagnosis = normalize_adni_diagnosis(row.get(diagnosis_column))
            if subject_id and diagnosis is not None and subject_id not in lookup:
                lookup[subject_id] = diagnosis
    return lookup


def build_subgroup_metrics(predictions: list[dict[str, float | str]], diagnosis_lookup: dict[str, str]) -> dict[str, dict[str, float | int]]:
    grouped_targets: dict[str, list[float]] = defaultdict(list)
    grouped_predictions: dict[str, list[float]] = defaultdict(list)
    grouped_signed_errors: dict[str, list[float]] = defaultdict(list)

    for row in predictions:
        subject_id = str(row["subject_id"])
        diagnosis = diagnosis_lookup.get(subject_id)
        if diagnosis is None:
            continue
        target = float(row["target_mmse"])
        prediction = float(row["predicted_mmse"])
        grouped_targets[diagnosis].append(target)
        grouped_predictions[diagnosis].append(prediction)
        grouped_signed_errors[diagnosis].append(prediction - target)

    metrics_by_group: dict[str, dict[str, float | int]] = {}
    for diagnosis in sorted(grouped_targets):
        metrics = regression_metrics(grouped_targets[diagnosis], grouped_predictions[diagnosis])
        signed_errors = grouped_signed_errors[diagnosis]
        metrics["num_subjects"] = len(signed_errors)
        metrics["mean_signed_error"] = sum(signed_errors) / len(signed_errors)
        metrics_by_group[diagnosis] = metrics
    return metrics_by_group


def write_csv(path: Path, subgroup_metrics: dict[str, dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["diagnosis", "num_subjects", "mae", "rmse", "pearson_r", "r2", "mean_signed_error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for diagnosis, metrics in subgroup_metrics.items():
            writer.writerow(
                {
                    "diagnosis": diagnosis,
                    "num_subjects": metrics.get("num_subjects"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "pearson_r": metrics.get("pearson_r"),
                    "r2": metrics.get("r2"),
                    "mean_signed_error": metrics.get("mean_signed_error"),
                }
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    input_config = config.get("inputs", {})
    output_config = config.get("outputs", {})

    predictions_path = Path(str(input_config["predictions_file"])).resolve()
    metadata_path = Path(str(input_config["metadata_file"])).resolve()
    output_dir = Path(str(output_config.get("run_dir", "outputs/adni_diagnosis_analysis"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(predictions_path)
    diagnosis_lookup = load_diagnosis_lookup(
        metadata_path,
        subject_id_column=str(input_config.get("subject_id_column", "PTID")),
        diagnosis_column=str(input_config.get("diagnosis_column", "DX_bl")),
    )
    subgroup_metrics = build_subgroup_metrics(predictions, diagnosis_lookup)

    json_path = output_dir / "diagnosis_subgroup_metrics.json"
    csv_path = output_dir / "diagnosis_subgroup_metrics.csv"
    summary_path = output_dir / "analysis_summary.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(subgroup_metrics, handle, indent=2)
    write_csv(csv_path, subgroup_metrics)
    summary_path.write_text(
        "\n".join([f"{diagnosis}: {metrics}" for diagnosis, metrics in subgroup_metrics.items()]) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(output_dir), "subgroup_metrics": subgroup_metrics}, indent=2))


if __name__ == "__main__":
    main()
