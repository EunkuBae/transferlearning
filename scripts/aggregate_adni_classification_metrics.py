from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TRACKED_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]
PER_CLASS_METRICS = [
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ADNI classification metrics across seed runs.")
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        help="Metrics JSON files from ADNI classification runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/adni_classification_seed_summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional flat CSV summary path for paper tables.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def summarize_values(values: list[float]) -> dict[str, object]:
    return {
        "mean": mean(values),
        "std": std(values),
        "values": values,
    }


def build_csv_rows(summary: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric_group, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_summary in metrics.items():
            if isinstance(metric_summary, dict) and "mean" in metric_summary:
                rows.append(
                    {
                        "group": metric_group,
                        "metric": metric_name,
                        "mean": metric_summary["mean"],
                        "std": metric_summary["std"],
                        "values": json.dumps(metric_summary["values"]),
                    }
                )
            elif isinstance(metric_summary, dict):
                for class_name, class_summary in metric_summary.items():
                    rows.append(
                        {
                            "group": metric_group,
                            "metric": f"{metric_name}.{class_name}",
                            "mean": class_summary["mean"],
                            "std": class_summary["std"],
                            "values": json.dumps(class_summary["values"]),
                        }
                    )
    return rows


def main() -> None:
    args = parse_args()
    payloads = []
    for metrics_path in args.metrics:
        with metrics_path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))

    summary = {
        "num_runs": len(payloads),
        "runs": [str(path) for path in args.metrics],
        "experiments": [str(payload.get("experiment_name", "unknown")) for payload in payloads],
        "test_metrics": {},
        "per_class_test_metrics": {},
        "confusion_matrices": [payload.get("test_metrics", {}).get("confusion_matrix") for payload in payloads],
    }
    for key in TRACKED_METRICS:
        values = [float(payload["test_metrics"][key]) for payload in payloads]
        summary["test_metrics"][key] = summarize_values(values)

    class_names: set[str] = set()
    for payload in payloads:
        for metric_name in PER_CLASS_METRICS:
            class_names.update(payload.get("test_metrics", {}).get(metric_name, {}).keys())

    for metric_name in PER_CLASS_METRICS:
        metric_summary: dict[str, dict[str, object]] = {}
        for class_name in sorted(class_names, key=lambda value: int(value)):
            values = [float(payload.get("test_metrics", {}).get(metric_name, {}).get(class_name, 0.0)) for payload in payloads]
            metric_summary[class_name] = summarize_values(values)
        summary["per_class_test_metrics"][metric_name] = metric_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_csv_rows(
        {
            "test_metrics": summary["test_metrics"],
            "per_class_test_metrics": summary["per_class_test_metrics"],
        }
    )
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "metric", "mean", "std", "values"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
