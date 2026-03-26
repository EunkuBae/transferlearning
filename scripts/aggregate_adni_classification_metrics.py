from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def main() -> None:
    args = parse_args()
    payloads = []
    for metrics_path in args.metrics:
        with metrics_path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))

    tracked_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]
    summary = {
        "num_runs": len(payloads),
        "runs": [str(path) for path in args.metrics],
        "test_metrics": {},
    }
    for key in tracked_metrics:
        values = [float(payload["test_metrics"][key]) for payload in payloads]
        summary["test_metrics"][key] = {
            "mean": mean(values),
            "std": std(values),
            "values": values,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
