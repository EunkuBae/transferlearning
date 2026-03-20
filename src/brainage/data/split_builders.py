"""Utilities for generating frozen LODO split files."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from brainage.paths import SPLITS_DIR


def build_lodo_split_rows(
    records: Iterable[dict],
    holdout_cohort: str,
    validation_ratio: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        grouped[str(record["cohort"]).lower()].append(record)

    rows: list[dict] = []
    for cohort, cohort_records in grouped.items():
        cohort_records = list(cohort_records)
        if cohort == holdout_cohort.lower():
            for record in cohort_records:
                rows.append(
                    {
                        "subject_id": record["subject_id"],
                        "cohort": cohort,
                        "split": "test",
                        "fold_name": f"lodo_{holdout_cohort.lower()}_holdout",
                    }
                )
            continue

        rng.shuffle(cohort_records)
        val_count = max(1, int(len(cohort_records) * validation_ratio)) if cohort_records else 0
        for index, record in enumerate(cohort_records):
            split = "val" if index < val_count else "train"
            rows.append(
                {
                    "subject_id": record["subject_id"],
                    "cohort": cohort,
                    "split": split,
                    "fold_name": f"lodo_{holdout_cohort.lower()}_holdout",
                }
            )

    return rows


def save_split_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "cohort", "split", "fold_name"])
        writer.writeheader()
        writer.writerows(rows)


def default_split_path(holdout_cohort: str) -> Path:
    return SPLITS_DIR / f"lodo_{holdout_cohort.lower()}_holdout.csv"
