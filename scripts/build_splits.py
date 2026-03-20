"""Build frozen LODO split files from a cohort metadata CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from brainage.data.schemas import REQUIRED_METADATA_COLUMNS, VALID_COHORTS, VALID_DIAGNOSIS_LABELS
from brainage.data.split_builders import build_lodo_split_rows, default_split_path, save_split_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LODO split files.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to merged metadata CSV.")
    parser.add_argument("--holdout-cohort", type=str, required=True, help="Cohort name to hold out.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Validation ratio within source cohorts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument("--output", type=Path, default=None, help="Optional custom output CSV path.")
    parser.add_argument(
        "--allow-missing-mmse",
        action="store_true",
        help="Allow rows with empty MMSE values without warning escalation.",
    )
    return parser.parse_args()


def load_metadata_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_metadata_rows(rows: list[dict], allow_missing_mmse: bool = False) -> None:
    if not rows:
        raise ValueError("Merged metadata CSV is empty.")

    columns = set(rows[0].keys())
    missing_columns = [column for column in REQUIRED_METADATA_COLUMNS if column not in columns]
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {missing_columns}")

    seen_subject_ids: set[str] = set()
    valid_cohorts = set(VALID_COHORTS)
    valid_diagnosis = set(VALID_DIAGNOSIS_LABELS)

    for row_index, row in enumerate(rows, start=2):
        subject_id = str(row.get("subject_id", "")).strip()
        cohort = str(row.get("cohort", "")).strip().lower()
        image_path = str(row.get("image_path", "")).strip()
        diagnosis = str(row.get("diagnosis", "")).strip().upper()
        mmse = str(row.get("mmse", "")).strip()

        if not subject_id:
            raise ValueError(f"Row {row_index}: subject_id is empty.")
        if subject_id in seen_subject_ids:
            raise ValueError(f"Row {row_index}: duplicate subject_id '{subject_id}'.")
        seen_subject_ids.add(subject_id)

        if cohort not in valid_cohorts:
            raise ValueError(f"Row {row_index}: invalid cohort '{cohort}'. Expected one of {sorted(valid_cohorts)}.")
        if not image_path:
            raise ValueError(f"Row {row_index}: image_path is empty.")
        if diagnosis not in valid_diagnosis:
            raise ValueError(
                f"Row {row_index}: invalid diagnosis '{diagnosis}'. Expected one of {sorted(valid_diagnosis)}."
            )
        if mmse:
            try:
                float(mmse)
            except ValueError as exc:
                raise ValueError(f"Row {row_index}: mmse must be numeric when provided, got '{mmse}'.") from exc
        elif not allow_missing_mmse:
            print(f"Warning: Row {row_index} has empty mmse for subject_id={subject_id}")


def main() -> None:
    args = parse_args()
    holdout_cohort = args.holdout_cohort.lower()
    if holdout_cohort not in VALID_COHORTS:
        raise ValueError(f"Invalid holdout cohort '{args.holdout_cohort}'. Expected one of {VALID_COHORTS}.")

    rows = load_metadata_rows(args.metadata)
    validate_metadata_rows(rows, allow_missing_mmse=args.allow_missing_mmse)
    split_rows = build_lodo_split_rows(
        records=rows,
        holdout_cohort=holdout_cohort,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    output_path = args.output or default_split_path(holdout_cohort)
    save_split_rows(split_rows, output_path)
    print(f"Saved {len(split_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
