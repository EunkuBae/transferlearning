"""Utilities for LODO MMSE regression built from merged cohort metadata."""

from __future__ import annotations

import csv
from pathlib import Path

from brainage.data.hcp_mmse import HCPMMSEExample


DATA_ROOT_TOKEN = "${BRAINAGE_DATA_ROOT}"


def resolve_metadata_image_path(raw_path: str, data_root: Path) -> Path:
    value = str(raw_path or "").strip()
    if not value:
        raise ValueError("image_path is empty in merged metadata")
    if value.startswith(DATA_ROOT_TOKEN):
        relative = value[len(DATA_ROOT_TOKEN) :].lstrip("/\\")
        return (data_root / Path(relative)).resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (data_root / path).resolve()


def load_merged_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_lodo_mmse_examples(
    metadata_path: Path,
    data_root: Path,
    cohort_filter: set[str] | None = None,
) -> list[HCPMMSEExample]:
    rows = load_merged_metadata(metadata_path)
    examples: list[HCPMMSEExample] = []
    seen_subject_ids: set[str] = set()

    for row_index, row in enumerate(rows, start=2):
        subject_id = str(row.get("subject_id", "")).strip()
        cohort = str(row.get("cohort", "")).strip().lower()
        mmse_value = str(row.get("mmse", "")).strip()
        if cohort_filter is not None and cohort not in cohort_filter:
            continue
        if not subject_id or not mmse_value or subject_id in seen_subject_ids:
            continue

        try:
            mmse = float(mmse_value)
        except ValueError as exc:
            raise ValueError(f"Row {row_index}: invalid mmse value '{mmse_value}'") from exc

        image_path = resolve_metadata_image_path(row.get("image_path", ""), data_root)
        if not image_path.exists():
            continue

        age_raw = str(row.get("age", "")).strip()
        age = float(age_raw) if age_raw else None
        sex = str(row.get("sex", "")).strip().upper() or None

        examples.append(
            HCPMMSEExample(
                subject_id=subject_id,
                image_path=image_path,
                mmse=mmse,
                age=age,
                sex=sex,
            )
        )
        seen_subject_ids.add(subject_id)

    if not examples:
        raise ValueError("No LODO MMSE examples were matched between merged metadata and image files.")

    return examples


def load_lodo_split_assignments(path: Path, examples: list[HCPMMSEExample]) -> tuple[dict[str, list[HCPMMSEExample]], str]:
    example_by_subject = {example.subject_id: example for example in examples}
    split_sets = {"train": [], "val": [], "test": []}
    seen_subject_ids: set[str] = set()
    fold_name = "unknown"

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(row.get("subject_id", "")).strip()
            split_name = str(row.get("split", "")).strip().lower()
            fold_name = str(row.get("fold_name", fold_name)).strip() or fold_name
            if not subject_id:
                raise ValueError(f"Split file contains an empty subject_id: {path}")
            if split_name not in split_sets:
                raise ValueError(f"Split file contains invalid split '{split_name}' for subject_id={subject_id}")
            if subject_id in seen_subject_ids:
                raise ValueError(f"Split file contains duplicate subject_id '{subject_id}'")
            if subject_id not in example_by_subject:
                raise ValueError(f"Split file references subject_id '{subject_id}' that is missing from merged metadata")
            split_sets[split_name].append(example_by_subject[subject_id])
            seen_subject_ids.add(subject_id)

    missing_subject_ids = sorted(set(example_by_subject) - seen_subject_ids)
    if missing_subject_ids:
        preview = ", ".join(missing_subject_ids[:5])
        raise ValueError(f"Split file is missing {len(missing_subject_ids)} subjects. First few: {preview}")

    for split_name, split_examples in split_sets.items():
        if not split_examples:
            raise ValueError(f"Split file produced an empty '{split_name}' split")

    return split_sets, fold_name
