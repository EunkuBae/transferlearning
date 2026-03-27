"""ADNI MMSE metadata loading utilities."""

from __future__ import annotations

import csv
from pathlib import Path, PureWindowsPath

from brainage.data.hcp_mmse import HCPMMSEExample


ADNI_DIAGNOSIS_NORMALIZATION = {
    "CN": "NC",
    "MCI": "MCI",
    "LMCI": "MCI",
    "EMCI": "MCI",
    "AD": "AD",
}


def normalize_adni_diagnosis(raw_value: str | None) -> str | None:
    value = str(raw_value or "").strip().upper()
    return ADNI_DIAGNOSIS_NORMALIZATION.get(value)


def normalize_adni_sex(raw_value: str | None) -> str | None:
    value = str(raw_value or "").strip().upper()
    if value in {"M", "MALE"}:
        return "M"
    if value in {"F", "FEMALE"}:
        return "F"
    return None


def basename_any_path(raw_value: str | None) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    windows_name = PureWindowsPath(value).name
    posix_name = Path(value).name
    return windows_name if len(windows_name) < len(posix_name) else posix_name


def discover_adni_mmse_examples(
    metadata_path: Path,
    image_dir: Path,
    subject_id_column: str = "PTID",
    target_column: str = "MMSE",
    age_column: str = "AGE",
    sex_column: str = "PTGENDER",
    diagnosis_column: str = "DX_bl",
) -> list[HCPMMSEExample]:
    image_lookup = {path.name: path for path in sorted(image_dir.glob("*.nii"))}

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"ADNI metadata CSV has no header row: {metadata_path}")

        examples: list[HCPMMSEExample] = []
        seen_subject_ids: set[str] = set()
        for row_index, row in enumerate(reader, start=2):
            subject_id = str(row.get(subject_id_column, "")).strip()
            mmse_raw = str(row.get(target_column, "")).strip()
            if not subject_id or not mmse_raw or subject_id in seen_subject_ids:
                continue

            diagnosis = normalize_adni_diagnosis(row.get(diagnosis_column))
            if diagnosis is None:
                continue

            image_path = None
            for key in ("copied_file", "file_name", "source_file"):
                candidate_name = basename_any_path(row.get(key))
                if not candidate_name:
                    continue
                image_path = image_lookup.get(candidate_name)
                if image_path is not None:
                    break
            if image_path is None:
                continue

            try:
                mmse = float(mmse_raw)
            except ValueError as exc:
                raise ValueError(f"Row {row_index}: invalid MMSE value '{mmse_raw}'") from exc

            age_raw = str(row.get(age_column, "")).strip()
            age = float(age_raw) if age_raw else None
            sex = normalize_adni_sex(row.get(sex_column))

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
        raise ValueError("No ADNI MMSE examples were matched between the CSV metadata and MRI files.")

    return examples
