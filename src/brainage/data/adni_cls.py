"""ADNI baseline T1 classification dataset utilities."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from brainage.data.hcp_mmse import HCPMMSEDataset, require_hcp_dependencies


VALID_ADNI_DIAGNOSIS = ("NC", "MCI", "AD")
ADNI_LABEL_TO_INDEX = {label: index for index, label in enumerate(VALID_ADNI_DIAGNOSIS)}


@dataclass(frozen=True)
class ADNIClassificationExample:
    subject_id: str
    image_path: Path
    diagnosis: str
    age: float | None = None
    sex: str | None = None
    mmse: float | None = None


def normalize_adni_diagnosis(raw_label: str) -> str | None:
    label = str(raw_label or "").strip().upper()
    if label == "CN":
        return "NC"
    if label in VALID_ADNI_DIAGNOSIS:
        return label
    return None


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return float(stripped)


def discover_adni_classification_examples(
    metadata_path: Path,
    image_dir: Path,
    subject_id_column: str = "PTID",
    diagnosis_column: str = "DX_bl",
    age_column: str = "AGE",
    sex_column: str = "PTGENDER",
    mmse_column: str = "MMSE",
    image_name_column: str = "file_name",
) -> list[ADNIClassificationExample]:
    require_hcp_dependencies()
    image_lookup = {path.name: path for path in sorted(image_dir.glob("*.nii"))}

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {metadata_path}")

        examples: list[ADNIClassificationExample] = []
        seen_subject_ids: set[str] = set()
        for row in reader:
            subject_id = str(row.get(subject_id_column, "")).strip()
            diagnosis = normalize_adni_diagnosis(str(row.get(diagnosis_column, "")))
            image_name = str(row.get(image_name_column, "")).strip()

            if not subject_id or diagnosis is None or not image_name:
                continue
            if subject_id in seen_subject_ids:
                continue

            image_path = image_lookup.get(image_name)
            if image_path is None:
                continue

            examples.append(
                ADNIClassificationExample(
                    subject_id=subject_id,
                    image_path=image_path,
                    diagnosis=diagnosis,
                    age=_parse_optional_float(row.get(age_column)),
                    sex=str(row.get(sex_column, "")).strip().upper() or None,
                    mmse=_parse_optional_float(row.get(mmse_column)),
                )
            )
            seen_subject_ids.add(subject_id)

    if not examples:
        raise ValueError("No ADNI examples were matched between metadata and image directory.")

    return examples


def stratified_split_examples(
    examples: list[ADNIClassificationExample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[ADNIClassificationExample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    grouped: dict[str, list[ADNIClassificationExample]] = {label: [] for label in VALID_ADNI_DIAGNOSIS}
    for example in examples:
        grouped[example.diagnosis].append(example)

    rng = random.Random(seed)
    split_sets = {"train": [], "val": [], "test": []}
    for label in VALID_ADNI_DIAGNOSIS:
        label_examples = list(grouped[label])
        if not label_examples:
            continue
        rng.shuffle(label_examples)

        total = len(label_examples)
        test_count = max(1, int(total * test_ratio))
        val_count = max(1, int(total * val_ratio))
        train_end = total - val_count - test_count
        if train_end <= 0:
            raise ValueError(f"Split ratios leave no training samples for class '{label}'.")

        split_sets["train"].extend(label_examples[:train_end])
        split_sets["val"].extend(label_examples[train_end : train_end + val_count])
        split_sets["test"].extend(label_examples[train_end + val_count :])

    for split_name in split_sets:
        rng.shuffle(split_sets[split_name])

    return split_sets


class ADNIClassificationDataset(HCPMMSEDataset):
    """Dataset that loads ADNI baseline T1w MRI volumes for classification."""

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        volume = self._load_volume(example)
        item = {
            "image": volume,
            "target": self._build_target(example.diagnosis),
            "target_name": example.diagnosis,
            "subject_id": example.subject_id,
        }
        if self.use_demographics:
            item["tabular"] = self._build_tabular_features(example)
        return item

    @staticmethod
    def _build_target(diagnosis: str):
        require_hcp_dependencies()
        import torch

        return torch.tensor(ADNI_LABEL_TO_INDEX[diagnosis], dtype=torch.long)
