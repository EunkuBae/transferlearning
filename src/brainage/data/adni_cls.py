"""ADNI baseline T1 classification dataset utilities."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

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
    if label in {"MCI", "LMCI", "EMCI"}:
        return "MCI"
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


def _extract_any_path_basename(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    windows_name = PureWindowsPath(value).name
    posix_name = Path(value).name
    return windows_name if len(windows_name) < len(posix_name) else posix_name


def _candidate_image_names(row: dict[str, str], image_name_column: str) -> list[str]:
    candidates: list[str] = []
    for column_name in (image_name_column, "copied_file", "source_file"):
        raw_value = str(row.get(column_name, "") or "").strip()
        if not raw_value:
            continue
        name = _extract_any_path_basename(raw_value)
        if name and name not in candidates:
            candidates.append(name)
    return candidates


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
    image_lookup: dict[str, Path] = {}
    for pattern in ("*.nii", "*.nii.gz"):
        for image_path in sorted(image_dir.rglob(pattern)):
            image_lookup.setdefault(image_path.name, image_path)

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {metadata_path}")

        examples: list[ADNIClassificationExample] = []
        seen_subject_ids: set[str] = set()
        for row in reader:
            subject_id = str(row.get(subject_id_column, "")).strip()
            diagnosis = normalize_adni_diagnosis(str(row.get(diagnosis_column, "")))
            candidate_names = _candidate_image_names(row, image_name_column)

            if not subject_id or diagnosis is None or not candidate_names:
                continue
            if subject_id in seen_subject_ids:
                continue

            image_path = None
            for candidate_name in candidate_names:
                image_path = image_lookup.get(candidate_name)
                if image_path is not None:
                    break
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

    def __init__(
        self,
        examples: list[ADNIClassificationExample],
        image_size: tuple[int, int, int],
        use_demographics: bool = False,
        cache_dir: Path | None = None,
        cache_prefix: str = "dataset",
        aux_feature_lookup: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__(
            examples=examples,
            image_size=image_size,
            use_demographics=use_demographics,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
        )
        self.aux_feature_lookup = aux_feature_lookup or {}

    def __getitem__(self, index: int) -> dict[str, object]:
        require_hcp_dependencies()
        import torch

        example = self.examples[index]
        volume = self._load_volume(example)
        item = {
            "image": volume,
            "target": self._build_target(example.diagnosis),
            "target_name": example.diagnosis,
            "subject_id": example.subject_id,
            "mmse": float(example.mmse) if example.mmse is not None else None,
        }

        tabular_parts = []
        if self.use_demographics:
            tabular_parts.append(self._build_tabular_features(example))
        if self.aux_feature_lookup:
            aux_values = self.aux_feature_lookup.get(example.subject_id)
            if aux_values is None:
                raise KeyError(f"Missing auxiliary features for subject_id='{example.subject_id}'")
            tabular_parts.append(torch.tensor(aux_values, dtype=torch.float32))
        if tabular_parts:
            item["tabular"] = torch.cat(tabular_parts, dim=0)
        return item

    @staticmethod
    def _build_target(diagnosis: str):
        require_hcp_dependencies()
        import torch

        return torch.tensor(ADNI_LABEL_TO_INDEX[diagnosis], dtype=torch.long)
