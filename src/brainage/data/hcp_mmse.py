"""HCP MMSE dataset utilities for 3D CNN regression."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_IMPORT_ERROR: Exception | None = None

try:
    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    _IMPORT_ERROR = exc
    nib = None
    np = None
    torch = None
    F = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback Dataset placeholder when torch is unavailable."""

        pass


def require_hcp_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "HCP MMSE training requires `torch`, `numpy`, and `nibabel`. "
            "Install them with `pip install -r requirements.txt`."
        ) from _IMPORT_ERROR


@dataclass(frozen=True)
class HCPMMSEExample:
    subject_id: str
    image_path: Path
    mmse: float
    age: float | None = None
    sex: str | None = None


def extract_subject_id_from_filename(filename: str) -> str:
    return filename.split("_", 1)[0].strip()


def _resolve_csv_column(fieldnames: Iterable[str], desired_name: str) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames if name is not None}
    key = desired_name.strip().lower()
    if key not in normalized:
        raise KeyError(f"Column '{desired_name}' was not found in CSV headers: {sorted(normalized)}")
    return normalized[key]


def _parse_optional_float(value: str | None, row_index: int, column_name: str) -> float | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"Row {row_index}: invalid {column_name} value '{stripped}'") from exc


def discover_hcp_mmse_examples(
    csv_path: Path,
    image_dir: Path,
    subject_id_column: str = "subject_id",
    target_column: str = "mmse",
    age_column: str | None = "age",
    sex_column: str | None = "sex",
) -> list[HCPMMSEExample]:
    image_lookup = {
        extract_subject_id_from_filename(path.name): path
        for path in sorted(image_dir.glob("*.nii.gz"))
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        subject_col = _resolve_csv_column(reader.fieldnames, subject_id_column)
        target_col = _resolve_csv_column(reader.fieldnames, target_column)
        age_col = _resolve_csv_column(reader.fieldnames, age_column) if age_column else None
        sex_col = _resolve_csv_column(reader.fieldnames, sex_column) if sex_column else None

        examples: list[HCPMMSEExample] = []
        seen_subject_ids: set[str] = set()
        for row_index, row in enumerate(reader, start=2):
            subject_id = str(row.get(subject_col, "")).strip()
            target_value = str(row.get(target_col, "")).strip()

            if not subject_id or not target_value:
                continue
            if subject_id in seen_subject_ids:
                continue
            image_path = image_lookup.get(subject_id)
            if image_path is None:
                continue

            try:
                mmse = float(target_value)
            except ValueError as exc:
                raise ValueError(f"Row {row_index}: invalid MMSE value '{target_value}'") from exc

            age = _parse_optional_float(row.get(age_col) if age_col else None, row_index, age_col or "age")
            sex = str(row.get(sex_col, "")).strip().upper() if sex_col else None
            if sex == "":
                sex = None

            seen_subject_ids.add(subject_id)
            examples.append(
                HCPMMSEExample(
                    subject_id=subject_id,
                    image_path=image_path,
                    mmse=mmse,
                    age=age,
                    sex=sex,
                )
            )

    if not examples:
        raise ValueError("No HCP MMSE examples were matched between the CSV and image directory.")

    return examples


def split_examples(
    examples: list[HCPMMSEExample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[HCPMMSEExample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = max(1, int(total * test_ratio))
    val_count = max(1, int(total * val_ratio))
    train_end = total - val_count - test_count
    if train_end <= 0:
        raise ValueError("Split ratios leave no training samples.")

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end : train_end + val_count],
        "test": shuffled[train_end + val_count :],
    }


class HCPMMSEDataset(Dataset):
    """Dataset that loads HCP T1w MRI volumes for MMSE regression."""

    def __init__(
        self,
        examples: list[HCPMMSEExample],
        image_size: tuple[int, int, int],
        use_demographics: bool = False,
        cache_dir: Path | None = None,
        cache_prefix: str = "dataset",
    ) -> None:
        require_hcp_dependencies()
        self.examples = examples
        self.image_size = image_size
        self.use_demographics = use_demographics
        self.cache_dir = cache_dir
        self.cache_prefix = cache_prefix
        self.age_mean, self.age_std = self._compute_age_stats(examples)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        volume = self._load_volume(example)
        target = torch.tensor(example.mmse, dtype=torch.float32)
        item = {
            "image": volume,
            "target": target,
            "subject_id": example.subject_id,
        }
        if self.use_demographics:
            item["tabular"] = self._build_tabular_features(example)
        return item

    def _build_tabular_features(self, example: HCPMMSEExample):
        age = 0.0 if example.age is None else (example.age - self.age_mean) / self.age_std
        sex = 1.0 if str(example.sex or "").upper() == "M" else 0.0
        return torch.tensor([age, sex], dtype=torch.float32)

    def _cache_path(self, example: HCPMMSEExample) -> Path | None:
        if self.cache_dir is None:
            return None
        size_str = "x".join(str(v) for v in self.image_size)
        return self.cache_dir / f"{self.cache_prefix}_{example.subject_id}_{size_str}.pt"

    def _load_volume(self, example: HCPMMSEExample):
        cache_path = self._cache_path(example)
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, map_location="cpu")

        image = nib.load(str(example.image_path))
        volume = image.get_fdata(dtype=np.float32)

        if volume.ndim == 4:
            volume = volume[..., 0]
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D MRI volume, got shape {volume.shape} for {example.image_path}")

        volume = np.nan_to_num(volume, copy=False)
        volume = self._zscore(volume)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        volume_tensor = F.interpolate(
            volume_tensor,
            size=self.image_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).contiguous()

        if cache_path is not None:
            torch.save(volume_tensor, cache_path)

        return volume_tensor

    @staticmethod
    def _compute_age_stats(examples: list[HCPMMSEExample]) -> tuple[float, float]:
        ages = [float(example.age) for example in examples if example.age is not None]
        if not ages:
            return 0.0, 1.0
        mean = sum(ages) / len(ages)
        variance = sum((age - mean) ** 2 for age in ages) / len(ages)
        std = variance ** 0.5
        return mean, std if std > 1e-6 else 1.0

    @staticmethod
    def _zscore(volume):
        nonzero = volume[np.nonzero(volume)]
        if nonzero.size == 0:
            return volume.astype(np.float32)

        mean = float(nonzero.mean())
        std = float(nonzero.std())
        if std < 1e-6:
            return (volume - mean).astype(np.float32)
        return ((volume - mean) / std).astype(np.float32)
