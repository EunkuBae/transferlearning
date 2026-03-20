"""Shared schemas for cohort metadata and experiment splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    cohort: str
    image_path: Path
    diagnosis: str
    age: Optional[float] = None
    sex: Optional[str] = None
    mmse: Optional[float] = None
    split: Optional[str] = None
    site: Optional[str] = None


@dataclass(frozen=True)
class SplitAssignment:
    subject_id: str
    cohort: str
    split: str
    fold_name: str


REQUIRED_METADATA_COLUMNS = (
    "subject_id",
    "cohort",
    "image_path",
    "diagnosis",
)

RECOMMENDED_METADATA_COLUMNS = (
    "age",
    "sex",
    "mmse",
    "site",
    "scan_id",
    "split_group",
    "notes",
)

VALID_COHORTS = (
    "hcp",
    "oasis",
    "adni",
)

VALID_DIAGNOSIS_LABELS = (
    "NC",
    "QC",
    "MCI",
    "AD",
)
