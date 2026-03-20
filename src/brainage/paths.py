"""Common path helpers."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def get_env_path(name: str, default: Path | None = None) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is None:
        raise KeyError(f"Environment variable '{name}' is not set and no default was provided.")
    return default


def get_data_root() -> Path:
    return get_env_path("BRAINAGE_DATA_ROOT", DATA_DIR)


def get_output_root() -> Path:
    return get_env_path("BRAINAGE_OUTPUT_ROOT", OUTPUTS_DIR)


def get_metadata_root() -> Path:
    return get_env_path("BRAINAGE_METADATA_ROOT", METADATA_DIR)


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        return (base_dir / path).resolve()
    return (PROJECT_ROOT / path).resolve()
