"""Entry point stub for LODO experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from brainage.paths import get_data_root, get_metadata_root, get_output_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LODO experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and config summary without training.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load experiment configs. Install it with `pip install -r requirements.txt`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_runtime_paths(config: dict, config_path: Path) -> dict[str, Path]:
    config_dir = config_path.parent
    env_config = config.get("environment", {})
    data_root = Path(os.environ.get(env_config.get("data_root_env", "BRAINAGE_DATA_ROOT"), str(get_data_root())))
    output_root = Path(
        os.environ.get(env_config.get("output_root_env", "BRAINAGE_OUTPUT_ROOT"), str(get_output_root()))
    )
    metadata_root = Path(
        os.environ.get(env_config.get("metadata_root_env", "BRAINAGE_METADATA_ROOT"), str(get_metadata_root()))
    )

    metadata_file = config.get("data", {}).get("metadata_file", "merged_metadata.csv")
    split_file = config.get("split", {}).get("split_file", "data/splits/lodo_adni_holdout.csv")
    outputs = config.get("outputs", {})

    return {
        "data_root": data_root,
        "metadata_root": metadata_root,
        "metadata_file": resolve_path(metadata_file, metadata_root),
        "split_file": resolve_path(split_file, config_dir),
        "output_root": output_root,
        "checkpoint_dir": resolve_path(outputs.get("checkpoint_dir", "outputs/checkpoints"), config_dir),
        "prediction_dir": resolve_path(outputs.get("prediction_dir", "outputs/predictions"), config_dir),
        "metrics_dir": resolve_path(outputs.get("metrics_dir", "outputs/metrics"), config_dir),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resolved = resolve_runtime_paths(config, args.config.resolve())

    print(f"Experiment: {config.get('experiment_name', 'unknown')}")
    print(f"Task: {config.get('task', 'unknown')}")
    print(f"Train mode: {config.get('train_mode', 'unknown')}")
    print(f"Holdout cohort: {config.get('split', {}).get('holdout_cohort', 'unknown')}")
    print("Resolved paths:")
    for key, value in resolved.items():
        print(f"  - {key}: {value}")

    if args.dry_run:
        print("Dry run complete. Training loop is not implemented yet.")
        return

    print("[TODO] Connect the resolved config and paths to the training pipeline.")


if __name__ == "__main__":
    main()
