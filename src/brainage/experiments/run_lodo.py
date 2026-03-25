"""Run LODO MMSE regression experiments from merged metadata and frozen split files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from brainage.data.hcp_mmse import HCPMMSEDataset
from brainage.data.lodo_mmse import discover_lodo_mmse_examples, load_lodo_split_assignments
from brainage.models.factory import build_hcp_mmse_model
from brainage.paths import get_data_root, get_metadata_root, get_output_root, resolve_path
from brainage.training.loops.regression import train_hcp_mmse_regressor
from brainage.utils.seed import set_global_seed

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Running LODO MMSE experiments requires `torch`. Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Running LODO experiments requires `PyYAML`. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LODO MMSE regression experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config YAML.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def write_predictions(path: Path, predictions: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "target_mmse", "predicted_mmse"])
        writer.writeheader()
        writer.writerows(predictions)


def write_summary_report(path: Path, payload: dict) -> None:
    lines = [
        f"Experiment: {payload['experiment_name']}",
        f"Task: {payload['task']}",
        f"Device: {payload['device']}",
        f"Checkpoint: {payload['checkpoint_path']}",
        f"Use demographics: {payload['use_demographics']}",
        f"Loss: {payload['loss_name']}",
        f"Selection metric: {payload['selection_metric']}",
        f"Holdout cohort: {payload['holdout_cohort']}",
        f"Fold name: {payload['fold_name']}",
        "",
        "Resolved Paths:",
        f"  metadata_path: {payload['resolved_paths']['metadata_path']}",
        f"  split_file: {payload['resolved_paths']['split_file']}",
        f"  output_dir: {payload['resolved_paths']['output_dir']}",
        f"  cache_dir: {payload['resolved_paths'].get('cache_dir', 'None')}",
        "",
        "Dataset:",
        f"  num_examples: {payload['num_examples']}",
        f"  train: {payload['split_sizes']['train']}",
        f"  val: {payload['split_sizes']['val']}",
        f"  test: {payload['split_sizes']['test']}",
        f"  cohort_counts: {payload['cohort_counts']}",
        "",
        "Best Validation Metrics:",
    ]
    for key, value in payload["best_val_metrics"].items():
        lines.append(f"  {key}: {value}")
    lines.extend([
        "",
        "Test Metrics:",
    ])
    for key, value in payload["test_metrics"].items():
        lines.append(f"  {key}: {value}")
    lines.extend([
        "",
        "Artifacts:",
        f"  metrics_json: {payload['metrics_json']}",
        f"  history_json: {payload['history_json']}",
        f"  test_predictions_csv: {payload['test_predictions_csv']}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    data_config = config.get("data", {})
    split_config = config.get("split", {})
    output_config = config.get("outputs", {})

    metadata_file = data_config.get("metadata_file", "merged_metadata.csv")
    split_file = split_config.get("split_file", "data/splits/lodo_adni_holdout.csv")

    output_base_dir = output_root.parent if output_root.name else output_root

    return {
        "data_root": data_root,
        "metadata_root": metadata_root,
        "metadata_path": resolve_path(metadata_file, metadata_root),
        "split_file": resolve_path(split_file, config_dir),
        "output_root": output_root,
        "output_dir": resolve_path(output_config.get("run_dir", "outputs/lodo_mmse_adni_holdout"), output_base_dir),
        "cache_dir": resolve_path(output_config.get("cache_dir", "outputs/cache/lodo_mmse_adni_holdout"), output_base_dir),
    }


def maybe_limit_examples(examples, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or max_samples >= len(examples):
        return examples
    return list(examples[:max_samples])


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    resolved = resolve_runtime_paths(config, config_path)

    seed = int(config.get("training", {}).get("seed", 42))
    set_global_seed(seed)

    model_config = config.get("model", {})
    use_demographics = bool(model_config.get("use_demographics", False))

    data_config = config.get("data", {})
    examples = discover_lodo_mmse_examples(
        metadata_path=resolved["metadata_path"],
        data_root=resolved["data_root"],
        cohort_filter={str(value).lower() for value in data_config.get("cohorts", ["hcp", "adni"])},
    )
    max_samples = data_config.get("max_samples")
    if max_samples is not None:
        examples = maybe_limit_examples(examples, int(max_samples))

    split_sets, fold_name = load_lodo_split_assignments(resolved["split_file"], examples)

    image_size = tuple(int(value) for value in data_config.get("image_size", [96, 96, 96]))
    cache_dir = resolved["cache_dir"]
    output_dir = resolved["output_dir"]

    train_dataset = HCPMMSEDataset(
        split_sets["train"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "train",
        cache_prefix="train",
    )
    val_dataset = HCPMMSEDataset(
        split_sets["val"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "val",
        cache_prefix="val",
    )
    test_dataset = HCPMMSEDataset(
        split_sets["test"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "test",
        cache_prefix="test",
    )

    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 2))
    num_workers = int(training_config.get("num_workers", 0))

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_hcp_mmse_model(config)
    results = train_hcp_mmse_regressor(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=output_dir,
    )

    cohort_counts: dict[str, int] = {}
    for example in examples:
        cohort_name = "adni" if "C3_ADNI" in str(example.image_path) else "hcp"
        cohort_counts[cohort_name] = cohort_counts.get(cohort_name, 0) + 1

    metrics_payload = {
        "experiment_name": config.get("experiment_name", "lodo_mmse_regression"),
        "task": config.get("task", "mmse_regression"),
        "use_demographics": use_demographics,
        "holdout_cohort": str(config.get("split", {}).get("holdout_cohort", "unknown")).lower(),
        "fold_name": fold_name,
        "num_examples": len(examples),
        "split_sizes": {key: len(value) for key, value in split_sets.items()},
        "cohort_counts": cohort_counts,
        "device": results["device"],
        "loss_name": results["loss_name"],
        "selection_metric": results["selection_metric"],
        "best_val_mae": results["best_val_mae"],
        "best_val_metrics": results["best_val_metrics"],
        "test_metrics": {key: value for key, value in results["test_metrics"].items() if key != "predictions"},
        "checkpoint_path": str(results["checkpoint_path"]),
        "split_file": str(resolved["split_file"]),
        "metadata_path": str(resolved["metadata_path"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths = {
        "config_path": str(config_path),
        "metadata_path": str(resolved["metadata_path"]),
        "split_file": str(resolved["split_file"]),
        "data_root": str(resolved["data_root"]),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
    }
    metrics_json_path = output_dir / "metrics.json"
    history_json_path = output_dir / "history.json"
    predictions_csv_path = output_dir / "test_predictions.csv"
    summary_txt_path = output_dir / "training_summary.txt"

    with (output_dir / "resolved_paths.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_paths, handle, indent=2)
    with metrics_json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    with history_json_path.open("w", encoding="utf-8") as handle:
        json.dump(results["history"], handle, indent=2)
    write_predictions(predictions_csv_path, results["test_metrics"]["predictions"])
    write_summary_report(
        summary_txt_path,
        {
            **metrics_payload,
            "resolved_paths": resolved_paths,
            "metrics_json": str(metrics_json_path),
            "history_json": str(history_json_path),
            "test_predictions_csv": str(predictions_csv_path),
        },
    )

    print(json.dumps({**metrics_payload, "summary_txt": str(summary_txt_path)}, indent=2))


if __name__ == "__main__":
    main()
