"""Run HCP-only MMSE regression with a 3D CNN baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from brainage.data.hcp_mmse import HCPMMSEDataset, discover_hcp_mmse_examples, split_examples
from brainage.models.factory import build_hcp_mmse_model
from brainage.paths import resolve_path
from brainage.training.loops.regression import train_hcp_mmse_regressor
from brainage.utils.seed import set_global_seed

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Running HCP MMSE experiments requires `torch`. Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Running HCP MMSE experiments requires `PyYAML`. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HCP MMSE 3D-CNN baseline training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/hcp_mmse_baseline.yaml"),
        help="Path to HCP MMSE experiment config YAML.",
    )
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
        f"Device: {payload['device']}",
        f"Checkpoint: {payload['checkpoint_path']}",
        f"Use demographics: {payload['use_demographics']}",
        "",
        "Resolved Paths:",
        f"  csv_path: {payload['resolved_paths']['csv_path']}",
        f"  image_dir: {payload['resolved_paths']['image_dir']}",
        f"  output_dir: {payload['resolved_paths']['output_dir']}",
        f"  cache_dir: {payload['resolved_paths'].get('cache_dir', 'None')}",
        "",
        "Dataset:",
        f"  num_examples: {payload['num_examples']}",
        f"  train: {payload['split_sizes']['train']}",
        f"  val: {payload['split_sizes']['val']}",
        f"  test: {payload['split_sizes']['test']}",
        "",
        "Best Validation Metrics:",
    ]
    for key, value in payload['best_val_metrics'].items():
        lines.append(f"  {key}: {value}")
    lines.extend([
        "",
        "Test Metrics:",
    ])
    for key, value in payload['test_metrics'].items():
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


def resolve_config_path(raw_value: str, env_name: str | None, base_dir: Path) -> Path:
    env_value = os.environ.get(env_name) if env_name else None
    value = env_value or raw_value
    return resolve_path(value, base_dir)


def maybe_limit_examples(examples, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or max_samples >= len(examples):
        return examples
    return list(examples[:max_samples])


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    runtime_root = Path.cwd().resolve()

    seed = int(config.get("training", {}).get("seed", 42))
    set_global_seed(seed)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    use_demographics = bool(model_config.get("use_demographics", False))
    csv_path = resolve_config_path(
        raw_value=str(data_config["csv_file"]),
        env_name=data_config.get("csv_file_env"),
        base_dir=runtime_root,
    )
    image_dir = resolve_config_path(
        raw_value=str(data_config["image_dir"]),
        env_name=data_config.get("image_dir_env"),
        base_dir=runtime_root,
    )
    subject_id_column = str(data_config.get("subject_id_column", "subject_id"))
    target_column = str(data_config.get("target_column", "mmse"))

    examples = discover_hcp_mmse_examples(
        csv_path=csv_path,
        image_dir=image_dir,
        subject_id_column=subject_id_column,
        target_column=target_column,
        age_column=data_config.get("age_column", "age"),
        sex_column=data_config.get("sex_column", "sex"),
    )
    max_samples = data_config.get("max_samples")
    if max_samples is not None:
        examples = maybe_limit_examples(examples, int(max_samples))

    split_config = config.get("split", {})
    split_sets = split_examples(
        examples=examples,
        val_ratio=float(split_config.get("val_ratio", 0.15)),
        test_ratio=float(split_config.get("test_ratio", 0.15)),
        seed=seed,
    )

    image_size = tuple(int(value) for value in data_config.get("image_size", [96, 96, 96]))
    output_config = config.get("outputs", {})
    output_dir = resolve_config_path(
        raw_value=str(output_config.get("run_dir", "outputs/hcp_mmse_baseline")),
        env_name=output_config.get("run_dir_env"),
        base_dir=runtime_root,
    )
    cache_dir = None
    if output_config.get("cache_dir") is not None or output_config.get("cache_dir_env") is not None:
        cache_dir = resolve_config_path(
            raw_value=str(output_config.get("cache_dir", "outputs/cache/hcp_mmse")),
            env_name=output_config.get("cache_dir_env"),
            base_dir=runtime_root,
        )

    train_dataset = HCPMMSEDataset(
        split_sets["train"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "train" if cache_dir is not None else None,
        cache_prefix="train",
    )
    val_dataset = HCPMMSEDataset(
        split_sets["val"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "val" if cache_dir is not None else None,
        cache_prefix="val",
    )
    test_dataset = HCPMMSEDataset(
        split_sets["test"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "test" if cache_dir is not None else None,
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

    metrics_payload = {
        "experiment_name": config.get("experiment_name", "hcp_mmse"),
        "use_demographics": use_demographics,
        "num_examples": len(examples),
        "split_sizes": {key: len(value) for key, value in split_sets.items()},
        "device": results["device"],
        "best_val_mae": results["best_val_mae"],
        "best_val_metrics": results["best_val_metrics"],
        "test_metrics": {
            key: value
            for key, value in results["test_metrics"].items()
            if key != "predictions"
        },
        "checkpoint_path": str(results["checkpoint_path"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths = {
        "config_path": str(config_path),
        "runtime_root": str(runtime_root),
        "csv_path": str(csv_path),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
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
