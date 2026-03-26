"""Run healthy multi-source MMSE pretraining with ERM or GroupDRO."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from brainage.data.mmse_pretraining import (
    MultiSourceMMSEDataset,
    discover_hcp_source_examples,
    discover_oasis_source_examples,
    domain_counts,
    split_multisource_examples,
)
from brainage.models.factory import build_hcp_mmse_model
from brainage.paths import resolve_path
from brainage.training.loops.multisource_regression import train_multisource_mmse_regressor
from brainage.utils.experiment_tracking import record_experiment_run
from brainage.utils.seed import set_global_seed

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Running multi-source MMSE pretraining requires `torch`.") from exc

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Running multi-source MMSE pretraining requires `PyYAML`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run healthy multi-source MMSE pretraining.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/mmse_pretraining_erm.yaml"))
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


def resolve_config_path(raw_value: str, env_name: str | None, base_dir: Path) -> Path:
    env_value = os.environ.get(env_name) if env_name else None
    value = env_value or raw_value
    return resolve_path(value, base_dir)


def write_predictions(path: Path, predictions: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "domain_name", "target_mmse", "predicted_mmse"])
        writer.writeheader()
        writer.writerows(predictions)


def write_summary_report(path: Path, payload: dict) -> None:
    lines = [
        f"Experiment: {payload['experiment_name']}",
        f"Device: {payload['device']}",
        f"Checkpoint: {payload['checkpoint_path']}",
        f"DG method: {payload['dg_method']}",
        f"Loss: {payload['loss_name']}",
        f"Selection metric: {payload['selection_metric']}",
        f"Use demographics: {payload['use_demographics']}",
        "",
        "Domain Counts:",
    ]
    for split_name, counts in payload["split_domain_counts"].items():
        lines.append(f"  {split_name}: {counts}")
    lines.extend(["", "Best Validation Metrics (overall):"])
    for key, value in payload["best_val_metrics"].get("overall", {}).items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Best Validation Metrics (by_domain):"])
    for domain_name, metrics in payload["best_val_metrics"].get("by_domain", {}).items():
        lines.append(f"  {domain_name}: {metrics}")
    lines.extend(["", "Test Metrics (overall):"])
    for key, value in payload["test_metrics"].get("overall", {}).items():
        if key != "predictions":
            lines.append(f"  {key}: {value}")
    lines.extend(["", "Test Metrics (by_domain):"])
    for domain_name, metrics in payload["test_metrics"].get("by_domain", {}).items():
        lines.append(f"  {domain_name}: {metrics}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    runtime_root = Path.cwd().resolve()

    training_config = config.get("training", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    output_config = config.get("outputs", {})
    seed = int(training_config.get("seed", 42))
    set_global_seed(seed)

    hcp_config = data_config.get("hcp", {})
    oasis_config = data_config.get("oasis", {})
    hcp_examples = discover_hcp_source_examples(
        csv_path=resolve_config_path(str(hcp_config["csv_file"]), hcp_config.get("csv_file_env"), runtime_root),
        image_dir=resolve_config_path(str(hcp_config["image_dir"]), hcp_config.get("image_dir_env"), runtime_root),
        subject_id_column=str(hcp_config.get("subject_id_column", "subject_id")),
        target_column=str(hcp_config.get("target_column", "mmse")),
        age_column=hcp_config.get("age_column", "age"),
        sex_column=hcp_config.get("sex_column", "sex"),
        domain_name=str(hcp_config.get("domain_name", "hcp")),
        domain_index=int(hcp_config.get("domain_index", 0)),
    )
    oasis_examples = discover_oasis_source_examples(
        metadata_path=resolve_config_path(str(oasis_config["metadata_file"]), oasis_config.get("metadata_file_env"), runtime_root),
        image_dir=resolve_config_path(str(oasis_config["image_dir"]), oasis_config.get("image_dir_env"), runtime_root),
        subject_id_column=str(oasis_config.get("subject_id_column", "subject_id")),
        target_column=str(oasis_config.get("target_column", "mmse")),
        age_column=str(oasis_config.get("age_column", "age")),
        sex_column=str(oasis_config.get("sex_column", "gender")),
        filter_column=oasis_config.get("filter_column"),
        filter_values=list(oasis_config.get("filter_values", [])),
        domain_name=str(oasis_config.get("domain_name", "oasis_nc")),
        domain_index=int(oasis_config.get("domain_index", 1)),
    )
    all_examples = list(hcp_examples) + list(oasis_examples)
    split_config = config.get("split", {})
    split_sets = split_multisource_examples(
        examples=all_examples,
        val_ratio=float(split_config.get("val_ratio", 0.15)),
        test_ratio=float(split_config.get("test_ratio", 0.15)),
        seed=seed,
    )

    image_size = tuple(int(value) for value in data_config.get("image_size", [96, 96, 96]))
    use_demographics = bool(model_config.get("use_demographics", False))
    output_dir = resolve_config_path(str(output_config.get("run_dir", "outputs/mmse_pretraining")), output_config.get("run_dir_env"), runtime_root)
    cache_dir = None
    if output_config.get("cache_dir") is not None or output_config.get("cache_dir_env") is not None:
        cache_dir = resolve_config_path(str(output_config.get("cache_dir", "outputs/cache/mmse_pretraining")), output_config.get("cache_dir_env"), runtime_root)

    train_dataset = MultiSourceMMSEDataset(split_sets["train"], image_size=image_size, use_demographics=use_demographics, cache_dir=cache_dir / "train" if cache_dir else None, cache_prefix="train")
    val_dataset = MultiSourceMMSEDataset(split_sets["val"], image_size=image_size, use_demographics=use_demographics, cache_dir=cache_dir / "val" if cache_dir else None, cache_prefix="val")
    test_dataset = MultiSourceMMSEDataset(split_sets["test"], image_size=image_size, use_demographics=use_demographics, cache_dir=cache_dir / "test" if cache_dir else None, cache_prefix="test")

    batch_size = int(training_config.get("batch_size", 2))
    num_workers = int(training_config.get("num_workers", 0))
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_hcp_mmse_model(config)
    results = train_multisource_mmse_regressor(model, train_loader, val_loader, test_loader, config=config, output_dir=output_dir)

    resolved_paths = {
        "config_path": str(config_path),
        "runtime_root": str(runtime_root),
        "hcp_csv_path": str(resolve_config_path(str(hcp_config["csv_file"]), hcp_config.get("csv_file_env"), runtime_root)),
        "hcp_image_dir": str(resolve_config_path(str(hcp_config["image_dir"]), hcp_config.get("image_dir_env"), runtime_root)),
        "oasis_metadata_path": str(resolve_config_path(str(oasis_config["metadata_file"]), oasis_config.get("metadata_file_env"), runtime_root)),
        "oasis_image_dir": str(resolve_config_path(str(oasis_config["image_dir"]), oasis_config.get("image_dir_env"), runtime_root)),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    metrics_payload = {
        "experiment_name": config.get("experiment_name", "mmse_pretraining"),
        "use_demographics": use_demographics,
        "dg_method": results["dg_method"],
        "group_weights": results["group_weights"],
        "num_examples": len(all_examples),
        "split_sizes": {key: len(value) for key, value in split_sets.items()},
        "split_domain_counts": {key: domain_counts(value) for key, value in split_sets.items()},
        "device": results["device"],
        "loss_name": results["loss_name"],
        "selection_metric": results["selection_metric"],
        "best_val_metrics": results["best_val_metrics"],
        "test_metrics": {key: value for key, value in results["test_metrics"].items() if key != "predictions"},
        "checkpoint_path": str(results["checkpoint_path"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = output_dir / "metrics.json"
    history_json_path = output_dir / "history.json"
    resolved_json_path = output_dir / "resolved_paths.json"
    predictions_csv_path = output_dir / "test_predictions.csv"
    summary_txt_path = output_dir / "training_summary.txt"

    resolved_json_path.write_text(json.dumps(resolved_paths, indent=2), encoding="utf-8")
    metrics_json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    history_json_path.write_text(json.dumps(results["history"], indent=2), encoding="utf-8")
    write_predictions(predictions_csv_path, results["test_metrics"]["predictions"])
    write_summary_report(summary_txt_path, metrics_payload)

    record_experiment_run(
        experiment_name=str(metrics_payload["experiment_name"]),
        output_dir=output_dir,
        config_path=config_path,
        metrics_payload=metrics_payload,
        resolved_paths=resolved_paths,
        artifact_paths={
            "metrics_json": metrics_json_path,
            "history_json": history_json_path,
            "resolved_paths_json": resolved_json_path,
            "test_predictions_csv": predictions_csv_path,
            "summary_txt": summary_txt_path,
        },
    )

    print(json.dumps({**metrics_payload, "summary_txt": str(summary_txt_path)}, indent=2))


if __name__ == "__main__":
    main()
