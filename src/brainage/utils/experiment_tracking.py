from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

SUMMARY_FIELD_ORDER = [
    "timestamp",
    "started_at",
    "ended_at",
    "status",
    "experiment_name",
    "task",
    "config_path",
    "output_dir",
    "run_record_dir",
    "checkpoint_path",
    "source_checkpoint_path",
    "use_demographics",
    "holdout_cohort",
    "fold_name",
    "load_mode",
    "freeze_backbone",
    "selection_metric",
    "loss_name",
    "device",
    "num_examples",
    "split_train",
    "split_val",
    "split_test",
    "split_source",
    "split_file",
    "metadata_path",
    "csv_path",
    "image_dir",
    "cache_dir",
    "best_val_mae",
    "best_val_mse",
    "best_val_rmse",
    "best_val_pearson_r",
    "best_val_r2",
    "best_val_accuracy",
    "best_val_balanced_accuracy",
    "best_val_macro_precision",
    "best_val_macro_recall",
    "best_val_macro_f1",
    "test_mae",
    "test_mse",
    "test_rmse",
    "test_pearson_r",
    "test_r2",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "best_val_metrics_json",
    "test_metrics_json",
    "resolved_paths_json",
]


def record_experiment_run(
    *,
    experiment_name: str,
    output_dir: Path,
    config_path: Path,
    metrics_payload: dict[str, Any],
    resolved_paths: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    started_at = os.environ.get("BRAINAGE_RUN_STARTED_AT", now.strftime("%Y-%m-%dT%H:%M:%S"))
    ended_at = now.strftime("%Y-%m-%dT%H:%M:%S")
    status = os.environ.get("BRAINAGE_RUN_STATUS", "success")

    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    run_record_dir_env = os.environ.get("BRAINAGE_RUN_RECORD_DIR")
    if run_record_dir_env:
        run_record_dir = Path(run_record_dir_env).resolve()
        wrapper_managed = True
    else:
        run_record_dir = (output_dir / "run_history" / timestamp).resolve()
        wrapper_managed = False
    run_record_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = run_record_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = run_record_dir / "config_snapshot.yaml"
    shutil.copy2(config_path, config_snapshot_path)

    copied_artifacts: dict[str, str | None] = {}
    for artifact_name, artifact_path in artifact_paths.items():
        if artifact_path is None:
            copied_artifacts[artifact_name] = None
            continue
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            copied_artifacts[artifact_name] = None
            continue
        target_path = artifacts_dir / artifact_path.name
        shutil.copy2(artifact_path, target_path)
        copied_artifacts[artifact_name] = str(target_path)

    output_root = _locate_output_root(output_dir)
    global_metrics_dir = output_root / "metrics"
    global_metrics_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": timestamp,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": status,
        "experiment_name": experiment_name,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "run_record_dir": str(run_record_dir),
        "config_snapshot": str(config_snapshot_path),
        "artifact_paths": copied_artifacts,
        "metrics_json": str(artifact_paths.get("metrics_json", "")),
        "history_json": str(artifact_paths.get("history_json", "")),
        "resolved_paths_json": str(artifact_paths.get("resolved_paths_json", "")),
        "test_predictions_csv": str(artifact_paths.get("test_predictions_csv", "")),
        "summary_txt": str(artifact_paths.get("summary_txt", "")),
    }

    summary_row = _build_summary_row(
        timestamp=timestamp,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        experiment_name=experiment_name,
        config_path=config_path,
        output_dir=output_dir,
        run_record_dir=run_record_dir,
        metrics_payload=metrics_payload,
        resolved_paths=resolved_paths,
    )

    (run_record_dir / "paper_summary.json").write_text(json.dumps(summary_row, indent=2) + "\n", encoding="utf-8")
    (run_record_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    if not wrapper_managed:
        local_registry = output_dir / "run_registry.jsonl"
        _append_jsonl(local_registry, record)

    _append_jsonl(global_metrics_dir / "experiment_runs.jsonl", summary_row)
    _append_csv(global_metrics_dir / "experiment_runs.csv", summary_row)
    return summary_row


def _locate_output_root(output_dir: Path) -> Path:
    for candidate in [output_dir, *output_dir.parents]:
        if candidate.name == "outputs":
            return candidate
    return output_dir.parent


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELD_ORDER, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in SUMMARY_FIELD_ORDER})


def _build_summary_row(
    *,
    timestamp: str,
    started_at: str,
    ended_at: str,
    status: str,
    experiment_name: str,
    config_path: Path,
    output_dir: Path,
    run_record_dir: Path,
    metrics_payload: dict[str, Any],
    resolved_paths: dict[str, Any],
) -> dict[str, Any]:
    split_sizes = metrics_payload.get("split_sizes", {}) or {}
    best_val_metrics = metrics_payload.get("best_val_metrics", {}) or {}
    test_metrics = metrics_payload.get("test_metrics", {}) or {}

    row: dict[str, Any] = {
        "timestamp": timestamp,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": status,
        "experiment_name": experiment_name,
        "task": metrics_payload.get("task"),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "run_record_dir": str(run_record_dir),
        "checkpoint_path": metrics_payload.get("checkpoint_path"),
        "source_checkpoint_path": metrics_payload.get("source_checkpoint_path"),
        "use_demographics": metrics_payload.get("use_demographics"),
        "holdout_cohort": metrics_payload.get("holdout_cohort"),
        "fold_name": metrics_payload.get("fold_name"),
        "load_mode": metrics_payload.get("load_mode"),
        "freeze_backbone": metrics_payload.get("freeze_backbone"),
        "selection_metric": metrics_payload.get("selection_metric"),
        "loss_name": metrics_payload.get("loss_name"),
        "device": metrics_payload.get("device"),
        "num_examples": metrics_payload.get("num_examples"),
        "split_train": split_sizes.get("train"),
        "split_val": split_sizes.get("val"),
        "split_test": split_sizes.get("test"),
        "split_source": metrics_payload.get("split_source"),
        "split_file": metrics_payload.get("split_file") or resolved_paths.get("split_file"),
        "metadata_path": metrics_payload.get("metadata_path") or resolved_paths.get("metadata_path"),
        "csv_path": resolved_paths.get("csv_path"),
        "image_dir": resolved_paths.get("image_dir"),
        "cache_dir": resolved_paths.get("cache_dir"),
        "best_val_mae": metrics_payload.get("best_val_mae", best_val_metrics.get("mae")),
        "best_val_mse": best_val_metrics.get("mse"),
        "best_val_rmse": best_val_metrics.get("rmse"),
        "best_val_pearson_r": best_val_metrics.get("pearson_r"),
        "best_val_r2": best_val_metrics.get("r2"),
        "best_val_accuracy": best_val_metrics.get("accuracy"),
        "best_val_balanced_accuracy": best_val_metrics.get("balanced_accuracy"),
        "best_val_macro_precision": best_val_metrics.get("macro_precision"),
        "best_val_macro_recall": best_val_metrics.get("macro_recall"),
        "best_val_macro_f1": best_val_metrics.get("macro_f1"),
        "test_mae": test_metrics.get("mae"),
        "test_mse": test_metrics.get("mse"),
        "test_rmse": test_metrics.get("rmse"),
        "test_pearson_r": test_metrics.get("pearson_r"),
        "test_r2": test_metrics.get("r2"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "test_macro_precision": test_metrics.get("macro_precision"),
        "test_macro_recall": test_metrics.get("macro_recall"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "best_val_metrics_json": json.dumps(best_val_metrics, sort_keys=True),
        "test_metrics_json": json.dumps(test_metrics, sort_keys=True),
        "resolved_paths_json": json.dumps(resolved_paths, sort_keys=True),
    }
    return row
