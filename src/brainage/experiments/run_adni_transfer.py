"""Run HCP-pretrained ADNI diagnosis classification transfer experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from brainage.data.adni_cls import (
    ADNI_LABEL_TO_INDEX,
    ADNIClassificationDataset,
    discover_adni_classification_examples,
    stratified_split_examples,
)
from brainage.data.hcp_mmse import HCPMMSEDataset
from brainage.models.factory import build_adni_classification_model, build_hcp_mmse_model
from brainage.paths import resolve_path
from brainage.training.loops.classification import train_adni_classifier
from brainage.utils.experiment_tracking import record_experiment_run
from brainage.utils.seed import set_global_seed

try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Running ADNI transfer experiments requires `torch`. Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Running ADNI transfer experiments requires `PyYAML`. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HCP-pretrained ADNI diagnosis classification transfer training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/adni_cls_transfer_full_ft.yaml"),
        help="Path to ADNI transfer experiment config YAML.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def resolve_config_path(raw_value: str, env_name: str | None, base_dir: Path) -> Path:
    env_value = os.environ.get(env_name) if env_name else None
    value = env_value or raw_value
    return resolve_path(value, base_dir)


def load_pretrained_weights(model, checkpoint_path: Path, load_mode: str) -> dict[str, list[str] | int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if load_mode == "full":
        candidate_state = dict(state_dict)
    elif load_mode == "backbone":
        candidate_state = {key: value for key, value in state_dict.items() if key.startswith("backbone.")}
    else:
        raise ValueError(f"Unsupported load_mode: {load_mode}")

    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped_mismatched_keys: list[str] = []

    for key, value in candidate_state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_mismatched_keys.append(key)
            continue
        compatible_state[key] = value

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    return {
        "loaded_keys": len(compatible_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "skipped_mismatched_keys": skipped_mismatched_keys,
    }


def load_regression_model_from_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise KeyError(f"Checkpoint does not contain a config: {checkpoint_path}")
    model = build_hcp_mmse_model(checkpoint_config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, checkpoint_config


def predict_mmse_lookup(
    examples: list,
    predictor_checkpoint_path: Path,
    image_size: tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    cache_dir: Path | None = None,
):
    regressor, regressor_config = load_regression_model_from_checkpoint(predictor_checkpoint_path)
    use_demographics = bool(regressor_config.get("model", {}).get("use_demographics", False))
    dataset = HCPMMSEDataset(
        examples,
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir,
        cache_prefix="mmse_aux",
    )
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regressor = regressor.to(device)

    predicted_lookup: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            tabular = batch.get("tabular")
            if tabular is not None:
                tabular = tabular.to(device, non_blocking=device.type == "cuda")
            outputs = regressor(images, tabular)
            for subject_id, prediction in zip(batch["subject_id"], outputs.detach().cpu().tolist(), strict=True):
                predicted_lookup[str(subject_id)] = float(prediction)
    return predicted_lookup


def build_mmse_aux_feature_lookup(
    split_sets: dict[str, list],
    transfer_config: dict,
    image_size: tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    runtime_root: Path,
    cache_dir: Path | None = None,
):
    aux_config = transfer_config.get("mmse_aux_features", {})
    if not bool(aux_config.get("enabled", False)):
        return None, None

    predictor_checkpoint_path = resolve_config_path(
        raw_value=str(aux_config.get("predictor_checkpoint", transfer_config["source_checkpoint"])),
        env_name=aux_config.get("predictor_checkpoint_env") or transfer_config.get("source_checkpoint_env"),
        base_dir=runtime_root,
    )
    combined_examples = []
    for split_name in ("train", "val", "test"):
        combined_examples.extend(split_sets[split_name])
    predicted_lookup = predict_mmse_lookup(
        examples=combined_examples,
        predictor_checkpoint_path=predictor_checkpoint_path,
        image_size=image_size,
        batch_size=int(aux_config.get("batch_size", batch_size)),
        num_workers=int(aux_config.get("num_workers", num_workers)),
        cache_dir=cache_dir / "mmse_aux" if cache_dir is not None else None,
    )

    feature_names: list[str] = []
    raw_lookup: dict[str, list[float]] = {}
    include_actual = bool(aux_config.get("include_actual_mmse", True))
    include_predicted = bool(aux_config.get("include_predicted_mmse", True))
    include_deviation = bool(aux_config.get("include_deviation", True))
    if include_actual:
        feature_names.append("actual_mmse")
    if include_predicted:
        feature_names.append("predicted_mmse")
    if include_deviation:
        feature_names.append("mmse_deviation")

    for split_name in ("train", "val", "test"):
        for example in split_sets[split_name]:
            actual_mmse = float(example.mmse) if example.mmse is not None else 0.0
            predicted_mmse = float(predicted_lookup[example.subject_id])
            deviation = predicted_mmse - actual_mmse
            values: list[float] = []
            if include_actual:
                values.append(actual_mmse)
            if include_predicted:
                values.append(predicted_mmse)
            if include_deviation:
                values.append(deviation)
            raw_lookup[example.subject_id] = values

    train_values = [raw_lookup[example.subject_id] for example in split_sets["train"]]
    means = []
    stds = []
    for feature_index in range(len(feature_names)):
        values = [row[feature_index] for row in train_values]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = variance ** 0.5
        means.append(mean)
        stds.append(std if std > 1e-6 else 1.0)

    normalized_lookup = {
        subject_id: [
            (value - means[index]) / stds[index]
            for index, value in enumerate(values)
        ]
        for subject_id, values in raw_lookup.items()
    }
    return normalized_lookup, {
        "enabled": True,
        "feature_names": feature_names,
        "predictor_checkpoint_path": str(predictor_checkpoint_path),
        "train_feature_stats": {
            feature_name: {"mean": float(means[index]), "std": float(stds[index])}
            for index, feature_name in enumerate(feature_names)
        },
    }


def apply_freeze_strategy(model, freeze_backbone: bool, trainable_backbone_stages: int | None = None) -> int:
    backbone_parameters = list(model.backbone.parameters())
    if freeze_backbone:
        for parameter in backbone_parameters:
            parameter.requires_grad = False
        return 0

    total_stages = len(model.backbone.features) // 4
    if trainable_backbone_stages is None or trainable_backbone_stages >= total_stages:
        for parameter in backbone_parameters:
            parameter.requires_grad = True
        return total_stages

    if trainable_backbone_stages <= 0:
        for parameter in backbone_parameters:
            parameter.requires_grad = False
        return 0

    for parameter in backbone_parameters:
        parameter.requires_grad = False

    modules_per_stage = 4
    start_module_index = max(0, (total_stages - trainable_backbone_stages) * modules_per_stage)
    for module in model.backbone.features[start_module_index:]:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return trainable_backbone_stages


def write_predictions(path: Path, predictions: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_id", "target_label", "target_index", "predicted_index"],
        )
        writer.writeheader()
        writer.writerows(predictions)


def write_summary_report(path: Path, payload: dict) -> None:
    lines = [
        f"Experiment: {payload['experiment_name']}",
        f"Device: {payload['device']}",
        f"Checkpoint: {payload['checkpoint_path']}",
        f"Source checkpoint: {payload['source_checkpoint_path']}",
        f"Load mode: {payload['load_mode']}",
        f"Freeze backbone: {payload['freeze_backbone']}",
        f"Trainable backbone stages: {payload.get('trainable_backbone_stages', 'all')}",
        f"Staged unfreeze: {payload.get('staged_unfreeze', 'none')}",
        f"Use demographics: {payload['use_demographics']}",
        f"Selection metric: {payload['selection_metric']}",
        f"Class weighting: {payload.get('class_weighting', 'none')}",
        f"Sampler strategy: {payload.get('sampler_strategy', 'none')}",
        f"MMSE auxiliary features: {payload.get('mmse_aux_features', {'enabled': False})}",
        "",
        "Resolved Paths:",
        f"  metadata_path: {payload['resolved_paths']['metadata_path']}",
        f"  image_dir: {payload['resolved_paths']['image_dir']}",
        f"  output_dir: {payload['resolved_paths']['output_dir']}",
        f"  cache_dir: {payload['resolved_paths'].get('cache_dir', 'None')}",
        f"  split_file: {payload['resolved_paths'].get('split_file', 'None')}",
        f"  split_source: {payload.get('split_source', 'unknown')}",
        "",
        "Dataset:",
        f"  num_examples: {payload['num_examples']}",
        f"  train: {payload['split_sizes']['train']}",
        f"  val: {payload['split_sizes']['val']}",
        f"  test: {payload['split_sizes']['test']}",
        "",
        "Checkpoint Load Report:",
        f"  loaded_keys: {payload['checkpoint_load_report'].get('loaded_keys', 0)}",
        f"  missing_keys: {payload['checkpoint_load_report']['missing_keys']}",
        f"  unexpected_keys: {payload['checkpoint_load_report']['unexpected_keys']}",
        f"  skipped_mismatched_keys: {payload['checkpoint_load_report'].get('skipped_mismatched_keys', [])}",
        "",
        "Label Mapping:",
    ]
    for label_name, label_index in payload["label_mapping"].items():
        lines.append(f"  {label_name}: {label_index}")
    if payload.get("class_weights") is not None:
        lines.extend(["", f"Class Weights: {payload['class_weights']}"])
    lines.extend(["", "Best Validation Metrics:"])
    for key, value in payload["best_val_metrics"].items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Test Metrics:"])
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


def write_split_assignments(path: Path, split_sets: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "split", "diagnosis"])
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for example in split_sets[split_name]:
                writer.writerow(
                    {
                        "subject_id": example.subject_id,
                        "split": split_name,
                        "diagnosis": example.diagnosis,
                    }
                )


def load_split_assignments(path: Path, examples: list) -> dict[str, list]:
    example_by_subject = {example.subject_id: example for example in examples}
    split_sets = {"train": [], "val": [], "test": []}
    seen_subject_ids: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(row.get("subject_id", "")).strip()
            split_name = str(row.get("split", "")).strip().lower()
            diagnosis = str(row.get("diagnosis", "")).strip().upper()
            if not subject_id:
                raise ValueError(f"Split file contains an empty subject_id: {path}")
            if split_name not in split_sets:
                raise ValueError(f"Split file contains invalid split '{split_name}' for subject_id={subject_id}")
            if subject_id in seen_subject_ids:
                raise ValueError(f"Split file contains duplicate subject_id '{subject_id}'")
            if subject_id not in example_by_subject:
                raise ValueError(f"Split file references subject_id '{subject_id}' that is missing from the dataset")
            if diagnosis and example_by_subject[subject_id].diagnosis != diagnosis:
                raise ValueError(f"Split file diagnosis mismatch for subject_id '{subject_id}'")
            split_sets[split_name].append(example_by_subject[subject_id])
            seen_subject_ids.add(subject_id)

    missing_subject_ids = sorted(set(example_by_subject) - seen_subject_ids)
    if missing_subject_ids:
        preview = ", ".join(missing_subject_ids[:5])
        raise ValueError(f"Split file is missing {len(missing_subject_ids)} subjects. First few: {preview}")

    return split_sets


def build_or_load_split_sets(examples: list, split_config: dict, seed: int, runtime_root: Path):
    split_file = None
    split_file_raw = split_config.get("split_file")
    split_file_env = split_config.get("split_file_env")
    if split_file_raw is not None or split_file_env is not None:
        split_file = resolve_config_path(
            raw_value=str(split_file_raw or "data/splits/adni_cls_seed42.csv"),
            env_name=split_file_env,
            base_dir=runtime_root,
        )

    if split_file is not None and split_file.exists():
        split_sets = load_split_assignments(split_file, examples)
        split_source = "existing_split_file"
    else:
        split_sets = stratified_split_examples(
            examples=examples,
            val_ratio=float(split_config.get("val_ratio", 0.15)),
            test_ratio=float(split_config.get("test_ratio", 0.15)),
            seed=seed,
        )
        split_source = "generated_from_seed"
        if split_file is not None and bool(split_config.get("save_split_file", True)):
            write_split_assignments(split_file, split_sets)
            split_source = "generated_and_saved_split_file"

    return split_sets, split_file, split_source


def compute_class_weights(examples: list, num_classes: int):
    counts = Counter(ADNI_LABEL_TO_INDEX[example.diagnosis] for example in examples)
    total = sum(counts.values())
    weights = []
    for class_index in range(num_classes):
        count = counts.get(class_index, 0)
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def build_balanced_sampler(examples: list):
    counts = Counter(ADNI_LABEL_TO_INDEX[example.diagnosis] for example in examples)
    sample_weights = [1.0 / counts[ADNI_LABEL_TO_INDEX[example.diagnosis]] for example in examples]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    runtime_root = Path.cwd().resolve()

    seed = int(config.get("training", {}).get("seed", 42))
    set_global_seed(seed)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    transfer_config = config.get("transfer", {})
    output_config = config.get("outputs", {})
    use_demographics = bool(model_config.get("use_demographics", False))

    metadata_path = resolve_config_path(
        raw_value=str(data_config["metadata_file"]),
        env_name=data_config.get("metadata_file_env"),
        base_dir=runtime_root,
    )
    image_dir = resolve_config_path(
        raw_value=str(data_config["image_dir"]),
        env_name=data_config.get("image_dir_env"),
        base_dir=runtime_root,
    )
    source_checkpoint_path = resolve_config_path(
        raw_value=str(transfer_config["source_checkpoint"]),
        env_name=transfer_config.get("source_checkpoint_env"),
        base_dir=runtime_root,
    )

    examples = discover_adni_classification_examples(
        metadata_path=metadata_path,
        image_dir=image_dir,
        subject_id_column=str(data_config.get("subject_id_column", "PTID")),
        diagnosis_column=str(data_config.get("diagnosis_column", "DX_bl")),
        age_column=str(data_config.get("age_column", "AGE")),
        sex_column=str(data_config.get("sex_column", "PTGENDER")),
        mmse_column=str(data_config.get("mmse_column", "MMSE")),
        image_name_column=str(data_config.get("image_name_column", "file_name")),
    )

    split_config = config.get("split", {})
    split_sets, split_file, split_source = build_or_load_split_sets(
        examples=examples,
        split_config=split_config,
        seed=seed,
        runtime_root=runtime_root,
    )

    image_size = tuple(int(value) for value in data_config.get("image_size", [96, 96, 96]))
    output_dir = resolve_config_path(
        raw_value=str(output_config.get("run_dir", "outputs/adni_cls_transfer")),
        env_name=output_config.get("run_dir_env"),
        base_dir=runtime_root,
    )
    cache_dir = None
    if output_config.get("cache_dir") is not None or output_config.get("cache_dir_env") is not None:
        cache_dir = resolve_config_path(
            raw_value=str(output_config.get("cache_dir", "outputs/cache/adni_cls_transfer")),
            env_name=output_config.get("cache_dir_env"),
            base_dir=runtime_root,
        )

    training_config = config.get("training", {})
    staged_unfreeze = training_config.get("staged_unfreeze")
    batch_size = int(training_config.get("batch_size", 2))
    num_workers = int(training_config.get("num_workers", 0))
    num_classes = int(model_config.get("num_classes", len(ADNI_LABEL_TO_INDEX)))

    mmse_aux_feature_lookup, mmse_aux_metadata = build_mmse_aux_feature_lookup(
        split_sets=split_sets,
        transfer_config=transfer_config,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        runtime_root=runtime_root,
        cache_dir=cache_dir,
    )
    aux_feature_dim = len(mmse_aux_metadata["feature_names"]) if mmse_aux_metadata is not None else 0
    config.setdefault("model", {})["tabular_input_dim"] = (2 if use_demographics else 0) + aux_feature_dim

    train_dataset = ADNIClassificationDataset(
        split_sets["train"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "train" if cache_dir is not None else None,
        cache_prefix="train",
        aux_feature_lookup=mmse_aux_feature_lookup,
    )
    val_dataset = ADNIClassificationDataset(
        split_sets["val"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "val" if cache_dir is not None else None,
        cache_prefix="val",
        aux_feature_lookup=mmse_aux_feature_lookup,
    )
    test_dataset = ADNIClassificationDataset(
        split_sets["test"],
        image_size=image_size,
        use_demographics=use_demographics,
        cache_dir=cache_dir / "test" if cache_dir is not None else None,
        cache_prefix="test",
        aux_feature_lookup=mmse_aux_feature_lookup,
    )

    sampler_strategy = str(training_config.get("sampler", "none")).lower()
    train_sampler = None
    if sampler_strategy == "balanced":
        train_sampler = build_balanced_sampler(split_sets["train"])
    elif sampler_strategy not in {"none", "off", "false", "0", ""}:
        raise ValueError(f"Unsupported sampler strategy: {sampler_strategy}")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        sampler=train_sampler,
    )
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_weight_mode = str(training_config.get("class_weighting", "none")).lower()
    class_weights = None
    if class_weight_mode == "balanced":
        class_weights = compute_class_weights(split_sets["train"], num_classes=num_classes)
    elif class_weight_mode not in {"none", "off", "false", "0", ""}:
        raise ValueError(f"Unsupported class_weighting strategy: {class_weight_mode}")

    model = build_adni_classification_model(config)
    checkpoint_load_report = load_pretrained_weights(
        model,
        checkpoint_path=source_checkpoint_path,
        load_mode=str(transfer_config.get("load_mode", "backbone")),
    )
    trainable_backbone_stages_config = transfer_config.get("trainable_backbone_stages")
    trainable_backbone_stages = apply_freeze_strategy(
        model,
        freeze_backbone=bool(transfer_config.get("freeze_backbone", False)),
        trainable_backbone_stages=(
            None if trainable_backbone_stages_config is None else int(trainable_backbone_stages_config)
        ),
    )

    results = train_adni_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=output_dir,
        class_weights=class_weights,
    )

    metrics_payload = {
        "experiment_name": config.get("experiment_name", "adni_cls_transfer"),
        "use_demographics": use_demographics,
        "source_checkpoint_path": str(source_checkpoint_path),
        "load_mode": str(transfer_config.get("load_mode", "backbone")),
        "freeze_backbone": bool(transfer_config.get("freeze_backbone", False)),
        "trainable_backbone_stages": trainable_backbone_stages,
        "staged_unfreeze": staged_unfreeze,
        "num_examples": len(examples),
        "split_sizes": {key: len(value) for key, value in split_sets.items()},
        "label_mapping": ADNI_LABEL_TO_INDEX,
        "device": results["device"],
        "selection_metric": results["selection_metric"],
        "best_val_metrics": results["best_val_metrics"],
        "checkpoint_load_report": checkpoint_load_report,
        "test_metrics": {key: value for key, value in results["test_metrics"].items() if key != "predictions"},
        "checkpoint_path": str(results["checkpoint_path"]),
        "class_weights": results["class_weights"],
        "class_weighting": class_weight_mode,
        "sampler_strategy": sampler_strategy,
        "mmse_aux_features": mmse_aux_metadata or {"enabled": False},
        "split_source": split_source,
        "split_file": str(split_file) if split_file is not None else None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths = {
        "config_path": str(config_path),
        "runtime_root": str(runtime_root),
        "metadata_path": str(metadata_path),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "split_file": str(split_file) if split_file is not None else None,
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

    record_experiment_run(
        experiment_name=str(metrics_payload["experiment_name"]),
        output_dir=output_dir,
        config_path=config_path,
        metrics_payload=metrics_payload,
        resolved_paths=resolved_paths,
        artifact_paths={
            "metrics_json": metrics_json_path,
            "history_json": history_json_path,
            "resolved_paths_json": output_dir / "resolved_paths.json",
            "test_predictions_csv": predictions_csv_path,
            "summary_txt": summary_txt_path,
        },
    )

    print(json.dumps({**metrics_payload, "summary_txt": str(summary_txt_path)}, indent=2))


if __name__ == "__main__":
    main()
