"""Training loop for ADNI classification."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

from brainage.models.backbones.cnn3d import require_torch
from brainage.utils.metrics import classification_metrics

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    torch = None
    nn = None


HIGHER_IS_BETTER_METRICS = {"accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall"}


def train_adni_classifier(
    model,
    train_loader,
    val_loader,
    test_loader,
    config: dict,
    output_dir: Path,
    class_weights=None,
):
    require_torch()
    training_config = config.get("training", {})
    num_classes = int(config.get("model", {}).get("num_classes", 3))
    device = _resolve_device(training_config.get("device", "auto"))
    mixed_precision = _resolve_mixed_precision(training_config.get("mixed_precision", "auto"), device)
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    mmse_aux_weight = float(training_config.get("mmse_aux_loss_weight", 0.0))
    mmse_criterion = _build_mmse_criterion(training_config) if mmse_aux_weight > 0 else None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-5)),
    )
    scheduler = _build_scheduler(training_config, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(training_config.get("cudnn_benchmark", True))

    selection_metric = str(training_config.get("selection_metric", "balanced_accuracy")).lower()
    best_state = deepcopy(model.state_dict())
    best_metric_value = float("-inf")
    best_val_metrics: dict[str, float | list[list[int]]] | None = None
    history: list[dict[str, float]] = []

    epochs = int(training_config.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        _apply_staged_unfreeze_if_needed(model, training_config, epoch)
        train_stats = _run_epoch(
            model,
            train_loader,
            optimizer,
            classification_criterion,
            mmse_criterion,
            mmse_aux_weight,
            device,
            scaler,
            mixed_precision,
        )
        val_metrics = evaluate_classification_model(model, val_loader, device, num_classes=num_classes)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_classification_loss": float(train_stats["classification_loss"]),
            "train_mmse_loss": float(train_stats["mmse_loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_cls_loss={train_stats['classification_loss']:.4f} "
            f"train_mmse_loss={train_stats['mmse_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        current_metric_value = float(val_metrics[selection_metric])
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_val_metrics = {key: value for key, value in val_metrics.items() if key != "predictions"}
            best_state = deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler_metric = float(
                val_metrics.get(str(training_config.get("scheduler_metric", selection_metric)).lower(), val_metrics[selection_metric])
            )
            scheduler.step(scheduler_metric)

    model.load_state_dict(best_state)
    test_metrics = evaluate_classification_model(model, test_loader, device, num_classes=num_classes)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "selection_metric": selection_metric,
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        },
        checkpoint_path,
    )

    return {
        "history": history,
        "best_val_metrics": best_val_metrics or {},
        "test_metrics": test_metrics,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "selection_metric": selection_metric,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "mmse_aux_loss_weight": mmse_aux_weight,
    }


def evaluate_classification_model(model, data_loader, device, num_classes: int):
    require_torch()
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    subject_ids: list[str] = []
    target_names: list[str] = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            batch_targets = batch["target"].to(device, non_blocking=device.type == "cuda")
            tabular = batch.get("tabular")
            if tabular is not None:
                tabular = tabular.to(device, non_blocking=device.type == "cuda")
            outputs = model(images, tabular)
            logits, _ = _split_model_outputs(outputs)
            batch_predictions = torch.argmax(logits, dim=1)

            predictions.extend(batch_predictions.detach().cpu().tolist())
            targets.extend(batch_targets.detach().cpu().tolist())
            subject_ids.extend(list(batch["subject_id"]))
            target_names.extend(list(batch["target_name"]))

    metrics = classification_metrics(targets, predictions, num_classes=num_classes)
    metrics["predictions"] = [
        {
            "subject_id": subject_id,
            "target_label": target_name,
            "target_index": int(target),
            "predicted_index": int(prediction),
        }
        for subject_id, target_name, target, prediction in zip(subject_ids, target_names, targets, predictions, strict=True)
    ]
    return metrics


def _run_epoch(model, data_loader, optimizer, classification_criterion, mmse_criterion, mmse_aux_weight, device, scaler, mixed_precision: bool):
    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_mmse_loss = 0.0
    total_examples = 0

    autocast_context = torch.cuda.amp.autocast if mixed_precision else nullcontext

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device, non_blocking=device.type == "cuda")
        tabular = batch.get("tabular")
        if tabular is not None:
            tabular = tabular.to(device, non_blocking=device.type == "cuda")
        mmse_targets = batch.get("mmse")
        if mmse_targets is not None:
            mmse_targets = mmse_targets.to(device, non_blocking=device.type == "cuda").float()

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            outputs = model(images, tabular)
            logits, mmse_pred = _split_model_outputs(outputs)
            classification_loss = classification_criterion(logits, targets)
            mmse_loss = _compute_mmse_aux_loss(mmse_criterion, mmse_pred, mmse_targets, logits)
            loss = classification_loss + (mmse_aux_weight * mmse_loss)

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_classification_loss += classification_loss.item() * batch_size
        total_mmse_loss += mmse_loss.item() * batch_size
        total_examples += batch_size

    denominator = max(total_examples, 1)
    return {
        "loss": total_loss / denominator,
        "classification_loss": total_classification_loss / denominator,
        "mmse_loss": total_mmse_loss / denominator,
    }


def _split_model_outputs(outputs):
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        if logits is None:
            raise KeyError("Model output dictionary must contain 'logits'")
        return logits, outputs.get("mmse_pred")
    return outputs, None


def _compute_mmse_aux_loss(mmse_criterion, mmse_pred, mmse_targets, reference_tensor):
    if mmse_criterion is None or mmse_pred is None or mmse_targets is None:
        return reference_tensor.new_tensor(0.0)
    return mmse_criterion(mmse_pred.float(), mmse_targets.float())


def _build_mmse_criterion(training_config: dict):
    require_torch()
    loss_name = str(training_config.get("mmse_aux_loss", "mse")).lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"l1", "mae"}:
        return nn.L1Loss()
    if loss_name in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss(beta=float(training_config.get("mmse_aux_loss_beta", 1.0)))
    raise ValueError(f"Unsupported mmse_aux_loss: {loss_name}")


def _build_scheduler(training_config: dict, optimizer):
    require_torch()
    scheduler_name = str(training_config.get("scheduler", "none")).lower()
    if scheduler_name in {"none", "off", "false", "0", ""}:
        return None
    if scheduler_name == "plateau":
        metric_name = str(training_config.get("scheduler_metric", "balanced_accuracy")).lower()
        mode = "max" if metric_name in HIGHER_IS_BETTER_METRICS else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(training_config.get("scheduler_factor", 0.5)),
            patience=int(training_config.get("scheduler_patience", 3)),
            min_lr=float(training_config.get("scheduler_min_lr", 1e-6)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _apply_staged_unfreeze_if_needed(model, training_config: dict, epoch: int) -> None:
    staged_unfreeze = training_config.get("staged_unfreeze")
    if not isinstance(staged_unfreeze, dict):
        return

    unfreeze_epoch = int(staged_unfreeze.get("epoch", 0))
    if unfreeze_epoch <= 0 or epoch != unfreeze_epoch:
        return

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "features"):
        raise ValueError("staged_unfreeze requires a model with backbone.features")

    total_stages = len(model.backbone.features) // 4
    target = staged_unfreeze.get("trainable_backbone_stages_after")
    if target in {None, "all"}:
        trainable_backbone_stages = total_stages
    else:
        trainable_backbone_stages = int(target)

    _set_trainable_backbone_stages(model, trainable_backbone_stages)
    display_value = trainable_backbone_stages if trainable_backbone_stages < total_stages else "all"
    print(f"Staged unfreeze applied at epoch {epoch}: trainable_backbone_stages={display_value}")


def _set_trainable_backbone_stages(model, trainable_backbone_stages: int) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    total_stages = len(model.backbone.features) // 4
    if trainable_backbone_stages >= total_stages:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        return

    if trainable_backbone_stages <= 0:
        return

    modules_per_stage = 4
    start_module_index = max(0, (total_stages - trainable_backbone_stages) * modules_per_stage)
    for module in model.backbone.features[start_module_index:]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def _resolve_device(device_name: str):
    require_torch()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_mixed_precision(setting, device) -> bool:
    if isinstance(setting, bool):
        return setting and device.type == "cuda"
    if str(setting).lower() == "auto":
        return device.type == "cuda"
    return str(setting).lower() in {"1", "true", "yes", "fp16"} and device.type == "cuda"
