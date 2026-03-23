"""Training loop for HCP MMSE regression."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

from brainage.models.backbones.cnn3d import require_torch
from brainage.utils.metrics import regression_metrics

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    torch = None
    nn = None


HIGHER_IS_BETTER_METRICS = {"pearson_r", "r2"}


def train_hcp_mmse_regressor(
    model,
    train_loader,
    val_loader,
    test_loader,
    config: dict,
    output_dir: Path,
):
    require_torch()
    training_config = config.get("training", {})
    device = _resolve_device(training_config.get("device", "auto"))
    mixed_precision = _resolve_mixed_precision(training_config.get("mixed_precision", "auto"), device)
    model = model.to(device)

    criterion = _build_loss(training_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-5)),
    )
    scheduler = _build_scheduler(training_config, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(training_config.get("cudnn_benchmark", True))

    selection_metric = str(training_config.get("selection_metric", "mae")).lower()
    best_state = deepcopy(model.state_dict())
    best_metric_value = float("-inf") if selection_metric in HIGHER_IS_BETTER_METRICS else float("inf")
    best_val_mae = float("inf")
    best_val_metrics: dict[str, float] | None = None
    history: list[dict[str, float]] = []

    epochs = int(training_config.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, device, scaler, mixed_precision)
        val_metrics = evaluate_regression_model(model, val_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_pearson_r": val_metrics["pearson_r"],
            "val_r2": val_metrics["r2"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_mse={val_metrics['mse']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_r={val_metrics['pearson_r']:.4f} "
            f"val_r2={val_metrics['r2']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        current_metric_value = float(val_metrics[selection_metric])
        if _is_better(current_metric_value, best_metric_value, selection_metric):
            best_metric_value = current_metric_value
            best_val_mae = float(val_metrics["mae"])
            best_val_metrics = {key: value for key, value in val_metrics.items() if key != "predictions"}
            best_state = deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler_metric = float(val_metrics.get(str(training_config.get("scheduler_metric", "mae")).lower(), val_metrics["mae"]))
            scheduler.step(scheduler_metric)

    model.load_state_dict(best_state)
    test_metrics = evaluate_regression_model(model, test_loader, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "loss_name": str(training_config.get("loss", "mse")).lower(),
            "selection_metric": selection_metric,
        },
        checkpoint_path,
    )

    return {
        "history": history,
        "best_val_mae": best_val_mae,
        "best_val_metrics": best_val_metrics or {},
        "test_metrics": test_metrics,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "selection_metric": selection_metric,
        "loss_name": str(training_config.get("loss", "mse")).lower(),
    }


def evaluate_regression_model(model, data_loader, device):
    require_torch()
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    subject_ids: list[str] = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            batch_targets = batch["target"].to(device, non_blocking=device.type == "cuda")
            tabular = batch.get("tabular")
            if tabular is not None:
                tabular = tabular.to(device, non_blocking=device.type == "cuda")
            outputs = model(images, tabular)

            predictions.extend(outputs.detach().cpu().tolist())
            targets.extend(batch_targets.detach().cpu().tolist())
            subject_ids.extend(list(batch["subject_id"]))

    metrics = regression_metrics(targets, predictions)
    metrics["predictions"] = [
        {
            "subject_id": subject_id,
            "target_mmse": float(target),
            "predicted_mmse": float(prediction),
        }
        for subject_id, target, prediction in zip(subject_ids, targets, predictions, strict=True)
    ]
    return metrics


def _run_epoch(model, data_loader, optimizer, criterion, device, scaler, mixed_precision: bool) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    autocast_context = torch.cuda.amp.autocast if mixed_precision else nullcontext

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device, non_blocking=device.type == "cuda")
        tabular = batch.get("tabular")
        if tabular is not None:
            tabular = tabular.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            predictions = model(images, tabular)
            loss = criterion(predictions, targets)

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def _build_loss(training_config: dict):
    require_torch()
    loss_name = str(training_config.get("loss", "mse")).lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"mae", "l1"}:
        return nn.L1Loss()
    if loss_name in {"huber", "smooth_l1"}:
        delta = float(training_config.get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta)
    raise ValueError(f"Unsupported regression loss: {loss_name}")


def _build_scheduler(training_config: dict, optimizer):
    require_torch()
    scheduler_name = str(training_config.get("scheduler", "none")).lower()
    if scheduler_name in {"none", "off", "false", "0", ""}:
        return None
    if scheduler_name == "plateau":
        metric_name = str(training_config.get("scheduler_metric", "mae")).lower()
        mode = "max" if metric_name in HIGHER_IS_BETTER_METRICS else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(training_config.get("scheduler_factor", 0.5)),
            patience=int(training_config.get("scheduler_patience", 3)),
            min_lr=float(training_config.get("scheduler_min_lr", 1e-6)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _is_better(current: float, best: float, metric_name: str) -> bool:
    if metric_name in HIGHER_IS_BETTER_METRICS:
        return current > best
    return current < best


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
