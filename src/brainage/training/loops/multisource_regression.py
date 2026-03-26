from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

from brainage.models.backbones.cnn3d import require_torch
from brainage.utils.metrics import regression_metrics

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


HIGHER_IS_BETTER_METRICS = {"pearson_r", "r2"}


def train_multisource_mmse_regressor(model, train_loader, val_loader, test_loader, config: dict, output_dir: Path):
    require_torch()
    training_config = config.get("training", {})
    dg_config = config.get("dg", {})
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
    best_val_metrics: dict[str, object] | None = None
    history: list[dict[str, object]] = []

    domain_names = _discover_domain_names(train_loader, val_loader, test_loader)
    group_weights = torch.ones(len(domain_names), dtype=torch.float32, device=device)
    if len(domain_names) > 0:
        group_weights = group_weights / group_weights.sum()
    groupdro_eta = float(dg_config.get("groupdro_eta", 0.1))
    dg_method = str(dg_config.get("method", "erm")).lower()

    epochs = int(training_config.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        train_summary = _run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            mixed_precision=mixed_precision,
            dg_method=dg_method,
            group_weights=group_weights,
            groupdro_eta=groupdro_eta,
        )
        val_metrics = evaluate_multisource_regression_model(model, val_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(train_summary["loss"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "val_mae": float(val_metrics["overall"]["mae"]),
            "val_rmse": float(val_metrics["overall"]["rmse"]),
            "val_pearson_r": float(val_metrics["overall"]["pearson_r"]),
            "group_weights": {name: float(weight) for name, weight in zip(domain_names, group_weights.detach().cpu().tolist(), strict=True)},
            "train_domain_losses": train_summary["domain_losses"],
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch}/{epochs} train_loss={train_summary['loss']:.4f} "
            f"val_mae={val_metrics['overall']['mae']:.4f} "
            f"val_rmse={val_metrics['overall']['rmse']:.4f} "
            f"val_r={val_metrics['overall']['pearson_r']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        current_metric_value = float(val_metrics["overall"][selection_metric])
        if _is_better(current_metric_value, best_metric_value, selection_metric):
            best_metric_value = current_metric_value
            best_val_metrics = val_metrics
            best_state = deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler_metric = float(val_metrics["overall"].get(str(training_config.get("scheduler_metric", "mae")).lower(), val_metrics["overall"]["mae"]))
            scheduler.step(scheduler_metric)

    model.load_state_dict(best_state)
    test_metrics = evaluate_multisource_regression_model(model, test_loader, device)

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
            "dg_method": dg_method,
            "group_weights": {name: float(weight) for name, weight in zip(domain_names, group_weights.detach().cpu().tolist(), strict=True)},
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
        "loss_name": str(training_config.get("loss", "mse")).lower(),
        "dg_method": dg_method,
        "group_weights": {name: float(weight) for name, weight in zip(domain_names, group_weights.detach().cpu().tolist(), strict=True)},
    }


def evaluate_multisource_regression_model(model, data_loader, device):
    require_torch()
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    subject_ids: list[str] = []
    domain_names: list[str] = []

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
            domain_names.extend(list(batch["domain_name"]))

    overall = regression_metrics(targets, predictions)
    by_domain: dict[str, dict[str, float]] = {}
    grouped_targets: dict[str, list[float]] = defaultdict(list)
    grouped_predictions: dict[str, list[float]] = defaultdict(list)
    for domain_name, target, prediction in zip(domain_names, targets, predictions, strict=True):
        grouped_targets[domain_name].append(target)
        grouped_predictions[domain_name].append(prediction)
    for domain_name in sorted(grouped_targets):
        by_domain[domain_name] = regression_metrics(grouped_targets[domain_name], grouped_predictions[domain_name])

    return {
        "overall": overall,
        "by_domain": by_domain,
        "predictions": [
            {
                "subject_id": subject_id,
                "domain_name": domain_name,
                "target_mmse": float(target),
                "predicted_mmse": float(prediction),
            }
            for subject_id, domain_name, target, prediction in zip(subject_ids, domain_names, targets, predictions, strict=True)
        ],
    }


def _run_epoch(model, data_loader, optimizer, criterion, device, scaler, mixed_precision: bool, dg_method: str, group_weights, groupdro_eta: float):
    model.train()
    total_loss = 0.0
    total_examples = 0
    domain_loss_accumulator: dict[str, list[float]] = defaultdict(list)

    autocast_context = torch.cuda.amp.autocast if mixed_precision else nullcontext

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device, non_blocking=device.type == "cuda")
        domain_indices = batch["domain_index"].to(device, non_blocking=device.type == "cuda")
        tabular = batch.get("tabular")
        if tabular is not None:
            tabular = tabular.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            predictions = model(images, tabular)
            per_example_loss = _per_example_loss(criterion, predictions, targets)
            if dg_method == "groupdro":
                loss, batch_domain_losses = _groupdro_loss(per_example_loss, domain_indices, group_weights, groupdro_eta)
            else:
                loss = per_example_loss.mean()
                batch_domain_losses = _domain_loss_dict(per_example_loss, domain_indices)

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
        batch_domain_names = list(batch["domain_name"])
        for domain_index, domain_loss in batch_domain_losses.items():
            domain_name = batch_domain_names[domain_indices.tolist().index(domain_index)] if domain_index in domain_indices.tolist() else str(domain_index)
            domain_loss_accumulator[domain_name].append(float(domain_loss))

    summarized_domain_losses = {
        domain_name: float(sum(values) / len(values))
        for domain_name, values in sorted(domain_loss_accumulator.items())
    }
    return {"loss": total_loss / max(total_examples, 1), "domain_losses": summarized_domain_losses}


def _domain_loss_dict(per_example_loss, domain_indices):
    losses = {}
    for domain_index in torch.unique(domain_indices).tolist():
        mask = domain_indices == int(domain_index)
        losses[int(domain_index)] = per_example_loss[mask].mean()
    return losses


def _groupdro_loss(per_example_loss, domain_indices, group_weights, groupdro_eta: float):
    domain_losses = _domain_loss_dict(per_example_loss, domain_indices)
    for domain_index, domain_loss in domain_losses.items():
        group_weights[domain_index] = group_weights[domain_index] * torch.exp(groupdro_eta * domain_loss.detach())
    group_weights /= group_weights.sum().clamp_min(1e-12)
    loss = torch.tensor(0.0, device=per_example_loss.device)
    for domain_index, domain_loss in domain_losses.items():
        loss = loss + group_weights[domain_index] * domain_loss
    return loss, domain_losses


def _per_example_loss(criterion, predictions, targets):
    if isinstance(criterion, nn.MSELoss):
        return (predictions - targets) ** 2
    if isinstance(criterion, nn.L1Loss):
        return torch.abs(predictions - targets)
    if isinstance(criterion, nn.HuberLoss):
        delta = float(criterion.delta)
        error = torch.abs(predictions - targets)
        quadratic = torch.minimum(error, torch.full_like(error, delta))
        linear = error - quadratic
        return 0.5 * quadratic ** 2 + delta * linear
    raise ValueError(f"Unsupported loss for per-example computation: {type(criterion).__name__}")


def _discover_domain_names(*loaders):
    domain_names = set()
    for loader in loaders:
        dataset = getattr(loader, "dataset", None)
        examples = getattr(dataset, "examples", None)
        if examples is None:
            continue
        for example in examples:
            domain_names.add(example.domain_name)
    return sorted(domain_names)


def _build_loss(training_config: dict):
    require_torch()
    loss_name = str(training_config.get("loss", "mse")).lower()
    if loss_name == "mse":
        return nn.MSELoss(reduction="none")
    if loss_name in {"mae", "l1"}:
        return nn.L1Loss(reduction="none")
    if loss_name in {"huber", "smooth_l1"}:
        delta = float(training_config.get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta, reduction="none")
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
