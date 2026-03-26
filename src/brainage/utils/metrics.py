"""Metric helpers."""

from __future__ import annotations

import math


def regression_metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length.")
    if not targets:
        raise ValueError("targets must not be empty.")

    errors = [prediction - target for target, prediction in zip(targets, predictions, strict=True)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    mae = sum(absolute_errors) / len(errors)
    mse = sum(squared_errors) / len(errors)
    rmse = math.sqrt(mse)

    target_mean = sum(targets) / len(targets)
    prediction_mean = sum(predictions) / len(predictions)
    covariance = sum(
        (target - target_mean) * (prediction - prediction_mean)
        for target, prediction in zip(targets, predictions, strict=True)
    )
    target_var = sum((target - target_mean) ** 2 for target in targets)
    prediction_var = sum((prediction - prediction_mean) ** 2 for prediction in predictions)
    denominator = math.sqrt(target_var * prediction_var)
    pearson_r = covariance / denominator if denominator > 0 else 0.0
    ss_res = sum(squared_errors)
    ss_tot = target_var
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "pearson_r": float(pearson_r),
        "r2": float(r2),
    }



def classification_metrics(
    targets: list[int],
    predictions: list[int],
    num_classes: int,
) -> dict[str, float | list[list[int]] | dict[str, float]]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length.")
    if not targets:
        raise ValueError("targets must not be empty.")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")

    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        if not 0 <= target < num_classes:
            raise ValueError(f"target class index out of range: {target}")
        if not 0 <= prediction < num_classes:
            raise ValueError(f"prediction class index out of range: {prediction}")
        confusion[target][prediction] += 1

    total = len(targets)
    correct = sum(confusion[i][i] for i in range(num_classes))
    accuracy = correct / total

    recalls: list[float] = []
    precisions: list[float] = []
    f1_scores: list[float] = []
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    for class_index in range(num_classes):
        true_positive = confusion[class_index][class_index]
        false_negative = sum(confusion[class_index]) - true_positive
        false_positive = sum(confusion[row][class_index] for row in range(num_classes)) - true_positive

        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0
        precision = true_positive / precision_denominator if precision_denominator > 0 else 0.0
        f1_denominator = precision + recall
        f1 = (2.0 * precision * recall / f1_denominator) if f1_denominator > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        key = str(class_index)
        per_class_precision[key] = float(precision)
        per_class_recall[key] = float(recall)
        per_class_f1[key] = float(f1)

    balanced_accuracy = sum(recalls) / num_classes
    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1_scores) / num_classes

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }
