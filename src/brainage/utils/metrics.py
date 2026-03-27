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
