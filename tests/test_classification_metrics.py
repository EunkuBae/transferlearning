from brainage.utils.metrics import classification_metrics


def test_classification_metrics_basic_case():
    metrics = classification_metrics(
        targets=[0, 1, 2, 0, 1, 2],
        predictions=[0, 1, 2, 1, 1, 0],
        num_classes=3,
    )

    assert round(float(metrics["accuracy"]), 4) == 0.6667
    assert round(float(metrics["balanced_accuracy"]), 4) == 0.6667
    assert len(metrics["confusion_matrix"]) == 3
