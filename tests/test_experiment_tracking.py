import json
from pathlib import Path

from brainage.utils.experiment_tracking import record_experiment_run


def test_record_experiment_run_creates_snapshots_and_global_registry(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "outputs" / "demo_experiment"
    output_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: demo\n", encoding="utf-8")

    metrics_json = output_dir / "metrics.json"
    history_json = output_dir / "history.json"
    resolved_paths_json = output_dir / "resolved_paths.json"
    predictions_csv = output_dir / "test_predictions.csv"
    summary_txt = output_dir / "training_summary.txt"

    metrics_json.write_text('{"ok": true}\n', encoding="utf-8")
    history_json.write_text('[]\n', encoding="utf-8")
    resolved_paths_json.write_text('{"output_dir": "demo"}\n', encoding="utf-8")
    predictions_csv.write_text('subject_id,target,prediction\n', encoding="utf-8")
    summary_txt.write_text('summary\n', encoding="utf-8")

    monkeypatch.delenv("BRAINAGE_RUN_RECORD_DIR", raising=False)
    monkeypatch.delenv("BRAINAGE_RUN_STARTED_AT", raising=False)

    summary_row = record_experiment_run(
        experiment_name="demo_experiment",
        output_dir=output_dir,
        config_path=config_path,
        metrics_payload={
            "experiment_name": "demo_experiment",
            "num_examples": 10,
            "split_sizes": {"train": 6, "val": 2, "test": 2},
            "selection_metric": "balanced_accuracy",
            "test_metrics": {"accuracy": 0.5, "macro_f1": 0.4},
            "best_val_metrics": {"accuracy": 0.6},
        },
        resolved_paths={
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "image_dir": "/tmp/images",
            "cache_dir": "/tmp/cache",
        },
        artifact_paths={
            "metrics_json": metrics_json,
            "history_json": history_json,
            "resolved_paths_json": resolved_paths_json,
            "test_predictions_csv": predictions_csv,
            "summary_txt": summary_txt,
        },
    )

    run_record_dir = Path(summary_row["run_record_dir"])
    assert run_record_dir.exists()
    assert (run_record_dir / "artifacts" / "metrics.json").exists()
    assert (run_record_dir / "paper_summary.json").exists()

    global_jsonl = tmp_path / "outputs" / "metrics" / "experiment_runs.jsonl"
    global_csv = tmp_path / "outputs" / "metrics" / "experiment_runs.csv"
    assert global_jsonl.exists()
    assert global_csv.exists()

    entries = [json.loads(line) for line in global_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries
    assert entries[-1]["experiment_name"] == "demo_experiment"
    assert entries[-1]["test_accuracy"] == 0.5
