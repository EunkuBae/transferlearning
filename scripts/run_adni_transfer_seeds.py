from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADNI transfer config across multiple seeds.")
    parser.add_argument("--config", type=Path, required=True, help="Base transfer config YAML.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seed values to run.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/adni_transfer_seed_runs"),
        help="Directory where per-seed configs and summaries are stored.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.config.resolve())
    base_name = str(base_config.get("experiment_name", args.config.stem))
    output_root = args.output_root.resolve() / base_name
    output_root.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, object]] = []
    for seed in args.seeds:
        config = deepcopy(base_config)
        config["experiment_name"] = f"{base_name}_seed{seed}"
        config.setdefault("training", {})["seed"] = int(seed)

        outputs = config.setdefault("outputs", {})
        base_run_dir = str(outputs.get("run_dir", f"outputs/{base_name}"))
        base_cache_dir = str(outputs.get("cache_dir", f"outputs/cache/{base_name}"))
        outputs["run_dir"] = f"{base_run_dir}_seed{seed}"
        outputs["cache_dir"] = f"{base_cache_dir}_seed{seed}"

        config_path = output_root / f"seed_{seed}.yaml"
        write_yaml(config_path, config)

        command = [
            sys.executable,
            "-m",
            "brainage.experiments.run_adni_transfer",
            "--config",
            str(config_path),
        ]
        subprocess.run(command, check=True)

        metrics_path = Path(outputs["run_dir"]) / "metrics.json"
        run_records.append(
            {
                "seed": seed,
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
            }
        )

    summary_path = output_root / "seed_runs.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(run_records, handle, indent=2)


if __name__ == "__main__":
    main()
