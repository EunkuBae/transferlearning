"""Migrate legacy brainage_outputs summaries into the repository outputs directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

LIGHTWEIGHT_EXTENSIONS = {".json", ".csv", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in outputs.")
    parser.add_argument(
        "--remove-legacy",
        action="store_true",
        help="Remove migrated legacy files and delete empty legacy directories when done.",
    )
    return parser.parse_args()


def migrate_run_dir(legacy_run_dir: Path, outputs_root: Path, overwrite: bool) -> list[tuple[Path, Path]]:
    destination_dir = outputs_root / legacy_run_dir.name
    destination_dir.mkdir(parents=True, exist_ok=True)

    migrated: list[tuple[Path, Path]] = []
    for path in sorted(legacy_run_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in LIGHTWEIGHT_EXTENSIONS:
            continue
        destination = destination_dir / path.name
        if destination.exists() and not overwrite:
            continue
        shutil.copy2(path, destination)
        migrated.append((path, destination))
    return migrated


def cleanup_legacy(legacy_root: Path) -> None:
    for child in sorted(legacy_root.iterdir(), reverse=True):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()
    if legacy_root.exists() and not any(legacy_root.iterdir()):
        legacy_root.rmdir()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    legacy_root = project_root / "brainage_outputs"
    outputs_root = project_root / "outputs"

    if not legacy_root.exists():
        print(f"No legacy directory found at: {legacy_root}")
        return

    outputs_root.mkdir(parents=True, exist_ok=True)
    migrated_count = 0

    for legacy_run_dir in sorted(path for path in legacy_root.iterdir() if path.is_dir()):
        migrated = migrate_run_dir(legacy_run_dir, outputs_root, overwrite=args.overwrite)
        for source, destination in migrated:
            print(f"Copied {source} -> {destination}")
            migrated_count += 1
            if args.remove_legacy:
                source.unlink(missing_ok=True)

    if args.remove_legacy:
        cleanup_legacy(legacy_root)

    print(f"Migrated {migrated_count} file(s) into {outputs_root}")


if __name__ == "__main__":
    main()
