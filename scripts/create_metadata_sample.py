"""Create a starter merged_metadata.csv from the template if it does not exist."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a starter merged metadata CSV.")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/metadata/merged_metadata_template.csv"),
        help="Path to metadata template CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/merged_metadata.csv"),
        help="Path to output merged metadata CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}. Use --force to overwrite.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.template, args.output)
    print(f"Created starter metadata file at {args.output}")


if __name__ == "__main__":
    main()
