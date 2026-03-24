"""Build merged metadata for LODO experiments from HCP and ADNI sources."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path, PureWindowsPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged metadata CSV for LODO experiments.")
    parser.add_argument("--hcp-csv", type=Path, default=Path("HCP_A_id_sex_age_mmse_moca.csv"))
    parser.add_argument("--hcp-image-dir", type=Path, required=True)
    parser.add_argument("--adni-metadata", type=Path, default=Path("ADNI_MPR_N3_metadata.csv"))
    parser.add_argument("--adni-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/metadata/merged_metadata.csv"))
    parser.add_argument("--data-root-token", type=str, default="${BRAINAGE_DATA_ROOT}")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def extract_hcp_subject_id(filename: str) -> str:
    return filename.split("_", 1)[0].strip()


def normalize_adni_diagnosis(raw_label: str) -> str | None:
    label = str(raw_label or "").strip().upper()
    if label == "CN":
        return "NC"
    if label in {"MCI", "LMCI", "EMCI"}:
        return "MCI"
    if label == "AD":
        return "AD"
    return None


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return float(stripped)


def normalize_sex(raw_value: str | None) -> str | None:
    value = str(raw_value or "").strip().upper()
    if value in {"M", "MALE"}:
        return "M"
    if value in {"F", "FEMALE"}:
        return "F"
    return None


def basename_any_path(raw_value: str | None) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    win_name = PureWindowsPath(value).name
    posix_name = Path(value).name
    return win_name if len(win_name) < len(posix_name) else posix_name


def to_env_relative(path: Path, anchor_dir: Path, data_root_token: str) -> str:
    relative = path.resolve().relative_to(anchor_dir.resolve())
    return f"{data_root_token}/{relative.as_posix()}"


def iter_image_files(image_dir: Path) -> list[Path]:
    paths = list(image_dir.rglob("*.nii"))
    paths.extend(image_dir.rglob("*.nii.gz"))
    return sorted(paths)


def build_hcp_rows(hcp_csv: Path, hcp_image_dir: Path, data_root_token: str) -> list[dict[str, str]]:
    image_lookup = {
        extract_hcp_subject_id(path.name): path
        for path in iter_image_files(hcp_image_dir)
    }
    rows: list[dict[str, str]] = []
    seen_subject_ids: set[str] = set()
    with hcp_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(row.get("subject_id", "")).strip()
            if not subject_id or subject_id in seen_subject_ids:
                continue
            image_path = image_lookup.get(subject_id)
            if image_path is None:
                continue
            seen_subject_ids.add(subject_id)
            rows.append(
                {
                    "subject_id": subject_id,
                    "cohort": "hcp",
                    "image_path": to_env_relative(image_path, hcp_image_dir.parents[1], data_root_token),
                    "diagnosis": "NC",
                    "age": str(parse_optional_float(row.get("age")) or ""),
                    "sex": normalize_sex(row.get("sex")) or "",
                    "mmse": str(parse_optional_float(row.get("mmse")) or ""),
                    "site": "",
                    "scan_id": f"{subject_id}_T1",
                    "split_group": "source",
                    "notes": "hcp source cohort for lodo baseline",
                }
            )
    return rows


def build_adni_rows(adni_metadata: Path, adni_image_dir: Path, data_root_token: str) -> list[dict[str, str]]:
    image_lookup = {path.name: path for path in iter_image_files(adni_image_dir)}
    rows: list[dict[str, str]] = []
    seen_subject_ids: set[str] = set()
    with adni_metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(row.get("PTID", "")).strip()
            diagnosis = normalize_adni_diagnosis(row.get("DX_bl", ""))
            if not subject_id or diagnosis is None or subject_id in seen_subject_ids:
                continue

            candidate_image_names: list[str] = []
            for key in ("copied_file", "file_name", "source_file"):
                image_name = basename_any_path(row.get(key, ""))
                if image_name:
                    candidate_image_names.append(image_name)
            if not candidate_image_names:
                continue

            image_path = None
            matched_name = ""
            for candidate_name in candidate_image_names:
                image_path = image_lookup.get(candidate_name)
                if image_path is not None:
                    matched_name = candidate_name
                    break
            if image_path is None:
                continue

            seen_subject_ids.add(subject_id)
            rows.append(
                {
                    "subject_id": subject_id,
                    "cohort": "adni",
                    "image_path": to_env_relative(image_path, adni_image_dir.parents[1], data_root_token),
                    "diagnosis": diagnosis,
                    "age": str(parse_optional_float(row.get("AGE")) or ""),
                    "sex": normalize_sex(row.get("PTGENDER")) or "",
                    "mmse": str(parse_optional_float(row.get("MMSE")) or ""),
                    "site": "",
                    "scan_id": matched_name,
                    "split_group": "target",
                    "notes": "adni target cohort for lodo baseline",
                }
            )
    return rows


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "subject_id",
        "cohort",
        "image_path",
        "diagnosis",
        "age",
        "sex",
        "mmse",
        "site",
        "scan_id",
        "split_group",
        "notes",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}. Use --force to overwrite.")

    hcp_rows = build_hcp_rows(args.hcp_csv, args.hcp_image_dir, args.data_root_token)
    adni_rows = build_adni_rows(args.adni_metadata, args.adni_image_dir, args.data_root_token)
    rows = sorted(hcp_rows + adni_rows, key=lambda row: (row["cohort"], row["subject_id"]))
    if not rows:
        raise ValueError("No merged metadata rows were created.")

    write_rows(rows, args.output)
    print(f"Saved {len(rows)} merged metadata rows to {args.output}")
    print(f"  HCP rows: {len(hcp_rows)}")
    print(f"  ADNI rows: {len(adni_rows)}")
    print("  OASIS rows: 0 (excluded from LODO/DG metadata in the current phase)")


if __name__ == "__main__":
    main()
