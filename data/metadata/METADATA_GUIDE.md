# Metadata Guide

This file defines the expected format for `merged_metadata.csv`.

## Required Columns

- `subject_id`: unique subject identifier across the merged table
- `cohort`: one of `hcp`, `oasis`, `adni`
- `image_path`: path to the processed MRI volume
- `diagnosis`: unified diagnosis label used by the model

These four columns are required by the current split builder.

## Recommended Columns

- `age`: numeric age
- `sex`: biological sex or recorded sex field
- `mmse`: numeric MMSE score if available
- `site`: scanner site, hospital, or collection center
- `scan_id`: scan-level identifier if subject has repeated scans
- `split_group`: optional manual grouping such as `pretrain` or `finetune`
- `notes`: free-text audit note

## Diagnosis Convention

Use unified labels after harmonization.

- `NC`
- `QC`
- `MCI`
- `AD`

The raw cohort-specific labels should be harmonized before this merged metadata file is finalized.

## Image Path Convention

Prefer environment-root-relative paths in planning documents and examples.

Example:

`${BRAINAGE_DATA_ROOT}/adni/sub-ADNI_0001/mri/t1_mni.nii.gz`

At runtime, the loader can later convert these into machine-specific absolute paths.

## Validation Rules in `build_splits.py`

The current split builder validates:

- required columns exist
- `subject_id` is not empty
- `subject_id` is unique
- `cohort` is one of `hcp`, `oasis`, `adni`
- `image_path` is not empty
- `diagnosis` is one of `NC`, `QC`, `MCI`, `AD`
- `mmse` is numeric when provided

Empty `mmse` is allowed, but the script prints a warning unless `--allow-missing-mmse` is used.

## Recommended Build Order

1. create cohort-specific metadata tables
2. harmonize diagnosis labels using `label_mapping.csv`
3. create `merged_metadata.csv` from `merged_metadata_template.csv`
4. validate and build splits with `scripts/build_splits.py`
5. freeze the generated split CSV files in `data/splits/`

## Convenience Command

Create a starter file:

```bash
python scripts/create_metadata_sample.py
```

Build a split file:

```bash
PYTHONPATH=src python scripts/build_splits.py   --metadata data/metadata/merged_metadata.csv   --holdout-cohort adni
```
