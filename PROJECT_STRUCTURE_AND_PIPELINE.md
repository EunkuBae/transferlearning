# Project Structure And Pipeline

## Scope

This repository is now organized around a single regression transfer study.
The active pipeline is:

1. HCP healthy-aging MMSE pretraining
2. OASIS external same-task MMSE transfer
3. ADNI external same-task MMSE transfer
4. ADNI diagnosis-aware subgroup analysis
5. optional LODO regression for extra domain-shift analysis

## Core Code

- `src/brainage/data/hcp_mmse.py`
- `src/brainage/data/oasis_mmse.py`
- `src/brainage/data/adni_mmse.py`
- `src/brainage/data/lodo_mmse.py`
- `src/brainage/experiments/run_hcp_mmse.py`
- `src/brainage/experiments/run_oasis_transfer.py`
- `src/brainage/experiments/run_adni_mmse_transfer.py`
- `src/brainage/experiments/run_adni_diagnosis_analysis.py`
- `src/brainage/experiments/run_lodo.py`
- `src/brainage/training/loops/regression.py`

## Config Layout

- `configs/experiment/hcp_mmse_baseline.yaml`
- `configs/experiment/hcp_mmse_baseline_linux.yaml`
- `configs/experiment/oasis_mmse_transfer_full_ft.yaml`
- `configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`
- `configs/experiment/adni_mmse_transfer_full_ft.yaml`
- `configs/experiment/adni_mmse_transfer_freeze_backbone.yaml`
- `configs/experiment/adni_diagnosis_analysis.yaml`
- `configs/experiment/lodo_mmse_adni_holdout.yaml`

## Linux Execution Helpers

- `scripts/run_hcp_mmse_from_env.sh`
- `scripts/run_oasis_transfer_from_env.sh`
- `scripts/run_adni_mmse_from_env.sh`
- `scripts/run_adni_diagnosis_analysis_from_env.sh`
- `scripts/pull_and_rerun_regression_suite.sh`

## Research Use

- HCP is the healthy-aging source cohort.
- OASIS is the first external same-task validation cohort.
- ADNI is the disease-including same-task validation cohort.
- ADNI diagnosis labels are used for subgroup and error interpretation.
- SSL remains the next fallback direction if supervised transfer proves too brittle.
