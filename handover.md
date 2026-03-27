# Handover

## Current Focus

The codebase now follows a single research storyline:
HCP healthy-aging MMSE pretraining, OASIS external same-task transfer, ADNI external same-task transfer, and diagnosis-aware analysis of failure patterns.
Raw source data were preserved during cleanup and reorganization.

## Active Experiments

### Stage 1. HCP MMSE pretraining
- entrypoint: `src/brainage/experiments/run_hcp_mmse.py`
- configs:
  - `configs/experiment/hcp_mmse_baseline.yaml`
  - `configs/experiment/hcp_mmse_baseline_linux.yaml`

### Stage 2. OASIS MMSE transfer
- entrypoint: `src/brainage/experiments/run_oasis_transfer.py`
- configs:
  - `configs/experiment/oasis_mmse_transfer_full_ft.yaml`
  - `configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`

### Stage 3. ADNI MMSE transfer
- entrypoint: `src/brainage/experiments/run_adni_mmse_transfer.py`
- configs:
  - `configs/experiment/adni_mmse_transfer_full_ft.yaml`
  - `configs/experiment/adni_mmse_transfer_freeze_backbone.yaml`

### Stage 4. ADNI diagnosis-aware analysis
- entrypoint: `src/brainage/experiments/run_adni_diagnosis_analysis.py`
- config:
  - `configs/experiment/adni_diagnosis_analysis.yaml`

### Stage 5. Optional LODO regression
- entrypoint: `src/brainage/experiments/run_lodo.py`
- config:
  - `configs/experiment/lodo_mmse_adni_holdout.yaml`

## Recommended Narrative

1. Learn a healthy-aging MMSE-informed representation on HCP.
2. Test whether it improves external same-task transfer on OASIS.
3. Test whether that same representation still carries cognitive signal in ADNI.
4. Use ADNI diagnosis labels to interpret subgroup error patterns instead of launching a separate disease-classification project.

## Main Scripts

- `scripts/run_hcp_mmse_from_env.sh`
- `scripts/run_oasis_transfer_from_env.sh`
- `scripts/run_adni_mmse_from_env.sh`
- `scripts/run_adni_diagnosis_analysis_from_env.sh`
- `scripts/pull_and_rerun_regression_suite.sh`

## Next Alternative

If backbone transfer from supervised HCP MMSE pretraining remains weak under external shift, the next planned extension is SSL pretraining followed by the same OASIS and ADNI MMSE evaluation stack.
