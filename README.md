# BrainAge Modeling

This repository is organized around a regression-first transfer and robustness study.
The active pipeline asks whether a healthy-aging MMSE-informed representation learned on HCP transfers to external same-task cognitive prediction in OASIS and ADNI.
Raw source data remain untouched.

## Study Pipeline

1. HCP MMSE pretraining
2. OASIS MMSE external transfer evaluation
3. ADNI MMSE external transfer evaluation
4. ADNI diagnosis-aware subgroup and error analysis
5. optional LODO regression for additional domain-shift analysis

## Core Question

How much value does a healthy-aging MMSE-informed representation provide for external same-task cognitive prediction, and where does that transfer become fragile under cohort shift and disease burden?

## Main Entry Points

- `python -m brainage.experiments.run_hcp_mmse --config configs/experiment/hcp_mmse_baseline.yaml`
- `python -m brainage.experiments.run_oasis_transfer --config configs/experiment/oasis_mmse_transfer_full_ft.yaml`
- `python -m brainage.experiments.run_adni_mmse_transfer --config configs/experiment/adni_mmse_transfer_full_ft.yaml`
- `python -m brainage.experiments.run_adni_diagnosis_analysis --config configs/experiment/adni_diagnosis_analysis.yaml`
- `python -m brainage.experiments.run_lodo --config configs/experiment/lodo_mmse_adni_holdout.yaml`

## Key Configs

- `configs/experiment/hcp_mmse_baseline.yaml`
- `configs/experiment/hcp_mmse_baseline_linux.yaml`
- `configs/experiment/oasis_mmse_transfer_full_ft.yaml`
- `configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`
- `configs/experiment/adni_mmse_transfer_full_ft.yaml`
- `configs/experiment/adni_mmse_transfer_freeze_backbone.yaml`
- `configs/experiment/adni_diagnosis_analysis.yaml`
- `configs/experiment/lodo_mmse_adni_holdout.yaml`

## Linux Rerun Helpers

- `scripts/run_hcp_mmse_from_env.sh`
- `scripts/run_oasis_transfer_from_env.sh`
- `scripts/run_adni_mmse_from_env.sh`
- `scripts/run_adni_diagnosis_analysis_from_env.sh`
- `scripts/pull_and_rerun_regression_suite.sh`

## Interpretation Frame

- HCP provides the healthy-aging MMSE-informed source representation.
- OASIS tests clean external same-task transfer in another aging cohort.
- ADNI tests same-task transfer in a disease-including cohort.
- ADNI diagnosis labels are used as subgroup-analysis signals rather than as a separate classification target.

## Planned Alternative

If supervised MMSE pretraining shows limited robustness, the next extension should be SSL backbone pretraining followed by the same HCP, OASIS, and ADNI regression evaluation stack.
