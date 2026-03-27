# Project Structure And Pipeline

## Scope

This repository now tracks a compact regression-first workflow.
The kept pipeline is:

1. pretrain an MMSE regressor on HCP
2. transfer that checkpoint to OASIS MMSE regression
3. optionally measure regression generalization with LODO holdout evaluation

Raw input data were preserved.

## Main Folders

### Code

- `src/brainage/experiments/run_hcp_mmse.py`
- `src/brainage/experiments/run_oasis_transfer.py`
- `src/brainage/experiments/run_lodo.py`
- `src/brainage/data/hcp_mmse.py`
- `src/brainage/data/oasis_mmse.py`
- `src/brainage/data/lodo_mmse.py`
- `src/brainage/models/factory.py`
- `src/brainage/training/loops/regression.py`

### Configs

- `configs/experiment/hcp_mmse_baseline.yaml`
- `configs/experiment/hcp_mmse_baseline_linux.yaml`
- `configs/experiment/oasis_mmse_transfer_full_ft.yaml`
- `configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`
- `configs/experiment/lodo_mmse_adni_holdout.yaml`

### Scripts

- `scripts/run_hcp_mmse_from_env.sh`
- `scripts/run_oasis_transfer_from_env.sh`
- `scripts/run_lodo_from_env.sh`
- `scripts/pull_and_rerun_hcp_oasis.sh`
- `scripts/build_merged_metadata.py`

### Outputs Kept

- `outputs/hcp_mmse_baseline/`
- `outputs/oasis_mmse_transfer_full_ft/`
- `outputs/oasis_mmse_transfer_freeze_backbone/`
- `outputs/lodo_mmse_adni_holdout/`

## Execution Order

### HCP baseline

```bash
bash scripts/run_hcp_mmse_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/hcp_mmse_baseline_linux.yaml
```

### OASIS transfer

```bash
bash scripts/run_oasis_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/oasis_mmse_transfer_full_ft.yaml
```

### Combined rerun on Linux

```bash
bash scripts/pull_and_rerun_hcp_oasis.sh
```

### Optional LODO regression

```bash
bash scripts/run_lodo_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/lodo_mmse_adni_holdout.yaml
```

## Research Framing

The cleaned repository is best suited for a study built around:

- same-task transfer of MMSE regression features
- full fine-tuning versus frozen-backbone adaptation
- domain-shift sensitivity across healthy aging and external cohorts
- whether simple demographic covariates improve generalization

## Preserved Assets

The cleanup did not remove the underlying source datasets or root metadata files.
Only derived experiment code, configs, caches, and outputs outside the current regression plan were pruned.
