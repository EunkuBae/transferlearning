# BrainAge Modeling

This repository is scoped to a regression-first study plan built around MMSE prediction from structural MRI.
The current codebase keeps only the pieces needed for HCP pretraining, OASIS same-task transfer, and optional regression generalization checks through LODO evaluation.
Raw source data are intentionally left untouched.

## Active Research Scope

The project currently supports three experiment families.

1. HCP MMSE regression pretraining
2. OASIS MMSE transfer using an HCP-pretrained checkpoint
3. LODO MMSE regression using merged HCP and ADNI metadata for holdout-style generalization checks

## Recommended Study Narrative

The most stable research direction from the current assets is:

- learn an MMSE-sensitive representation on healthy aging MRI from HCP
- test whether that representation transfers to OASIS MMSE prediction under a clean HCP-only pretraining setup
- analyze when transfer helps, when freezing hurts, and how performance changes under cohort shift

That supports paper-ready questions such as:

- Does MMSE pretraining improve same-task external transfer?
- Is full fine-tuning better than freezing the pretrained backbone?
- How large is the domain-shift gap between HCP, OASIS, and ADNI-style holdout evaluation?
- Do age and sex covariates stabilize MMSE regression across cohorts?

## Kept Experiments

### 1. HCP MMSE baseline

Primary config files:

- `configs/experiment/hcp_mmse_baseline.yaml`
- `configs/experiment/hcp_mmse_baseline_linux.yaml`

Run locally through the module entrypoint:

```bash
PYTHONPATH=src python -m brainage.experiments.run_hcp_mmse --config configs/experiment/hcp_mmse_baseline.yaml
```

Run on Linux with the helper script:

```bash
bash scripts/run_hcp_mmse_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/hcp_mmse_baseline_linux.yaml
```

### 2. OASIS MMSE transfer

Primary config files:

- `configs/experiment/oasis_mmse_transfer_full_ft.yaml`
- `configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`

Run on Linux:

```bash
bash scripts/run_oasis_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/oasis_mmse_transfer_full_ft.yaml
```

### 3. Pull and rerun helper

To update the repo on a Linux machine and rerun the main regression pipeline:

```bash
bash scripts/pull_and_rerun_hcp_oasis.sh
```

This script pulls `origin/main`, reruns the HCP baseline, and then reruns one or more OASIS transfer configs.

### 4. Optional LODO regression

Primary config file:

- `configs/experiment/lodo_mmse_adni_holdout.yaml`

Run on Linux:

```bash
bash scripts/run_lodo_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/lodo_mmse_adni_holdout.yaml
```

## Directory Guide

- `src/brainage/experiments/run_hcp_mmse.py`: HCP MMSE regression training
- `src/brainage/experiments/run_oasis_transfer.py`: OASIS MMSE transfer from HCP checkpoints
- `src/brainage/experiments/run_lodo.py`: leave-one-domain-out regression evaluation
- `src/brainage/data/hcp_mmse.py`: HCP regression dataset utilities
- `src/brainage/data/oasis_mmse.py`: OASIS regression dataset utilities
- `src/brainage/data/lodo_mmse.py`: merged-metadata regression utilities
- `scripts/run_hcp_mmse_from_env.sh`: Linux launcher for HCP baseline
- `scripts/run_oasis_transfer_from_env.sh`: Linux launcher for OASIS transfer
- `scripts/run_lodo_from_env.sh`: Linux launcher for LODO regression
- `scripts/pull_and_rerun_hcp_oasis.sh`: Linux pull-and-rerun helper

## Current Outputs To Keep

- `outputs/hcp_mmse_baseline/`
- `outputs/oasis_mmse_transfer_full_ft/`
- `outputs/oasis_mmse_transfer_freeze_backbone/`
- `outputs/lodo_mmse_adni_holdout/`

## Notes

- `data/` and root-level source metadata files are preserved as-is.
- The repository was intentionally slimmed down so future reruns and summaries stay centered on regression pretraining and same-task transfer.
