# Handover

## Current Direction

The repository is now organized around a regression-only workflow.
The active question is whether HCP MMSE pretraining learns a representation that transfers to downstream MMSE prediction in other cohorts, especially OASIS.
Raw source data were not deleted during cleanup.

## Active Pipeline

### Stage 1. HCP MMSE pretraining

- entrypoint: `src/brainage/experiments/run_hcp_mmse.py`
- primary configs:
  - `configs/experiment/hcp_mmse_baseline.yaml`
  - `configs/experiment/hcp_mmse_baseline_linux.yaml`
- main launcher: `scripts/run_hcp_mmse_from_env.sh`
- canonical output: `outputs/hcp_mmse_baseline/`

### Stage 2. OASIS MMSE transfer

- entrypoint: `src/brainage/experiments/run_oasis_transfer.py`
- primary configs:
  - `configs/experiment/oasis_mmse_transfer_full_ft.yaml`
  - `configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`
- main launcher: `scripts/run_oasis_transfer_from_env.sh`
- canonical outputs:
  - `outputs/oasis_mmse_transfer_full_ft/`
  - `outputs/oasis_mmse_transfer_freeze_backbone/`

### Stage 3. Optional LODO regression

- entrypoint: `src/brainage/experiments/run_lodo.py`
- primary config: `configs/experiment/lodo_mmse_adni_holdout.yaml`
- main launcher: `scripts/run_lodo_from_env.sh`
- canonical output: `outputs/lodo_mmse_adni_holdout/`

## Suggested Near-Term Study Questions

1. Compare HCP baseline versus HCP-pretrained OASIS transfer under full fine-tuning.
2. Measure whether freezing the backbone meaningfully harms OASIS adaptation.
3. Quantify external regression degradation with LODO holdout evaluation.
4. Build a paper narrative around same-task transfer and domain shift.

## Minimal Working Commands

```bash
bash scripts/pull_and_rerun_hcp_oasis.sh
```

```bash
bash scripts/run_lodo_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/lodo_mmse_adni_holdout.yaml
```

## Cleanup Boundary

The cleanup targeted only repository code, configs, and generated outputs.
These locations were deliberately preserved:

- `data/`
- root-level source metadata files such as `HCP_A_id_sex_age_mmse_moca.csv`, `ADNI_MPR_N3_metadata.csv`, and `OASIS_MMSE.xlsx`
- unrelated local edits such as `scripts/setup_tailscale_linux.sh`
