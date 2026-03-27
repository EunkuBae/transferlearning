# BrainAge Modeling

This repository is developed on Windows and executed on Linux GPU machines. The active study direction is now:

- pretrain the MMSE backbone on HCP only
- evaluate same-task external transfer on OASIS
- evaluate cross-task transfer on ADNI
- measure cross-cohort robustness with LODO-style MMSE evaluation

## Current Status

Working experiment families:

- HCP MMSE baseline pretraining
- HCP multimodal MMSE baseline
- OASIS MMSE transfer from HCP checkpoints
- ADNI scratch diagnosis classification
- ADNI transfer from HCP checkpoints
- LODO MMSE evaluation
- timestamped experiment tracking and lightweight Git-friendly result saving

Current high-level conclusions:

- HCP-only MMSE pretraining provides a valid cognition-informed backbone
- OASIS same-task transfer works, but it should now be interpreted only from HCP-pretrained checkpoints
- ADNI disease classification transfer still tends to collapse in current settings
- LODO still shows a substantial external generalization gap

Useful result registries:

- [`outputs/metrics/experiment_runs.csv`](./outputs/metrics/experiment_runs.csv)
- [`outputs/metrics/experiment_runs.jsonl`](./outputs/metrics/experiment_runs.jsonl)

## Output Policy

The canonical experiment directory is [`outputs/`](./outputs).

- local Windows runs save results under `outputs/`
- Linux runs also save results under `~/modeling/outputs/`
- the old `brainage_outputs/` folder is treated as legacy
- cache files stay under `outputs/cache/`
- large binary artifacts such as `.pt` checkpoints and cached tensors stay ignored by Git
- lightweight artifacts such as `metrics.json`, `history.json`, `resolved_paths.json`, `test_predictions.csv`, and `training_summary.txt` can be pushed to GitHub

If you still have legacy results in `brainage_outputs/`, migrate them with:

```bash
python scripts/migrate_legacy_outputs.py --remove-legacy
```

## Repository Layout

- `configs/`: experiment and environment configuration
- `data/`: metadata, split files, and placeholders for raw or processed data
- `outputs/`: canonical run directory for summaries, predictions, and local caches
- `scripts/`: helper scripts for launch, setup, migration, and split generation
- `src/brainage/`: Python source code
- `tests/`: lightweight tests

## Environment Variables

Common variables:

- `BRAINAGE_DATA_ROOT`
- `BRAINAGE_OUTPUT_ROOT`
- `HCP_MMSE_CSV`
- `HCP_IMAGE_DIR`
- `HCP_MMSE_OUTPUT_DIR`
- `HCP_MMSE_CACHE_DIR`
- `OASIS_MMSE_METADATA_FILE`
- `OASIS_IMAGE_DIR`
- `OASIS_MMSE_OUTPUT_DIR`
- `OASIS_MMSE_CACHE_DIR`
- `ADNI_METADATA_FILE`
- `ADNI_IMAGE_DIR`

### Windows example

```powershell
$env:BRAINAGE_DATA_ROOT="E:\brainage_data"
$env:BRAINAGE_OUTPUT_ROOT="E:\EwhaMediTech\research\brainage\modeling\outputs"
$env:HCP_MMSE_CSV="E:\EwhaMediTech\research\brainage\modeling\HCP_A_id_sex_age_mmse_moca.csv"
$env:HCP_IMAGE_DIR="D:\C1_HCP\hcp_aging"
$env:HCP_MMSE_OUTPUT_DIR="E:\EwhaMediTech\research\brainage\modeling\outputs\hcp_mmse_baseline"
$env:HCP_MMSE_CACHE_DIR="E:\EwhaMediTech\research\brainage\modeling\outputs\cache\hcp_mmse_baseline"
```

### Ubuntu example

```bash
export BRAINAGE_DATA_ROOT=/data
export BRAINAGE_OUTPUT_ROOT=/home/$USER/modeling/outputs
export HCP_MMSE_CSV=/home/$USER/modeling/HCP_A_id_sex_age_mmse_moca.csv
export HCP_IMAGE_DIR=/data/C1_HCP/hcp_aging
export HCP_MMSE_OUTPUT_DIR=/home/$USER/modeling/outputs/hcp_mmse_baseline
export HCP_MMSE_CACHE_DIR=/home/$USER/modeling/outputs/cache/hcp_mmse_baseline
export OASIS_MMSE_METADATA_FILE=/home/$USER/modeling/OASIS_MMSE.xlsx
export OASIS_IMAGE_DIR=/data/C2_OASIS
export OASIS_MMSE_OUTPUT_DIR=/home/$USER/modeling/outputs/oasis_mmse_transfer
export OASIS_MMSE_CACHE_DIR=/home/$USER/modeling/outputs/cache/oasis_mmse_transfer
export ADNI_METADATA_FILE=/data/C3_ADNI/ADNI_MPR_N3_metadata.csv
export ADNI_IMAGE_DIR=/data/C3_ADNI/ADNI_MPR_N3_ALL
```

The ready-to-use Ubuntu env file is:

- `configs/environment/ubuntu_data_layout.env`

## Merged Metadata And LODO

The current LODO preparation step uses HCP and ADNI only. OASIS remains reserved for external same-task transfer and is intentionally excluded from the merged metadata used for LODO baseline construction.

Build the merged metadata CSV on Linux:

```bash
cd ~/modeling
source configs/environment/ubuntu_data_layout.env
PYTHONPATH=src python scripts/build_merged_metadata.py \
  --hcp-image-dir /data/C1_HCP/hcp_aging \
  --adni-image-dir /data/C3_ADNI/ADNI_MPR_N3_ALL \
  --force
```

Build an ADNI-holdout LODO split:

```bash
cd ~/modeling
PYTHONPATH=src python scripts/build_splits.py \
  --metadata data/metadata/merged_metadata.csv \
  --holdout-cohort adni \
  --output data/splits/lodo_adni_holdout.csv
```

Run the LODO MMSE regression baseline:

```bash
cd ~/modeling
source configs/environment/ubuntu_data_layout.env
bash scripts/run_lodo_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/lodo_mmse_adni_holdout.yaml
```

## HCP Baseline

### Local smoke test

```powershell
cd E:\EwhaMediTech\research\brainage\modeling
$env:PYTHONPATH="src"
conda run -n ml python -m brainage.experiments.run_hcp_mmse --config configs/experiment/hcp_mmse_smoke.yaml
```

### Ubuntu baseline run

```bash
cd ~/modeling
source configs/environment/ubuntu_data_layout.env
bash scripts/run_hcp_mmse_from_env.sh
```

This uses:

- env file: `configs/environment/ubuntu_data_layout.env`
- config file: `configs/experiment/hcp_mmse_baseline_linux.yaml`
- output dir: `~/modeling/outputs/hcp_mmse_baseline`
- cache dir: `~/modeling/outputs/cache/hcp_mmse_baseline`
- frozen split file: `data/splits/hcp_mmse_seed42.csv`

Run the multimodal HCP baseline with the exact same split:

```bash
bash scripts/run_hcp_mmse_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/hcp_mmse_multimodal_baseline_linux.yaml
```

Current reference results:

- HCP MRI-only baseline test MAE: about `0.921`
- HCP multimodal baseline test MAE: about `0.869`

## OASIS External Transfer

The OASIS stage is now treated as a strict external same-task transfer benchmark from HCP-only checkpoints.

Recommended transfer order:

- `configs/experiment/oasis_mmse_transfer_full_ft.yaml`
- `configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`
- `configs/experiment/oasis_mmse_multimodal_transfer_full_ft.yaml`
- `configs/experiment/oasis_mmse_multimodal_transfer_freeze_backbone.yaml`

Run the HCP-to-OASIS MRI-only transfer:

```bash
cd ~/modeling
source configs/environment/ubuntu_data_layout.env
bash scripts/run_oasis_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/oasis_mmse_transfer_full_ft.yaml
```

Current reference result:

- HCP-pretrained full fine-tuning on OASIS test MAE: about `1.896`
- HCP-pretrained full fine-tuning on OASIS Pearson: about `0.417`

Interpretation rule:

- keep OASIS reserved for the external transfer stage

## ADNI Classification Baseline

The ADNI classification baseline uses `ADNI_MPR_N3_ALL` and `ADNI_MPR_N3_metadata.csv` as the canonical image and metadata sources.

Linux baseline config:

- `configs/experiment/adni_cls_baseline_linux.yaml`

Run the ADNI baseline:

```bash
cd ~/modeling
source configs/environment/ubuntu_data_layout.env
bash scripts/run_adni_classification_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/adni_cls_baseline_linux.yaml
```

Current reference result:

- MRI-only scratch baseline accuracy: `0.444`
- balanced accuracy: `0.319`
- macro-F1: `0.255`

## ADNI Transfer

The ADNI transfer stage tests whether an HCP MMSE-pretrained backbone helps diagnosis classification.

Implemented variants include:

- full fine-tuning
- freeze backbone
- partial freeze
- staged unfreezing
- MMSE auxiliary tabular features
- multi-task classification plus MMSE regression

Representative configs:

- `configs/experiment/adni_cls_transfer_staged_last1_to_all.yaml`
- `configs/experiment/adni_cls_transfer_full_ft.yaml`
- `configs/experiment/adni_cls_transfer_freeze_backbone.yaml`

Run a representative ADNI transfer experiment:

```bash
cd ~/modeling
source configs/environment/ubuntu_data_layout.env
bash scripts/run_adni_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/adni_cls_transfer_staged_last1_to_all.yaml
```

Current interpretation:

- ADNI scratch baseline is still the most interpretable classifier
- HCP-pretrained transfer variants still tend to collapse to a single class
- Stage 2B remains the main unresolved modeling problem

## Tailscale And Tmux

Set up the Linux server once:

```bash
bash scripts/setup_tailscale_linux.sh
```

Start HCP training in tmux:

```bash
cd ~/modeling
source configs/environment/ubuntu_data_layout.env
bash scripts/start_hcp_mmse_tmux.sh
```

Reattach later:

```bash
bash scripts/attach_hcp_mmse_tmux.sh
```

## Git Tracking Rule

Track in Git:

- source code
- YAML configs
- CSV or XLSX metadata files that you intentionally version
- lightweight run summaries in `outputs/`

Do not track in Git:

- MRI volumes
- cached tensors
- `.pt` checkpoints
- very large run folders

## Current Starter Files

- `configs/environment/example.env`
- `configs/environment/linux_gpu_hcp_mmse.yml`
- `configs/environment/ubuntu_data_layout.env`
- `configs/experiment/hcp_mmse_baseline.yaml`
- `configs/experiment/hcp_mmse_baseline_linux.yaml`
- `configs/experiment/hcp_mmse_multimodal_baseline.yaml`
- `configs/experiment/hcp_mmse_multimodal_baseline_linux.yaml`
- `configs/experiment/oasis_mmse_transfer.yaml`
- `scripts/migrate_legacy_outputs.py`
- `scripts/run_adni_classification_from_env.sh`
- `scripts/run_hcp_mmse_from_env.sh`
- `scripts/run_oasis_transfer_from_env.sh`
- `src/brainage/experiments/run_adni_classification.py`
- `src/brainage/experiments/run_adni_transfer.py`
- `src/brainage/experiments/run_hcp_mmse.py`
- `src/brainage/experiments/run_oasis_transfer.py`
