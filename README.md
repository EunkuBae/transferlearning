# BrainAge Modeling

This repository is developed on Windows and executed on Linux GPU machines. The code should run in both places by switching config files and environment variables instead of editing source code.

## Output Policy

The canonical experiment directory is [`outputs/`](./outputs).

From now on:

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
export ADNI_METADATA_FILE=/home/$USER/modeling/ADNI_MPR_N3_metadata.csv
export ADNI_IMAGE_DIR=/data/C3_ADNI
```

The ready-to-use Ubuntu env file is:

- `configs/environment/ubuntu_data_layout.env`

Load it with:

```bash
cd /home/$USER/modeling
source configs/environment/ubuntu_data_layout.env
```

## Update On Linux

Pull the latest GitHub contents:

```bash
cd ~/modeling
git pull origin main
```

If you use conda:

```bash
conda env update -n brainage-hcp-gpu -f configs/environment/linux_gpu_hcp_mmse.yml --prune
```

After pulling transfer-learning updates, run the env update once before OASIS experiments so dependencies such as `openpyxl` are installed.

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

To run the multimodal HCP baseline instead:

```bash
bash scripts/run_hcp_mmse_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/hcp_mmse_multimodal_baseline.yaml
```

## OASIS Transfer

The current transfer-learning stage supports HCP to OASIS MMSE fine-tuning.

Example config:

- `configs/experiment/oasis_mmse_transfer.yaml`

Smoke config:

- `configs/experiment/oasis_mmse_multimodal_transfer_smoke.yaml`

Recommended transfer-learning order:

- `configs/experiment/oasis_mmse_transfer_full_ft.yaml`
- `configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`
- `configs/experiment/oasis_mmse_multimodal_transfer_full_ft.yaml`
- `configs/experiment/oasis_mmse_multimodal_transfer_freeze_backbone.yaml`

Run an OASIS transfer experiment on Ubuntu:

```bash
cd ~/modeling
source configs/environment/ubuntu_data_layout.env
bash scripts/run_oasis_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/oasis_mmse_transfer_full_ft.yaml
```

Run the four transfer experiments in order:

```bash
bash scripts/run_oasis_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/oasis_mmse_transfer_full_ft.yaml

bash scripts/run_oasis_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml

bash scripts/run_oasis_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/oasis_mmse_multimodal_transfer_full_ft.yaml

bash scripts/run_oasis_transfer_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/oasis_mmse_multimodal_transfer_freeze_backbone.yaml
```

Each transfer config writes to its own `outputs/...` directory and also records per-run execution history under `outputs/.../run_history/<timestamp>/`. The runner appends a lightweight run index to `outputs/.../run_registry.jsonl`, while the main artifacts remain `metrics.json`, `history.json`, `training_summary.txt`, and `test_predictions.csv`.

## Tailscale And Tmux

Set up the Linux server once:

```bash
bash scripts/setup_tailscale_linux.sh
```

If the server does not have `curl` yet, the script now tries to install `curl` automatically and falls back to `wget` when available. If you want to do it manually first on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y curl tmux
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

Logs are written under `outputs/.../tmux_logs/`.

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
- `configs/experiment/oasis_mmse_transfer.yaml`
- `scripts/migrate_legacy_outputs.py`
- `scripts/run_hcp_mmse_from_env.sh`
- `scripts/start_hcp_mmse_tmux.sh`
- `src/brainage/experiments/run_hcp_mmse.py`
- `src/brainage/experiments/run_oasis_transfer.py`
