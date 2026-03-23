# BrainAge Modeling

This repository is developed on a Windows laptop with VS Code, pushed to GitHub, and executed on a Linux GPU server for large-scale experiments.

Repository SSH:

`git@github.com:EunkuBae/transferlearning.git`

## Workflow

1. Write and update code on Windows in VS Code.
2. Commit and push code to GitHub.
3. Pull the latest code on the Linux lab machine.
4. Run training and evaluation on the Linux GPU environment.
5. Save large outputs on the Linux server and only version lightweight artifacts in Git.

## Development Principle

The codebase should be platform-neutral.

- do not hardcode Windows paths like `E:\...`
- do not assume Linux-only absolute paths either
- load dataset roots, output roots, and split files from config or environment variables
- keep model code independent from local machine setup

## Repository Layout

- `configs/`: experiment, model, dataset, and environment configuration
- `data/metadata/`: lightweight CSV metadata and label mapping tables
- `data/splits/`: frozen LODO split files
- `src/brainage/`: Python source code
- `scripts/`: CLI helpers for metadata and split generation
- `outputs/`: local output root for checkpoints, metrics, figures, and predictions
- `tests/`: lightweight tests for reusable logic

## Recommended Git Strategy

Track in Git:

- source code
- YAML configs
- CSV metadata templates
- split definitions
- lightweight result summaries if needed

Do not track in Git:

- raw MRI files
- preprocessed image volumes
- large checkpoints
- attribution maps for all subjects
- large experiment outputs

Use `.gitignore` to keep heavy files out of the repository.

## Environment Variables

The same code should run on Windows and Linux by changing environment variables instead of editing source code.

Recommended variables:

- `BRAINAGE_DATA_ROOT`
- `BRAINAGE_OUTPUT_ROOT`
- `BRAINAGE_METADATA_ROOT`
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

Example Windows PowerShell:

```powershell
$env:BRAINAGE_DATA_ROOT="E:\brainage_data"
$env:BRAINAGE_OUTPUT_ROOT="E:\brainage_outputs"
$env:HCP_MMSE_CSV="E:\EwhaMediTech\research\brainage\modeling\HCP_A_id_sex_age_mmse_moca.csv"
$env:HCP_IMAGE_DIR="D:\C1_HCP\hcp_aging"
$env:HCP_MMSE_OUTPUT_DIR="E:\EwhaMediTech\research\brainage\modeling\outputs\hcp_mmse_baseline"
$env:HCP_MMSE_CACHE_DIR="E:\EwhaMediTech\research\brainage\modeling\outputs\cache\hcp_mmse_baseline"
```

Example Linux Bash:

```bash
export BRAINAGE_DATA_ROOT=/data
export BRAINAGE_OUTPUT_ROOT=/home/$USER/brainage_outputs
export HCP_MMSE_CSV=/home/$USER/modeling/HCP_A_id_sex_age_mmse_moca.csv
export HCP_IMAGE_DIR=/data/C1_HCP/hcp_aging
export HCP_MMSE_OUTPUT_DIR=/home/$USER/brainage_outputs/hcp_mmse_baseline
export HCP_MMSE_CACHE_DIR=/home/$USER/brainage_cache/hcp_mmse_baseline
export OASIS_MMSE_METADATA_FILE=/home/$USER/modeling/OASIS_MMSE.xlsx
export OASIS_IMAGE_DIR=/data/C2_OASIS
export OASIS_MMSE_OUTPUT_DIR=/home/$USER/brainage_outputs/oasis_mmse_transfer
export OASIS_MMSE_CACHE_DIR=/home/$USER/brainage_cache/oasis_mmse_transfer
export ADNI_METADATA_FILE=/home/$USER/modeling/ADNI_MPR_N3_metadata.csv
export ADNI_IMAGE_DIR=/data/C3_ADNI
```

## Typical Workflow

### Windows

```powershell
git clone git@github.com:EunkuBae/transferlearning.git
cd transferlearning
code .
```

### Linux

```bash
git clone https://github.com/EunkuBae/transferlearning.git modeling
cd modeling
```

## Ubuntu Data Layout Example

If your Ubuntu server uses this layout:

- HCP MRI: `/data/C1_HCP/hcp_aging`
- OASIS MRI: `/data/C2_OASIS`
- ADNI MRI: `/data/C3_ADNI`
- repository: `/home/$USER/modeling`

then use:

- env file: `configs/environment/ubuntu_data_layout.env`
- HCP launcher: `scripts/run_hcp_mmse_from_env.sh`

### 1. Load the environment variables

```bash
cd /home/$USER/modeling
source configs/environment/ubuntu_data_layout.env
```

### 2. Run HCP baseline training

```bash
bash scripts/run_hcp_mmse_from_env.sh
```

This defaults to:

- env file: `configs/environment/ubuntu_data_layout.env`
- config file: `configs/experiment/hcp_mmse_baseline_linux.yaml`
- conda env: `brainage-hcp-gpu`

### 3. Run a different config with the same env file

For example, the multimodal HCP baseline:

```bash
bash scripts/run_hcp_mmse_from_env.sh \
  configs/environment/ubuntu_data_layout.env \
  configs/experiment/hcp_mmse_multimodal_baseline.yaml
```

### 4. What this script does

`run_hcp_mmse_from_env.sh`:

- loads the env file
- creates output and cache directories if needed
- sets `PYTHONPATH=src`
- runs `brainage.experiments.run_hcp_mmse`

## HCP MMSE 3D-CNN Baseline

### Local smoke test

This verifies that the HCP MMSE regression pipeline runs end-to-end on a small subset.

```powershell
cd E:\EwhaMediTech\research\brainage\modeling
$env:PYTHONPATH="src"
conda run -n ml python -m brainage.experiments.run_hcp_mmse --config configs/experiment/hcp_mmse_smoke.yaml
```

Outputs are written under `outputs/hcp_mmse_smoke/`.
The run writes:

- `best_model.pt`
- `metrics.json`
- `history.json`
- `test_predictions.csv`
- `training_summary.txt`
- `resolved_paths.json`

### Linux GPU training

Use the prepared conda environment file:

```bash
conda env create -f configs/environment/linux_gpu_hcp_mmse.yml
conda activate brainage-hcp-gpu
```

Then set input and output locations.
The intended pattern is:

- input MRI and CSV on external SSD or mounted data disk
- checkpoints and logs on internal disk or home storage
- cached preprocessed tensors on internal disk or home storage

```bash
source configs/environment/ubuntu_data_layout.env
python -m brainage.experiments.run_hcp_mmse --config configs/experiment/hcp_mmse_baseline_linux.yaml
```

Or use the helper script:

```bash
bash scripts/run_hcp_mmse_from_env.sh
```

### Linux bootstrap script

For a fresh Linux GPU server, you can clone the repository, create or update the conda environment, and run the baseline with one command path.

```bash
export HCP_MMSE_CSV=/data/C1_HCP/HCP_A_id_sex_age_mmse_moca.csv
export HCP_IMAGE_DIR=/data/C1_HCP/hcp_aging
export HCP_MMSE_OUTPUT_DIR=/home/$USER/brainage_outputs/hcp_mmse_baseline
export HCP_MMSE_CACHE_DIR=/home/$USER/brainage_cache/hcp_mmse_baseline
bash scripts/bootstrap_hcp_mmse_linux.sh
```

Optional variables:

- `REPO_URL`: defaults to `https://github.com/EunkuBae/transferlearning.git`
- `REPO_DIR`: defaults to `transferlearning`
- `ENV_NAME`: defaults to `brainage-hcp-gpu`

## Tailscale And Tmux

### 1. Server-side Tailscale setup

On the Linux server:

```bash
bash scripts/setup_tailscale_linux.sh
```

This script:

- installs Tailscale if needed
- installs `tmux` on Debian or Ubuntu if needed
- runs `sudo tailscale up --ssh --accept-routes`

After that, check the server identity:

```bash
tailscale status
tailscale ip -4
```

### 2. Connect from your local machine

From your own computer, connect with Tailscale SSH:

```bash
ssh <linux-user>@<tailscale-ip>
```

or

```bash
ssh <linux-user>@<tailscale-device-name>
```

### 3. Start training in tmux

Once connected to the Linux server:

```bash
source /home/$USER/modeling/configs/environment/ubuntu_data_layout.env
bash /home/$USER/modeling/scripts/start_hcp_mmse_tmux.sh
```

This starts a tmux session named `hcp_mmse` by default, writes logs under the internal-disk output directory, and keeps training running after you disconnect.

### 4. Reattach later

```bash
bash scripts/attach_hcp_mmse_tmux.sh
```

Or directly:

```bash
tmux attach -t hcp_mmse
```

Detach without stopping training:

```text
Ctrl-b then d
```

### 5. Useful tmux checks

```bash
tmux ls
tmux capture-pane -pt hcp_mmse | tail -n 40
```

### Notes

- `device: auto` selects GPU automatically when CUDA is available.
- `mixed_precision: auto` enables AMP automatically on CUDA.
- cached tensors are stored as `.pt` files under the configured cache directory.
- the final `training_summary.txt` includes validation and test metrics such as `MAE`, `MSE`, `RMSE`, `Pearson r`, and `R2`.
- `scripts/start_hcp_mmse_tmux.sh` writes a per-run tmux log under `HCP_MMSE_OUTPUT_DIR/tmux_logs` by default.

## Current Starter Files

- `configs/environment/ubuntu_data_layout.env`
- `configs/experiment/baseline_tl.yaml`
- `configs/experiment/hcp_mmse_baseline.yaml`
- `configs/experiment/hcp_mmse_baseline_linux.yaml`
- `configs/experiment/hcp_mmse_multimodal_baseline.yaml`
- `configs/experiment/hcp_mmse_smoke.yaml`
- `configs/experiment/oasis_mmse_transfer.yaml`
- `configs/environment/linux_gpu_hcp_mmse.yml`
- `data/metadata/label_mapping.csv`
- `scripts/attach_hcp_mmse_tmux.sh`
- `scripts/bootstrap_hcp_mmse_linux.sh`
- `scripts/build_splits.py`
- `scripts/run_hcp_mmse_from_env.sh`
- `scripts/run_hcp_mmse_linux.sh`
- `scripts/setup_tailscale_linux.sh`
- `scripts/start_hcp_mmse_tmux.sh`
- `src/brainage/data/hcp_mmse.py`
- `src/brainage/data/oasis_mmse.py`
- `src/brainage/data/schemas.py`
- `src/brainage/data/split_builders.py`
- `src/brainage/experiments/run_hcp_mmse.py`
- `src/brainage/experiments/run_lodo.py`
- `src/brainage/experiments/run_oasis_transfer.py`

## Immediate Next Step

The next practical step is to pull the latest repository on Ubuntu, source `configs/environment/ubuntu_data_layout.env`, and run `bash scripts/run_hcp_mmse_from_env.sh` or start it inside tmux with `bash scripts/start_hcp_mmse_tmux.sh`.
