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
export BRAINAGE_DATA_ROOT=/data/brainage
export BRAINAGE_OUTPUT_ROOT=/data/brainage_outputs
export HCP_MMSE_CSV=/mnt/external_ssd/HCP_A_id_sex_age_mmse_moca.csv
export HCP_IMAGE_DIR=/mnt/external_ssd/hcp_aging
export HCP_MMSE_OUTPUT_DIR=/data/internal_disk/brainage_outputs/hcp_mmse_baseline
export HCP_MMSE_CACHE_DIR=/data/internal_disk/brainage_cache/hcp_mmse_baseline
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
git clone git@github.com:EunkuBae/transferlearning.git
cd transferlearning
python -m brainage.experiments.run_lodo --config configs/experiment/baseline_tl.yaml
```

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

- input MRI and CSV on external SSD
- checkpoints and logs on internal disk
- cached preprocessed tensors on internal disk

```bash
export PYTHONPATH=src
export HCP_MMSE_CSV=/mnt/external_ssd/HCP_A_id_sex_age_mmse_moca.csv
export HCP_IMAGE_DIR=/mnt/external_ssd/hcp_aging
export HCP_MMSE_OUTPUT_DIR=/data/internal_disk/brainage_outputs/hcp_mmse_baseline
export HCP_MMSE_CACHE_DIR=/data/internal_disk/brainage_cache/hcp_mmse_baseline
python -m brainage.experiments.run_hcp_mmse --config configs/experiment/hcp_mmse_baseline_linux.yaml
```

Or use the helper script:

```bash
bash scripts/run_hcp_mmse_linux.sh
```

### Notes

- `device: auto` selects GPU automatically when CUDA is available.
- `mixed_precision: auto` enables AMP automatically on CUDA.
- cached tensors are stored as `.pt` files under the configured cache directory.
- the final `training_summary.txt` includes validation and test metrics such as `MAE`, `MSE`, `RMSE`, `Pearson r`, and `R2`.

## Current Starter Files

- `configs/experiment/baseline_tl.yaml`
- `configs/experiment/hcp_mmse_baseline.yaml`
- `configs/experiment/hcp_mmse_baseline_linux.yaml`
- `configs/experiment/hcp_mmse_smoke.yaml`
- `configs/environment/linux_gpu_hcp_mmse.yml`
- `data/metadata/label_mapping.csv`
- `scripts/build_splits.py`
- `scripts/run_hcp_mmse_linux.sh`
- `src/brainage/data/schemas.py`
- `src/brainage/data/split_builders.py`
- `src/brainage/data/hcp_mmse.py`
- `src/brainage/experiments/run_lodo.py`
- `src/brainage/experiments/run_hcp_mmse.py`

## Immediate Next Step

The next practical step is to run the full HCP MMSE baseline on the Linux GPU machine and review `training_summary.txt`, `metrics.json`, and `test_predictions.csv` from the internal-disk output directory.
