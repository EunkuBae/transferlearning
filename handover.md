# BrainAge Modeling Handover

## 1. Project Summary

This project studies whether an HCP-trained, MMSE-informed structural MRI backbone can transfer to:

- external same-task MMSE regression on OASIS
- cross-task Alzheimer's-spectrum diagnosis classification on ADNI
- cross-cohort robustness settings such as LODO

The active paper logic is now:

1. Train a common 3D CNN backbone on HCP MMSE regression only
2. Transfer that backbone to:
   - OASIS MMSE regression
   - ADNI diagnosis classification
3. Evaluate robustness with LODO-style MMSE settings
4. Add repeated-seed summaries and lightweight analysis later

Important framing:

- pretraining is cognition-informed, not disease-supervised
- OASIS is reserved for external transfer evaluation and does not appear in Stage 1 pretraining

## 2. Current Research Status

### Main message supported by current results

The repository now supports a cleaner and safer study story:

- HCP MMSE pretraining learns a usable cognition-informed MRI representation
- that representation transfers to OASIS MMSE regression with moderate success
- the same representation does not automatically solve ADNI disease classification
- cross-task disease transfer remains the main unresolved bottleneck

### Practical interpretation

Strongest positive finding:

- HCP-only pretraining supports external OASIS MMSE transfer

Strongest limiting finding:

- ADNI diagnosis transfer still collapses or underperforms the scratch baseline in current settings

## 3. Codebase Map

### Core model files

- [`src/brainage/models/backbones/cnn3d.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/models/backbones/cnn3d.py)
- [`src/brainage/models/heads/regression.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/models/heads/regression.py)
- [`src/brainage/models/heads/classification.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/models/heads/classification.py)
- [`src/brainage/models/factory.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/models/factory.py)

### Data loading

- [`src/brainage/data/hcp_mmse.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/data/hcp_mmse.py)
- [`src/brainage/data/oasis_mmse.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/data/oasis_mmse.py)
- [`src/brainage/data/adni_cls.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/data/adni_cls.py)
- [`src/brainage/data/lodo_mmse.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/data/lodo_mmse.py)

### Training loops

- [`src/brainage/training/loops/regression.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/training/loops/regression.py)
- [`src/brainage/training/loops/classification.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/training/loops/classification.py)

### Experiment entrypoints

- [`src/brainage/experiments/run_hcp_mmse.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_hcp_mmse.py)
- [`src/brainage/experiments/run_oasis_transfer.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_oasis_transfer.py)
- [`src/brainage/experiments/run_adni_classification.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_adni_classification.py)
- [`src/brainage/experiments/run_adni_transfer.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_adni_transfer.py)
- [`src/brainage/experiments/run_lodo.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_lodo.py)

### Tracking and aggregation

- [`src/brainage/utils/experiment_tracking.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/utils/experiment_tracking.py)
- [`scripts/aggregate_adni_classification_metrics.py`](e:/EwhaMediTech/research/brainage/modeling/scripts/aggregate_adni_classification_metrics.py)
- [`scripts/run_adni_classification_seeds.py`](e:/EwhaMediTech/research/brainage/modeling/scripts/run_adni_classification_seeds.py)
- [`scripts/run_adni_transfer_seeds.py`](e:/EwhaMediTech/research/brainage/modeling/scripts/run_adni_transfer_seeds.py)

## 4. Experiment Stages And Results

### Stage 1. HCP MMSE pretraining

Main files:

- [`configs/experiment/hcp_mmse_baseline_linux.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/hcp_mmse_baseline_linux.yaml)
- [`configs/experiment/hcp_mmse_multimodal_baseline_linux.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/hcp_mmse_multimodal_baseline_linux.yaml)

Current results:

- HCP MRI-only baseline:
  - [`outputs/hcp_mmse_baseline/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/hcp_mmse_baseline/metrics.json)
  - test MAE about `0.921`

- HCP multimodal baseline:
  - [`outputs/hcp_mmse_multimodal_baseline/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/hcp_mmse_multimodal_baseline/metrics.json)
  - test MAE about `0.869`

Interpretation:

- the MMSE-informed backbone is valid inside HCP
- demographics help slightly inside HCP

### Stage 2A. OASIS external MMSE transfer

Main files:

- [`configs/experiment/oasis_mmse_transfer_full_ft.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/oasis_mmse_transfer_full_ft.yaml)
- [`configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/oasis_mmse_transfer_freeze_backbone.yaml)
- [`src/brainage/experiments/run_oasis_transfer.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_oasis_transfer.py)

Current results:

- HCP-pretrained full fine-tuning:
  - [`outputs/oasis_mmse_transfer_full_ft/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/oasis_mmse_transfer_full_ft/metrics.json)
  - test MAE about `1.896`
  - Pearson about `0.417`

- HCP-pretrained freeze backbone:
  - [`outputs/oasis_mmse_transfer_freeze_backbone/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/oasis_mmse_transfer_freeze_backbone/metrics.json)
  - test MAE about `2.345`

Interpretation:

- this stage remains a valid positive result under the corrected HCP-only protocol
- all OASIS claims should now be framed only from HCP-pretrained checkpoints

### Stage 2B. ADNI cross-task diagnosis classification

Current results:

- scratch MRI-only baseline:
  - [`outputs/adni_cls_baseline/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/adni_cls_baseline/metrics.json)
  - accuracy `0.444`
  - balanced accuracy `0.319`
  - macro-F1 `0.255`

- staged HCP-pretrained transfer:
  - [`outputs/adni_cls_transfer_staged_last1_to_all/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/adni_cls_transfer_staged_last1_to_all/metrics.json)
  - accuracy `0.481`
  - balanced accuracy `0.333`
  - macro-F1 `0.217`
  - predicts all `MCI` on the reference split

Multi-seed result summary:

- scratch baseline across seeds 42 to 44:
  - balanced accuracy about `0.315 +/- 0.017`
  - macro-F1 about `0.237 +/- 0.016`

- staged HCP transfer across seeds 42 to 44:
  - balanced accuracy about `0.289 +/- 0.037`
  - macro-F1 about `0.229 +/- 0.027`

Interpretation:

- Stage 2B is still not successful
- the core failure mode is repeated class collapse
- HCP MMSE pretraining alone does not recover disease-discriminative features reliably

### Stage 3. LODO and external generalization

Current result:

- [`outputs/lodo_mmse_adni_holdout/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/lodo_mmse_adni_holdout/metrics.json)
- test MAE about `3.014`

Interpretation:

- external cohort shift remains substantial
- LODO currently supports a limitation story more than a solved robustness story

## 5. What Is Ready For A Paper

The current repository can support:

- a reproducible HCP pretraining baseline
- an OASIS external transfer result from HCP-only pretraining
- a negative-but-important ADNI transfer result
- repeated-seed reporting for ADNI classification
- timestamped experiment tracking and lightweight aggregation

## 6. Recommended Paper Framing

Recommended story:

1. HCP MMSE pretraining produces a cognition-aware MRI representation
2. that representation transfers to an external same-task OASIS MMSE setting
3. however, disease classification is not automatically recovered on ADNI
4. therefore, cognition-aware pretraining and disease-discriminative adaptation are related but not interchangeable

Recommended positive emphasis:

- Stage 1 HCP MMSE pretraining
- Stage 2A HCP-to-OASIS external transfer

Recommended cautionary emphasis:

- Stage 2B ADNI collapse despite multiple HCP-based transfer strategies

## 7. Immediate Next Steps

Recommended next priorities:

1. rerun and lock the HCP baseline checkpoint if you want a fresh post-cleanup reference
2. rerun OASIS transfer from the HCP checkpoint only and treat that as the canonical external result
3. keep ADNI work focused on stronger classification formulations rather than more checkpoint-source variants
4. regenerate tables and figures only from the retained HCP-only experiment set

## 8. Bottom Line

If a new collaborator joins now, the safest summary is:

- Stage 1 HCP MMSE pretraining is implemented and working
- Stage 2A OASIS MMSE transfer is retained only under HCP-only pretraining
- Stage 2B ADNI diagnosis transfer remains unresolved
- the paper is now strongest as a cognition-transfer study plus a disease-transfer limitation study
