# BrainAge Modeling Handover

## 1. Project Summary

This project studies whether a healthy-aging, MMSE-informed structural MRI representation can transfer to:

- same-task external MMSE regression
- cross-task Alzheimer's-spectrum diagnosis classification
- cross-cohort generalization settings

The intended paper logic is:

1. Train a common 3D CNN backbone on healthy-aging cohorts with MMSE regression supervision
2. Transfer that backbone to:
   - OASIS MMSE regression
   - ADNI diagnosis classification
3. Evaluate robustness with cross-cohort settings such as LODO
4. Add repeated-seed summaries and lightweight interpretability/stability analysis

Important framing:

- Pretraining is cognition-informed, not disease-supervised
- Healthy cohorts are used to learn a shared MRI representation
- Disease classification is a downstream evaluation, not the pretraining target
- DG and TL are treated as separate stages

## 2. Current Research Status

### Main message supported by current results

The codebase now supports a coherent early paper story:

- healthy multi-source MMSE pretraining is useful for same-task external MMSE transfer
- the same representation does not automatically improve ADNI disease classification
- cross-task disease transfer remains the main unresolved bottleneck

### Practical interpretation

At the moment, the strongest positive finding is:

- multi-source healthy MMSE pretraining improves OASIS MMSE transfer

The strongest negative or limiting finding is:

- ADNI diagnosis transfer repeatedly collapses to a single class despite several adaptation attempts

## 3. Codebase Map

### Core model files

- [`src/brainage/models/backbones/cnn3d.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/models/backbones/cnn3d.py)
  - shared 3D CNN backbone
  - 4 repeated Conv3D -> BatchNorm3D -> ReLU -> MaxPool3D blocks
  - global average pooling output

- [`src/brainage/models/heads/regression.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/models/heads/regression.py)
  - scalar regression head for MMSE prediction

- [`src/brainage/models/heads/classification.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/models/heads/classification.py)
  - classification head for ADNI diagnosis prediction

- [`src/brainage/models/factory.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/models/factory.py)
  - central model builder
  - currently supports:
    - MRI-only regression
    - MRI + tabular regression
    - MRI-only classification
    - MRI + tabular classification
    - multi-task ADNI classifier with shared backbone and MMSE regression head

### Data loading

- [`src/brainage/data/hcp_mmse.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/data/hcp_mmse.py)
  - HCP MMSE dataset and examples

- [`src/brainage/data/oasis_mmse.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/data/oasis_mmse.py)
  - OASIS MMSE dataset and examples

- [`src/brainage/data/adni_cls.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/data/adni_cls.py)
  - ADNI diagnosis dataset
  - supports demographics and auxiliary tabular features
  - currently returns MMSE as well, enabling ADNI multi-task fine-tuning

- [`src/brainage/data/mmse_pretraining.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/data/mmse_pretraining.py)
  - multi-source healthy MMSE pretraining dataset
  - currently used for HCP + OASIS healthy-only pretraining

### Training loops

- [`src/brainage/training/loops/classification.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/training/loops/classification.py)
  - ADNI classification training loop
  - supports:
    - class-weighted CE
    - balanced sampling
    - staged unfreezing
    - optional MMSE auxiliary loss for multi-task transfer

- [`src/brainage/training/loops/multisource_regression.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/training/loops/multisource_regression.py)
  - multi-source MMSE pretraining loop
  - supports ERM and current GroupDRO variant

### Experiment entrypoints

- [`src/brainage/experiments/run_hcp_mmse.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_hcp_mmse.py)
  - HCP MMSE baseline pretraining

- [`src/brainage/experiments/run_oasis_transfer.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_oasis_transfer.py)
  - OASIS MMSE transfer

- [`src/brainage/experiments/run_adni_classification.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_adni_classification.py)
  - ADNI scratch baseline

- [`src/brainage/experiments/run_adni_transfer.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_adni_transfer.py)
  - ADNI transfer from MMSE-pretrained checkpoints
  - supports:
    - full FT
    - freeze backbone
    - partial freeze
    - staged unfreezing
    - MMSE auxiliary tabular features
    - multi-task MMSE loss

- [`src/brainage/experiments/run_lodo.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_lodo.py)
  - LODO-style MMSE evaluation

- [`src/brainage/experiments/run_mmse_pretraining.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_mmse_pretraining.py)
  - multi-source healthy MMSE pretraining

### Tracking and aggregation

- [`src/brainage/utils/experiment_tracking.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/utils/experiment_tracking.py)
  - automatic timestamped run snapshots
  - paper-friendly summary export

- [`scripts/aggregate_adni_classification_metrics.py`](e:/EwhaMediTech/research/brainage/modeling/scripts/aggregate_adni_classification_metrics.py)
  - mean and std aggregation for ADNI classification metrics

- [`scripts/run_adni_classification_seeds.py`](e:/EwhaMediTech/research/brainage/modeling/scripts/run_adni_classification_seeds.py)
- [`scripts/run_adni_transfer_seeds.py`](e:/EwhaMediTech/research/brainage/modeling/scripts/run_adni_transfer_seeds.py)
  - repeated-seed execution helpers

- [`scripts/push_results_to_github.sh`](e:/EwhaMediTech/research/brainage/modeling/scripts/push_results_to_github.sh)
  - pushes lightweight results only

## 4. Experiment Stages and Results

### Stage 1. Single-source HCP MMSE pretraining

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

- the basic MMSE-informed backbone is valid
- demographics help slightly inside HCP

### Stage 1B. Multi-source healthy MMSE pretraining

Main files:

- [`configs/experiment/mmse_pretraining_erm.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/mmse_pretraining_erm.yaml)
- [`configs/experiment/mmse_pretraining_groupdro.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/mmse_pretraining_groupdro.yaml)

Current results:

- multi-source ERM:
  - [`outputs/mmse_pretraining_erm/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/mmse_pretraining_erm/metrics.json)
  - overall test MAE about `0.992`
  - HCP test MAE about `0.768`
  - OASIS-NC test MAE about `1.508`

- multi-source GroupDRO:
  - [`outputs/mmse_pretraining_groupdro/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/mmse_pretraining_groupdro/metrics.json)
  - overall test MAE about `1.452`
  - OASIS-NC test MAE about `2.886`

Interpretation:

- multi-source ERM is currently the best Stage 1 setting
- the current GroupDRO implementation or hyperparameter setting is not stable enough yet

### Stage 2A. OASIS same-task MMSE transfer

Main files:

- [`src/brainage/experiments/run_oasis_transfer.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_oasis_transfer.py)
- multisource configs:
  - [`configs/experiment/oasis_mmse_transfer_multisource_erm_full_ft.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/oasis_mmse_transfer_multisource_erm_full_ft.yaml)
  - [`configs/experiment/oasis_mmse_transfer_multisource_erm_freeze_backbone.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/oasis_mmse_transfer_multisource_erm_freeze_backbone.yaml)

Current results:

- HCP-pretrained full FT:
  - [`outputs/oasis_mmse_transfer_full_ft/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/oasis_mmse_transfer_full_ft/metrics.json)
  - test MAE about `1.896`
  - Pearson about `0.417`

- multi-source ERM full FT:
  - [`outputs/oasis_mmse_transfer_multisource_erm_full_ft/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/oasis_mmse_transfer_multisource_erm_full_ft/metrics.json)
  - test MAE about `1.735`
  - Pearson about `0.565`

- HCP-pretrained freeze backbone:
  - [`outputs/oasis_mmse_transfer_freeze_backbone/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/oasis_mmse_transfer_freeze_backbone/metrics.json)
  - test MAE about `2.345`

- multi-source ERM freeze backbone:
  - [`outputs/oasis_mmse_transfer_multisource_erm_freeze_backbone/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/oasis_mmse_transfer_multisource_erm_freeze_backbone/metrics.json)
  - test MAE about `1.753`
  - Pearson about `0.626`

Interpretation:

- this is the clearest positive result in the project
- healthy multi-source MMSE pretraining improves same-task external transfer

### Stage 2B. ADNI cross-task diagnosis classification

Main files:

- scratch baseline:
  - [`configs/experiment/adni_cls_baseline_linux.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/adni_cls_baseline_linux.yaml)
  - [`configs/experiment/adni_cls_multimodal_baseline_linux.yaml`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment/adni_cls_multimodal_baseline_linux.yaml)

- transfer family:
  - full FT, freeze, partial freeze, staged unfreezing, MMSE-aux, multi-task configs under [`configs/experiment/`](e:/EwhaMediTech/research/brainage/modeling/configs/experiment)

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
  - predicts all `MCI`

- staged multi-source ERM transfer:
  - [`outputs/adni_cls_transfer_multisource_erm_staged_last1_to_all/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/adni_cls_transfer_multisource_erm_staged_last1_to_all/metrics.json)
  - same collapse pattern

- MMSE auxiliary feature transfer:
  - [`outputs/adni_cls_transfer_multisource_erm_mmse_aux_staged_last1_to_all/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/adni_cls_transfer_multisource_erm_mmse_aux_staged_last1_to_all/metrics.json)
  - same collapse pattern

- multi-task transfer:
  - [`outputs/adni_cls_transfer_multisource_erm_multitask_staged_last1_to_all/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/adni_cls_transfer_multisource_erm_multitask_staged_last1_to_all/metrics.json)
  - same collapse pattern

- tuned multi-task transfer:
  - [`outputs/adni_cls_transfer_multisource_erm_multitask_staged_last1_to_all_tuned/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/adni_cls_transfer_multisource_erm_multitask_staged_last1_to_all_tuned/metrics.json)
  - `mmse_aux_loss_weight` reduced from `0.2` to `0.01`
  - same collapse pattern

Multi-seed result summary:

- baseline summary:
  - [`outputs/metrics/_tmp_adni_cls_baseline_seed_summary.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/metrics/_tmp_adni_cls_baseline_seed_summary.json)
  - balanced accuracy about `0.315 +/- 0.017`
  - macro-F1 about `0.237 +/- 0.016`

- staged transfer summary:
  - [`outputs/metrics/_tmp_adni_cls_transfer_staged_seed_summary.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/metrics/_tmp_adni_cls_transfer_staged_seed_summary.json)
  - balanced accuracy about `0.289 +/- 0.037`
  - macro-F1 about `0.229 +/- 0.027`

Interpretation:

- Stage 2B is still not successful
- the core failure mode is repeated class collapse
- changing checkpoint source alone does not fix this
- MMSE auxiliary features alone do not fix this
- reducing the multi-task MMSE loss weight also did not change the outcome
- this now looks less like a simple loss-scale bug and more like a transfer-formulation mismatch

### Stage 3. LODO and external generalization

Main file:

- [`src/brainage/experiments/run_lodo.py`](e:/EwhaMediTech/research/brainage/modeling/src/brainage/experiments/run_lodo.py)

Current result:

- [`outputs/lodo_mmse_adni_holdout/metrics.json`](e:/EwhaMediTech/research/brainage/modeling/outputs/lodo_mmse_adni_holdout/metrics.json)
- test MAE about `3.014`

Interpretation:

- external cohort shift remains substantial
- LODO currently supports the “robustness problem exists” message more than a “performance solved” message

## 5. What Is Already Ready For A Paper

The current repository is already strong enough to support:

- a reproducible Stage 1 and Stage 2A study
- a negative-but-important Stage 2B result
- repeated-seed reporting for ADNI classification
- timestamped experiment tracking and paper-friendly aggregation

Useful tracking outputs:

- [`outputs/metrics/experiment_runs.csv`](e:/EwhaMediTech/research/brainage/modeling/outputs/metrics/experiment_runs.csv)
- [`outputs/metrics/experiment_runs.jsonl`](e:/EwhaMediTech/research/brainage/modeling/outputs/metrics/experiment_runs.jsonl)

Each run also stores:

- `metrics.json`
- `history.json`
- `resolved_paths.json`
- `test_predictions.csv`
- `training_summary.txt`

## 6. Recommended Paper Framing

Recommended story:

1. healthy-aging MMSE pretraining produces a cognition-aware MRI representation
2. adding a healthy external source cohort improves that representation for same-task transfer
3. however, disease classification is not recovered automatically from that representation
4. therefore, healthy cognition-aware representation learning and disease-discriminative adaptation are related but not interchangeable

Recommended positive result emphasis:

- Stage 1 multi-source ERM
- Stage 2A OASIS transfer improvement

Recommended cautionary result emphasis:

- Stage 2B ADNI collapse despite multiple transfer strategies

This can be written as a scientifically useful limitation rather than a failed side experiment.

## 7. Recommended Tables and Figures

### Tables

- cohort summary table
- Stage 1 pretraining comparison:
  - HCP baseline
  - multi-source ERM
  - GroupDRO
- Stage 2A transfer comparison:
  - HCP-pretrained vs multi-source ERM-pretrained
  - full FT vs freeze backbone
- Stage 2B transfer comparison:
  - scratch baseline
  - HCP-pretrained staged transfer
  - multi-source ERM staged transfer
  - MMSE auxiliary transfer
  - multi-task transfer
- LODO summary table

### Figures

- overall study design figure
- model architecture figure
- OASIS transfer comparison bar plot
- ADNI confusion matrix comparison:
  - scratch baseline
  - staged transfer
  - multi-task transfer
- optional seed summary plot for ADNI

## 8. Immediate Next Steps

Recommended next priorities:

1. stabilize Stage 2B multi-task loss
   - this was attempted and did not change the collapse outcome
   - further tuning of the same path is now lower priority

2. shift Stage 2B effort toward a stronger ADNI classification formulation
   - stronger classifier head
   - alternative loss such as focal loss
   - possibly simpler pairwise tasks such as `MCI vs AD`

3. improve DG only after a stronger ADNI transfer bridge is available
   - current GroupDRO is not yet publishable as a positive result

4. add reporting scripts for direct table and figure export

5. add lightweight explainability only after the main result tables are fixed

## 9. Operational Notes

- The project is usually edited on Windows and trained on Linux
- Use the Linux wrapper scripts under [`scripts/`](e:/EwhaMediTech/research/brainage/modeling/scripts) for stable run tracking
- Use [`scripts/push_results_to_github.sh`](e:/EwhaMediTech/research/brainage/modeling/scripts/push_results_to_github.sh) to push lightweight results after training
- There is often an unrelated local modification in [`scripts/setup_tailscale_linux.sh`](e:/EwhaMediTech/research/brainage/modeling/scripts/setup_tailscale_linux.sh); do not include it accidentally when committing research code

## 10. Bottom Line

If a new collaborator joins now, the safest summary is:

- Stage 1 single-source and multi-source MMSE pretraining are implemented
- Stage 2A OASIS MMSE transfer is working and improved by multi-source ERM
- Stage 2B ADNI diagnosis transfer is implemented in many variants and still collapses, including tuned multi-task
- the paper is currently strongest as a cognition-transfer success story plus a disease-transfer limitation story
