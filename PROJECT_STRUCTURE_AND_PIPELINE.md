# BrainAge Modeling Project Structure And Experiment Pipeline

## 1. Goal

This project supports a study built around:

- HCP MMSE pretraining
- OASIS external MMSE transfer
- ADNI diagnosis classification
- LODO external validation
- repeated-seed reporting

## 2. Current Paper Framing

The active staged question is:

1. learn an MMSE-informed backbone from HCP structural MRI
2. test same-task transfer on external OASIS MMSE regression
3. test cross-task transfer on ADNI diagnosis classification
4. quantify cross-cohort robustness with LODO-style MMSE evaluation
5. add lightweight reporting and interpretation later

Important interpretation:

- HCP is the only Stage 1 pretraining cohort
- OASIS is reserved for external transfer evaluation
- ADNI is a downstream cross-task target

## 3. Current Implementation Snapshot

The repository currently contains working versions of:

- HCP MMSE baseline training
- HCP multimodal MMSE baseline
- OASIS MMSE transfer from HCP checkpoints
- ADNI scratch classification
- ADNI transfer from HCP checkpoints
- LODO MMSE evaluation
- repeated-seed utilities for ADNI classification
- timestamped experiment tracking and run aggregation

Current strongest findings:

- HCP MMSE pretraining yields a usable cognition-informed representation
- OASIS same-task transfer works under the HCP-only protocol
- ADNI transfer remains unstable and often collapses to a single class
- LODO still shows a large external generalization gap

## 4. Recommended Project Structure

```text
modeling/
|- README.md
|- PROJECT_STRUCTURE_AND_PIPELINE.md
|- handover.md
|- configs/
|  |- environment/
|  \- experiment/
|- data/
|  |- metadata/
|  \- splits/
|- outputs/
|- scripts/
|- src/brainage/
|  |- data/
|  |- experiments/
|  |- models/
|  |- training/
|  \- utils/
\- tests/
```

## 5. Folder Responsibilities

### `configs/`

- experiment settings
- machine-specific paths
- checkpoint choices
- output destinations

### `data/`

- metadata tables
- split manifests
- cohort harmonization assets

### `src/brainage/data/`

- cohort-specific loaders for HCP, OASIS, ADNI, and LODO

### `src/brainage/models/`

- shared 3D CNN backbone
- regression and classification heads
- multi-task ADNI head support

### `src/brainage/training/`

- regression and classification loops
- evaluation logic

### `src/brainage/experiments/`

- `run_hcp_mmse.py`
- `run_oasis_transfer.py`
- `run_adni_classification.py`
- `run_adni_transfer.py`
- `run_lodo.py`

## 6. Recommended Experiment Pipeline

### Phase 1. Dataset audit and metadata standardization

Goal:
make HCP, OASIS, and ADNI comparable enough for fair transfer and LODO analysis.

Outputs:

- `data/metadata/*.csv`
- `data/metadata/label_mapping.csv`
- `data/metadata/merged_metadata.csv`

### Phase 2. Frozen split generation

Goal:
make experiments reproducible.

Rules:

- keep HCP splits frozen for pretraining comparisons
- keep ADNI splits frozen for classification comparisons
- do not introduce OASIS into Stage 1 pretraining if OASIS is the external transfer target

Outputs:

- `data/splits/hcp_mmse_seed42.csv`
- `data/splits/adni_cls_seed42.csv`
- `data/splits/lodo_adni_holdout.csv`

### Phase 3. HCP MMSE pretraining

Goal:
learn the cognition-informed backbone.

Primary outputs:

- `outputs/hcp_mmse_baseline/`
- `outputs/hcp_mmse_multimodal_baseline/`

### Phase 4. OASIS external transfer

Goal:
test whether the HCP MMSE representation transfers to an external same-task cohort.

Primary outputs:

- `outputs/oasis_mmse_transfer_full_ft/`
- `outputs/oasis_mmse_transfer_freeze_backbone/`
- optional multimodal transfer outputs

### Phase 5. ADNI classification and transfer

Goal:
test whether the HCP MMSE representation helps disease classification.

Primary outputs:

- `outputs/adni_cls_baseline/`
- `outputs/adni_cls_transfer_staged_last1_to_all/`
- related HCP-based transfer ablations

### Phase 6. LODO evaluation

Goal:
measure cross-cohort robustness gaps.

Primary outputs:

- `outputs/lodo_mmse_adni_holdout/`

## 7. Current Result Summary

### Stage 1

- HCP MMSE baseline is working
- HCP multimodal MMSE baseline is slightly better than MRI-only

### Stage 2A

- HCP-to-OASIS transfer works under the current HCP-only protocol
- reference MRI-only full fine-tuning result: test MAE about `1.896`

### Stage 2B

- ADNI scratch baseline is weak but still more interpretable than current transfer collapse
- HCP-based transfer variants have not solved the class-collapse issue yet

### Stage 3

- LODO shows a substantial external generalization gap

## 8. Minimal Viable Next Steps

1. rerun the HCP baseline checkpoint if you want a clean post-pruning reference
2. rerun OASIS transfer from the retained HCP checkpoint only
3. regenerate summary tables from the cleaned experiment set
4. continue ADNI work with stronger classifier formulations rather than more source-cohort variants

## 9. Design Principles

- keep external evaluation cohorts out of pretraining when they are later used as targets
- save predictions, not just summary metrics
- freeze split logic in files when comparisons must stay stable
- keep documentation aligned with the currently valid experimental story

## 10. Practical Manuscript Implication

The strongest current manuscript story is:

- a successful HCP MMSE pretraining and external OASIS transfer study
- plus a clear limitation result showing that disease-classification transfer is not automatic
