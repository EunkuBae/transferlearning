# BrainAge Modeling Project Structure and Experiment Pipeline

## 1. Goal

This project is designed to support the MIA submission strategy:

- structural MRI-based transfer learning
- dementia stage classification
- MMSE score prediction
- LODO external validation
- domain generalization
- transferability analysis
- explainability stability analysis

## 1B. Current Paper Framing

The project is now organized around the following staged question:

1. learn a healthy-aging, MMSE-informed backbone from structural MRI
2. test same-task transfer to OASIS MMSE regression
3. test cross-task transfer to ADNI diagnosis classification
4. quantify cross-cohort robustness with LODO-style evaluation
5. add repeated-seed summaries and lightweight explainability later

Important interpretation:

- healthy cohorts are used for cognition-informed pretraining
- this is not disease-supervised pretraining
- DG belongs to the pretraining stage
- TL belongs to downstream task adaptation

## 1C. Current Implementation Snapshot

The repository now already contains working versions of:

- HCP MMSE baseline training
- OASIS MMSE transfer
- ADNI scratch classification
- ADNI transfer variants
- LODO MMSE evaluation
- multi-source healthy MMSE pretraining with ERM and a first GroupDRO variant
- repeated-seed utilities for ADNI classification
- timestamped experiment tracking and run aggregation

The current strongest empirical findings are:

- multi-source ERM helps OASIS same-task transfer
- ADNI transfer remains unstable and often collapses to a single class
- current GroupDRO is not yet a positive result

The design below assumes the first implementation target is:

- baseline: `TL`
- first DG variant: `DANN-lite` or `GroupDRO`
- evaluation: `LODO`
- interpretation: `occlusion`

## 1A. Operating Workflow

This project is expected to run in a hybrid environment:

- Windows laptop:
  - VS Code development
  - config editing
  - lightweight testing
  - Git commit and push
- GitHub:
  - source of truth for code and configs
- Linux lab machine:
  - Git pull or clone
  - GPU training
  - large-scale LODO, DG, and seed experiments
  - heavy outputs stored locally on the server

Design implication:

- code must not depend on Windows-only paths
- dataset and output roots should come from config or environment variables
- large data and checkpoints should stay out of Git

---

## 2. Recommended Project Structure

```text
modeling/
├─ README.md
├─ PROJECT_STRUCTURE_AND_PIPELINE.md
├─ pyproject.toml
├─ configs/
│  ├─ data/
│  │  ├─ hcp.yaml
│  │  ├─ oasis.yaml
│  │  ├─ adni.yaml
│  │  └─ common_preprocessing.yaml
│  ├─ environment/
│  │  └─ example.env
│  ├─ experiment/
│  │  ├─ lodo_hcp_holdout.yaml
│  │  ├─ lodo_oasis_holdout.yaml
│  │  ├─ lodo_adni_holdout.yaml
│  │  ├─ baseline_tl.yaml
│  │  ├─ tl_dann.yaml
│  │  ├─ tl_groupdro.yaml
│  │  ├─ scratch.yaml
│  │  └─ explainability.yaml
│  └─ model/
│     ├─ backbone_3dcnn.yaml
│     ├─ classifier_head.yaml
│     └─ multitask_head.yaml
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  ├─ splits/
│  └─ metadata/
├─ src/
│  ├─ brainage/
│  │  ├─ __init__.py
│  │  ├─ constants.py
│  │  ├─ paths.py
│  │  ├─ utils/
│  │  │  ├─ io.py
│  │  │  ├─ seed.py
│  │  │  ├─ logging.py
│  │  │  └─ metrics.py
│  │  ├─ data/
│  │  │  ├─ schemas.py
│  │  │  ├─ metadata.py
│  │  │  ├─ transforms.py
│  │  │  ├─ datasets.py
│  │  │  ├─ samplers.py
│  │  │  └─ split_builders.py
│  │  ├─ models/
│  │  │  ├─ backbones/
│  │  │  │  └─ cnn3d.py
│  │  │  ├─ heads/
│  │  │  │  ├─ classification.py
│  │  │  │  ├─ regression.py
│  │  │  │  └─ multitask.py
│  │  │  ├─ domain/
│  │  │  │  ├─ grl.py
│  │  │  │  ├─ domain_classifier.py
│  │  │  │  └─ losses.py
│  │  │  └─ factory.py
│  │  ├─ training/
│  │  │  ├─ loops/
│  │  │  │  ├─ pretrain.py
│  │  │  │  ├─ finetune.py
│  │  │  │  ├─ multitask.py
│  │  │  │  └─ evaluate.py
│  │  │  ├─ losses.py
│  │  │  ├─ optim.py
│  │  │  ├─ callbacks.py
│  │  │  ├─ checkpointing.py
│  │  │  └─ freeze.py
│  │  ├─ experiments/
│  │  │  ├─ run_pretrain.py
│  │  │  ├─ run_finetune.py
│  │  │  ├─ run_lodo.py
│  │  │  ├─ run_ablation.py
│  │  │  ├─ run_freeze_sweep.py
│  │  │  ├─ run_similarity.py
│  │  │  ├─ run_occlusion.py
│  │  │  └─ run_stability.py
│  │  ├─ analysis/
│  │  │  ├─ representation_similarity.py
│  │  │  ├─ attribution.py
│  │  │  ├─ stability.py
│  │  │  ├─ statistics.py
│  │  │  └─ summarize.py
│  │  └─ reporting/
│  │     ├─ tables.py
│  │     ├─ figures.py
│  │     └─ export.py
├─ scripts/
│  ├─ prepare_metadata.py
│  ├─ build_splits.py
│  ├─ preprocess_images.py
│  ├─ launch_lodo_baseline.ps1
│  ├─ launch_lodo_dg.ps1
│  ├─ launch_transfer_analysis.ps1
│  └─ launch_explainability.ps1
├─ notebooks/
│  ├─ 01_dataset_audit.ipynb
│  ├─ 02_label_mapping_check.ipynb
│  ├─ 03_result_review.ipynb
│  └─ 04_figure_draft.ipynb
├─ outputs/
│  ├─ checkpoints/
│  ├─ predictions/
│  ├─ metrics/
│  ├─ attributions/
│  ├─ similarities/
│  ├─ figures/
│  └─ tables/
└─ tests/
   ├─ test_splits.py
   ├─ test_dataset_shapes.py
   ├─ test_losses.py
   ├─ test_metrics.py
   └─ test_stability.py
```

---

## 3. Folder Responsibilities

### `configs/`

Experiment settings must be separated from code.

- `data/`: cohort-specific paths, label mappings, MMSE field names, inclusion rules
- `environment/`: machine-specific root path examples for Windows and Linux
- `experiment/`: LODO split choice, training mode, DG on/off, seed, metrics
- `model/`: backbone and head hyperparameters

This makes it easy to compare:

- `TL`
- `TL + DANN-lite`
- `TL + GroupDRO`
- `Scratch`

without changing Python code.

### `data/`

This should not contain ambiguous files.

- `raw/`: original data links or mounted files
- `interim/`: intermediate results from preprocessing
- `processed/`: final tensor-ready volumes and merged metadata
- `splits/`: frozen LODO split CSV files
- `metadata/`: subject-level tables, cohort statistics, label harmonization tables

### `src/brainage/data/`

This is where dataset harmonization logic lives.

Core responsibilities:

- cohort metadata parsing
- class label mapping
- MMSE target cleaning
- subject inclusion and exclusion
- common transforms for all cohorts
- dataset object for PyTorch training

### `src/brainage/models/`

This layer should support a shared backbone with optional task heads:

- classification head for dementia stage
- regression head for MMSE
- multitask head for joint prediction
- optional domain classifier for DANN-lite

### `src/brainage/training/`

This separates training mechanics from experiment orchestration.

- `pretrain.py`: NC/QC-based source pretraining
- `finetune.py`: patient cohort fine-tuning
- `multitask.py`: joint loss handling
- `evaluate.py`: classification + regression evaluation
- `freeze.py`: freeze-depth control for transferability analysis

### `src/brainage/analysis/`

This is the most important layer for the paper contribution beyond plain accuracy.

- `representation_similarity.py`: layer-wise similarity between pretrained and fine-tuned models
- `attribution.py`: occlusion or attribution map generation
- `stability.py`: seed-wise overlap and reproducibility metrics
- `statistics.py`: mean, SD, confidence summaries, paired comparisons if needed

### `src/brainage/reporting/`

This should produce paper-ready outputs directly from saved metrics.

- table export
- figure generation
- merged summary for manuscript

---

## 4. Recommended Experiment Pipeline

The entire project should be run in the following order.

### Phase 1. Dataset audit and metadata standardization

Goal:
make HCP, OASIS, and ADNI comparable enough for a fair LODO setup.

Tasks:

- define subject ID rules for each cohort
- map diagnostic labels into unified classes
- identify which cohorts contain MMSE and how missing values are handled
- verify age, sex, diagnosis, MMSE availability
- generate cohort summary table

Outputs:

- `data/metadata/*.csv`
- `data/metadata/label_mapping.csv`
- `data/metadata/cohort_summary.csv`

### Phase 2. Preprocessing standardization

Goal:
ensure every volume arrives in the same input space.

Tasks:

- MNI alignment check
- voxel spacing standardization
- brain crop or resize policy
- intensity normalization policy
- save processed file manifest

Outputs:

- `data/processed/`
- `data/metadata/processed_manifest.csv`

### Phase 3. Frozen LODO split generation

Goal:
avoid leakage and make every experiment reproducible.

Tasks:

- define 3 folds:
  - train on 2 cohorts, test on 1 cohort
- define validation policy inside training cohorts
- write split files once and never regenerate casually

Recommended split unit:

- hold out the full target cohort as test
- split train and validation only inside source cohorts
- keep subject-level exclusivity

Outputs:

- `data/splits/lodo_hcp_holdout.csv`
- `data/splits/lodo_oasis_holdout.csv`
- `data/splits/lodo_adni_holdout.csv`

### Phase 4. Baseline transfer learning

Goal:
establish the core paper baseline.

Training flow:

1. pretrain backbone on normal/control data
2. attach task heads
3. fine-tune on patient task data
4. evaluate on held-out cohort

Variants:

- classification only
- regression only
- multitask classification + MMSE regression

Current status:

- implemented and running
- OASIS regression transfer is the strongest downstream result
- ADNI classification transfer remains the bottleneck

Outputs:

- pretrained checkpoints
- fine-tuned checkpoints
- fold-wise predictions
- fold-wise metrics

### Phase 5. DG module ablation

Goal:
test whether DG improves external cohort robustness.

Start simple:

- choose one DG method first
- compare against TL baseline

Recommended order:

1. `TL`
2. `TL + DANN-lite`
3. `TL + GroupDRO`
4. `Scratch`

Metrics to compare:

- classification: accuracy, balanced accuracy, macro F1, AUROC if applicable
- regression: MAE, RMSE, Pearson or Spearman correlation
- calibration or error profile if time allows

Outputs:

- `outputs/metrics/lodo_baseline.csv`
- `outputs/metrics/lodo_dann.csv`
- `outputs/metrics/lodo_groupdro.csv`
- `outputs/tables/ablation_main_table.csv`

Current status:

- a first `GroupDRO` variant is implemented for multi-source MMSE pretraining
- current result is weaker than ERM and should be treated as preliminary

### Phase 6. Transferability analysis

Goal:
show where transfer happens in the network.

Two mandatory analyses:

- freeze-depth sweep
- representation similarity

#### 6A. Freeze-depth sweep

Tasks:

- freeze 0 to N blocks during fine-tuning
- evaluate classification and regression performance
- summarize as heatmap

Interpretation:

- stable early layers suggest general anatomical features
- adaptive later layers suggest disease-specific transfer

#### 6B. Representation similarity

Tasks:

- capture activations per layer
- compare pretrained vs fine-tuned representations
- compute similarity metric by layer

Candidate metrics:

- CKA
- cosine similarity on pooled activations
- correlation-based similarity

Outputs:

- `outputs/similarities/freeze_sweep.csv`
- `outputs/similarities/layer_similarity.csv`
- `outputs/figures/freeze_sweep.png`
- `outputs/figures/layer_similarity.png`

### Phase 7. Explainability and stability

Goal:
show that explanations are not only plausible but reproducible.

Tasks:

- generate occlusion maps for selected subjects
- repeat training with multiple seeds
- compare top-k important regions across seeds

Recommended seed count:

- 5 seeds minimum

Recommended stability metrics:

- Dice overlap of top-k masks
- Jaccard overlap
- voxel-wise correlation
- rank consistency if ROI summaries are used

Compare:

- TL baseline stability
- TL + DG stability

Outputs:

- `outputs/attributions/`
- `outputs/metrics/stability_summary.csv`
- `outputs/figures/stability_comparison.png`

Current status:

- repeated-seed utilities exist for ADNI classification
- lightweight occlusion and stability experiments are still pending

### Phase 8. Reporting and manuscript asset generation

Goal:
make all paper figures reproducible from saved outputs.

Expected figure set:

- Fig 1. LODO study design
- Fig 2. model architecture + DG module
- Fig 3. LODO performance results
- Fig 4. freeze-depth sweep
- Fig 5. layer-wise representation similarity
- Fig 6. explainability stability

Expected table set:

- cohort summary table
- main LODO performance table
- ablation table
- stability summary table

---

## 5. Suggested Execution Units

To keep the project manageable, each script should do one thing.

Recommended entry points:

- `scripts/prepare_metadata.py`
- `scripts/build_splits.py`
- `src/brainage/experiments/run_pretrain.py`
- `src/brainage/experiments/run_finetune.py`
- `src/brainage/experiments/run_lodo.py`
- `src/brainage/experiments/run_ablation.py`
- `src/brainage/experiments/run_freeze_sweep.py`
- `src/brainage/experiments/run_similarity.py`
- `src/brainage/experiments/run_occlusion.py`
- `src/brainage/experiments/run_stability.py`

This avoids the common problem of one oversized training script trying to do everything.

---

## 6. Minimal Viable Implementation Order

If implementation must be staged, build in this order:

1. metadata schema and cohort loader
2. preprocessing manifest and processed dataset loader
3. frozen LODO split builder
4. 3D-CNN baseline for classification
5. MMSE regression head
6. multitask learning support
7. DG method 1
8. freeze-depth sweep
9. representation similarity
10. occlusion and stability analysis
11. reporting scripts

This order matches the paper's dependency chain.

---

## 7. Design Principles

### Keep split logic immutable

LODO splits should be saved as files and versioned.
Do not regenerate them ad hoc during experiments.

### Separate analysis from training

Interpretability and transfer analysis should read saved checkpoints and outputs.
They should not be embedded inside the training loop.

### Save predictions, not only summary metrics

For paper revision and error analysis, subject-level predictions are essential.

### Keep DG modular

DG components should be switchable by config.
Do not hardcode DANN-specific behavior into the backbone.

### Reproducibility first

Every run should save:

- config snapshot
- seed
- split identifier
- checkpoint path
- metric summary
- subject-level predictions

---

## 8. Example Run Sequence

```text
1. prepare_metadata.py
2. preprocess_images.py
3. build_splits.py
4. run_pretrain.py --config baseline_tl.yaml
5. run_lodo.py --config baseline_tl.yaml
6. run_lodo.py --config tl_dann.yaml
7. run_ablation.py
8. run_freeze_sweep.py
9. run_similarity.py
10. run_occlusion.py
11. run_stability.py
12. export tables and figures
```

---

## 9. What Should Be Defined Before Coding Starts

Before implementation, the following decisions should be fixed in writing:

- exact diagnostic class mapping across cohorts
- whether `QC` is truly a separate class or merged with `NC`
- which cohorts contribute to pretraining vs fine-tuning
- whether multitask learning is joint from the start or added after baseline
- whether the first paper version uses `DANN-lite`, `GroupDRO`, or both
- what attribution method is primary if occlusion is too expensive
- which metric is the primary endpoint for classification and for regression

---

## 10. Recommended Immediate Next Deliverables

The next practical files to create are:

1. `README.md`
   - quickstart
   - project scope
   - experiment naming rules

2. `data/metadata/label_mapping.csv`
   - unified diagnosis schema

3. `configs/experiment/baseline_tl.yaml`
   - first runnable baseline

4. `scripts/build_splits.py`
   - frozen LODO split generation

5. `src/brainage/data/schemas.py`
   - subject record definition

These five artifacts are enough to start implementation without losing the research logic.

---

## 11. Current Result Summary

### Stage 1

- HCP MMSE baseline is working
- HCP multimodal MMSE baseline is slightly better than MRI-only
- multi-source healthy ERM pretraining is currently the best Stage 1 extension
- current GroupDRO setup underperforms ERM

### Stage 2A

- OASIS MMSE transfer clearly benefits from multi-source ERM pretraining
- this is the cleanest positive result in the project so far

### Stage 2B

- ADNI scratch baseline is weak but still more meaningful than current transfer collapse
- transfer variants repeatedly predict a single dominant class
- HCP-pretrained, multi-source ERM-pretrained, MMSE-auxiliary, and multi-task variants have not solved this yet

### Stage 3

- LODO shows a substantial external generalization gap
- it is currently more useful as a diagnostic result than as a solved benchmark

### Practical manuscript implication

The current project is strongest if written as:

- a successful healthy MMSE pretraining and same-task transfer study
- plus a clear limitation result showing that disease-classification transfer is not automatic
