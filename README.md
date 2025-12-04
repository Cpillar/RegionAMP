AMP Region Model Package
========================

Refactored, modular Python project for AMP region detection. All former notebook logic is converted into reusable modules with clear English comments and CLI entrypoints.

Layout
------
- `model/config.py` - dataclasses for labels, training, nested CV, evaluation, prediction.
- `model/data.py` - FASTA parsing (3-line format) and datasets for stage1/2.
- `model/models.py` - stage1 token classifier and stage2 ESM+CRF tagger.
- `model/metrics.py` - token metrics, CRF metrics, IoU helper.
- `model/training.py` - training workflows for stage1 and stage2.
- `model/nested_cv.py` - nested CV search for stage2 hyperparameters.
- `model/eval.py` - evaluation on test sets and IoU-per-sequence variant.
- `model/predict.py` - FASTA batch prediction with the best model.
- `model/cli.py` - command-line interface wrapper for common tasks.
- `scripts/` - production-friendly entrypoints (train/eval/predict) that call the package APIs.

Dependencies
------------
- `torch`, `transformers`, `torchcrf`, `scikit-learn`, `safetensors`, `tqdm` (for CV progress).

Usage (CLI)
-----------
All commands accept `--max-len` to override the default 200. Run from repo root.

Train stage1 (token classification):
```
python -m model.cli train-stage1 \
  --train-fasta stage1_train.3line.fasta \
  --val-fasta stage1_test.3line.fasta \
  --output-dir esm_stage1
```

Train stage2 (ESM+CRF) using the stage1 best checkpoint:
```
python -m model.cli train-stage2 \
  --base-model esm_stage1/best_model \
  --train-fasta train_dataset_stage2.3line.fasta \
  --val-fasta validation_dataset_stage2.3line.fasta \
  --output-dir esm_stage2_crf
```

Nested CV to pick hyperparameters:
```
python -m model.cli nested-cv \
  --base-model esm_stage1/best_model \
  --train-fasta train_dataset_stage2.3line.fasta \
  --val-fasta validation_dataset_stage2.3line.fasta \
  --output-dir esm_stage2_cv_outputs
```

Evaluate on a test set:
```
python -m model.cli eval \
  --model-dir esm_stage2_crf/best_model \
  --test-fasta test_dataset_stage2.3line.fasta \
  --output-dir esm_stage2_crf/evaluation_results
```

Evaluate per-sequence IoU threshold (for label M by default):
```
python -m model.cli eval-iou \
  --model-dir esm_stage2_crf/best_model \
  --test-fasta test_dataset_stage2.3line.fasta \
  --output-dir esm_stage2_crf/evaluation_results_iou \
  --threshold 0.5
```

Predict labels for sequences in a FASTA (two-line records):
```
python -m model.cli predict \
  --model-dir esm_stage2_crf/best_model \
  --fasta your_sequences.fasta \
  --output predictions.tsv
```

Scripts
-------
Ops-friendly wrappers are in `scripts/`:
- `python scripts/train_stage1.py ...`
- `python scripts/train_stage2.py ...`
- `python scripts/nested_cv.py ...`
- `python scripts/evaluate.py ...`
- `python scripts/evaluate_iou.py ...`
- `python scripts/predict.py --fasta your_sequences.fasta ...`


Dataset availability
--------------------
The datasets will be published on the day our article is officially released.
