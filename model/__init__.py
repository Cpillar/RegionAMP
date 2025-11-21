"""Model package for AMP region workflows.

Contains reusable components for:
- FASTA parsing and dataset utilities.
- ESM-based token classification (stage 1) and ESM+CRF sequence labeling (stage 2).
- Training (stage 1/2), nested hyperparameter search, evaluation, and inference.
All public functions are imported in this namespace for convenience.
"""

from .config import (
    LabelConfig,
    Stage1Config,
    Stage2Config,
    NestedCVConfig,
    EvalConfig,
    PredictionConfig,
)
from .data import (
    FastaSample,
    ResidueClassificationDataset,
    ResidueCRFDataset,
    read_fasta_three_line,
)
from .metrics import (
    compute_token_metrics,
    compute_crf_metrics,
    compute_iou,
)
from .models import (
    EsmTokenClassifier,
    EsmCrfTagger,
)
from .training import (
    train_stage1,
    train_stage2,
)
from .nested_cv import run_nested_cv
from .eval import evaluate_model, evaluate_model_iou_threshold
from .predict import predict_fasta

__all__ = [
    # Configs
    "LabelConfig",
    "Stage1Config",
    "Stage2Config",
    "NestedCVConfig",
    "EvalConfig",
    "PredictionConfig",
    # Data
    "FastaSample",
    "ResidueClassificationDataset",
    "ResidueCRFDataset",
    "read_fasta_three_line",
    # Metrics
    "compute_token_metrics",
    "compute_crf_metrics",
    "compute_iou",
    # Models
    "EsmTokenClassifier",
    "EsmCrfTagger",
    # Training
    "train_stage1",
    "train_stage2",
    # Nested CV
    "run_nested_cv",
    # Evaluation
    "evaluate_model",
    "evaluate_model_iou_threshold",
    # Prediction
    "predict_fasta",
]
