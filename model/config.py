"""Configuration dataclasses for AMP region workflows."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LabelConfig:
    """Label mapping used across tasks."""

    label_map: Dict[str, int] = field(
        default_factory=lambda: {"S": 1, "M": 2, "N": 3}
    )
    pad_label: int = 0
    max_len: int = 200

    def num_labels(self) -> int:
        return max(self.label_map.values(), default=0) + 1


@dataclass
class Stage1Config:
    """Configuration for the stage 1 token classification model."""

    model_name: str = "facebook/esm2_t12_35M_UR50D"
    train_fasta: Path = Path("stage1_train.3line.fasta")
    val_fasta: Path = Path("stage1_test.3line.fasta")
    output_dir: Path = Path("./esm_stage1")
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 2
    use_class_weights: bool = True
    logging_steps: int = 200
    save_steps: int = 200


@dataclass
class Stage2Config:
    """Configuration for the stage 2 ESM+CRF fine-tuning."""

    base_model_path: Path = Path("./esm_stage1/best_model")
    train_fasta: Path = Path("train_dataset_stage2.3line.fasta")
    val_fasta: Path = Path("validation_dataset_stage2.3line.fasta")
    output_dir: Path = Path("./esm_stage2_crf")
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate_esm: float = 1e-5
    crf_head_lr_multiplier: float = 10.0
    weight_decay: float = 0.01
    num_train_epochs: int = 10
    logging_steps: int = 50
    save_total_limit: int = 2
    fp16: bool = True


@dataclass
class NestedCVConfig:
    """Configuration for nested cross-validation on stage 2."""

    base_model_path: Path = Path("./esm_stage1/best_model")
    train_fasta: Path = Path("train_dataset_stage2.3line.fasta")
    val_fasta: Path = Path("validation_dataset_stage2.3line.fasta")
    output_dir: Path = Path("./esm_stage2_cv_outputs")
    k_outer: int = 3
    k_inner: int = 5
    random_seed_split: int = 42
    random_seed_training: int = 123
    hyperparameter_grid: Dict[str, List] = field(
        default_factory=lambda: {
            "learning_rate_esm": [2e-5],
            "crf_head_lr_multiplier": [5.0, 10.0],
            "num_train_epochs": [15],
            "per_device_train_batch_size": [8],
            "weight_decay": [0.01],
        }
    )

    def temp_runs_dir(self) -> Path:
        return self.output_dir / "intermediate_cv_runs"

    def best_hparam_file(self) -> Path:
        return self.output_dir / "best_hyperparameters.json"

    def summary_file(self) -> Path:
        return self.output_dir / "nested_cv_performance_summary.json"


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    model_dir: Path = Path("./esm_stage2_crf/best_model")
    base_config_dir: Optional[Path] = None
    test_fasta: Path = Path("test_dataset_stage2.3line.fasta")
    output_dir: Path = Path("./esm_stage2_crf/evaluation_results")
    batch_size: int = 16
    iou_threshold: float = 0.5  # used by IOU evaluation only


@dataclass
class PredictionConfig:
    """Configuration for FASTA prediction."""

    model_dir: Path = Path("./esm_stage2_crf/best_model")
    fasta_path: Path = Path("input.fasta")
    output_path: Path = Path("./predictions.tsv")
    max_len: Optional[int] = None  # fallback to label config if None
