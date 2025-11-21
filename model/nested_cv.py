"""Nested cross-validation for stage 2 ESM+CRF."""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.optim import AdamW
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .config import LabelConfig, NestedCVConfig
from .data import ResidueCRFDataset, FastaSample, read_fasta_three_line
from .metrics import compute_crf_metrics
from .models import EsmCrfTagger


def _make_optimizer(model: EsmCrfTagger, lr_esm: float, lr_mult: float) -> AdamW:
    crf_params = list(model.crf.parameters())
    classifier_params = list(model.classifier.parameters())
    special_ids = {id(p) for p in crf_params + classifier_params}
    esm_params = [p for _, p in model.named_parameters() if p.requires_grad and id(p) not in special_ids]
    head_params = [p for _, p in model.named_parameters() if p.requires_grad and id(p) in special_ids]
    return AdamW(
        [
            {"params": esm_params, "lr": lr_esm},
            {"params": head_params, "lr": lr_esm * lr_mult},
        ]
    )


def _collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


def _train_eval_once(
    hp: Dict,
    train_samples: List[FastaSample],
    val_samples: List[FastaSample],
    cfg: NestedCVConfig,
    labels: LabelConfig,
    run_dir: Path,
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    model = EsmCrfTagger(str(cfg.base_model_path), num_labels=labels.num_labels())
    train_ds = ResidueCRFDataset(
        cfg.train_fasta, labels.label_map, tokenizer, labels.max_len, pad_label=labels.pad_label, samples=train_samples
    )
    val_ds = ResidueCRFDataset(
        cfg.val_fasta, labels.label_map, tokenizer, labels.max_len, pad_label=labels.pad_label, samples=val_samples
    )

    lr_esm = hp.get("learning_rate_esm", 1e-5)
    lr_mult = hp.get("crf_head_lr_multiplier", 10.0)
    args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=hp.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=hp.get("per_device_eval_batch_size", hp.get("per_device_train_batch_size", 8)),
        learning_rate=lr_esm,
        num_train_epochs=hp.get("num_train_epochs", 10),
        weight_decay=hp.get("weight_decay", 0.01),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_M",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        include_inputs_for_metrics=True,
        seed=cfg.random_seed_training,
    )

    optimizer = _make_optimizer(model, lr_esm, lr_mult)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_crf_metrics(eval_pred, labels.label_map, labels.max_len),
        data_collator=_collate,
        optimizers=(optimizer, None),
    )

    trainer.train()
    metrics = trainer.evaluate()
    # cleanup to save space
    if run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_dir():
                for sub in child.iterdir():
                    if sub.is_file():
                        sub.unlink()
                child.rmdir()
            else:
                child.unlink()
    return metrics


def run_nested_cv(cfg: NestedCVConfig, labels: LabelConfig) -> Tuple[Dict, float]:
    """Run nested CV and return best hyperparameters and mean outer f1_M."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = cfg.temp_runs_dir()
    temp_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    train_samples = read_fasta_three_line(cfg.train_fasta)
    val_samples = read_fasta_three_line(cfg.val_fasta)
    all_dev = np.array(train_samples + val_samples, dtype=object)

    outer_scores: List[float] = []
    outer_best_hps: List[Dict] = []

    import itertools

    hp_keys, hp_values = zip(*cfg.hyperparameter_grid.items())
    hp_grid = [dict(zip(hp_keys, combo)) for combo in itertools.product(*hp_values)]

    outer_splitter = KFold(n_splits=cfg.k_outer, shuffle=True, random_state=cfg.random_seed_split)
    for outer_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(all_dev), start=1):
        current_train = all_dev[train_idx].tolist()
        current_test = all_dev[test_idx].tolist()

        inner_splitter = KFold(n_splits=cfg.k_inner, shuffle=True, random_state=cfg.random_seed_split + outer_idx)
        hp_scores: Dict[str, List[float]] = {str(hp): [] for hp in hp_grid}

        for hp in hp_grid:
            for inner_train_idx, inner_val_idx in inner_splitter.split(current_train):
                inner_train = [current_train[i] for i in inner_train_idx]
                inner_val = [current_train[i] for i in inner_val_idx]
                run_dir = temp_dir / f"outer{outer_idx}_hp{str(hp)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                res = _train_eval_once(hp, inner_train, inner_val, cfg, labels, run_dir)
                hp_scores[str(hp)].append(res.get("eval_f1_M", -1.0))

        # select best hp
        best_hp = max(hp_grid, key=lambda h: np.mean(hp_scores[str(h)]) if hp_scores[str(h)] else -1.0)
        outer_best_hps.append(best_hp)

        # train on outer train + evaluate on outer test
        run_dir = temp_dir / f"outer{outer_idx}_final"
        run_dir.mkdir(parents=True, exist_ok=True)
        outer_res = _train_eval_once(best_hp, current_train, current_test, cfg, labels, run_dir)
        outer_scores.append(outer_res.get("eval_f1_M", -1.0))

    mean_f1 = float(np.mean(outer_scores)) if outer_scores else -1.0
    std_f1 = float(np.std(outer_scores)) if outer_scores else 0.0

    # most frequent best hp across outer folds
    counts = {}
    for hp in outer_best_hps:
        counts[str(hp)] = counts.get(str(hp), 0) + 1
    if counts:
        import ast

        best_overall = ast.literal_eval(max(counts, key=counts.get))
    else:
        best_overall = outer_best_hps[0] if outer_best_hps else {}

    # save summaries
    import json

    with open(cfg.best_hparam_file(), "w", encoding="utf-8") as f:
        json.dump(best_overall, f, indent=4)
    with open(cfg.summary_file(), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_outer_f1_M": mean_f1,
                "std_outer_f1_M": std_f1,
                "outer_fold_f1_M_scores": outer_scores,
                "best_hyperparams_per_outer_fold": outer_best_hps,
                "overall_best_hyperparameters_selected": best_overall,
                "hyperparameter_grid_searched": cfg.hyperparameter_grid,
            },
            f,
            indent=4,
        )

    return best_overall, mean_f1
