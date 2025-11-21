"""Training routines for stage 1 and stage 2."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .config import LabelConfig, Stage1Config, Stage2Config
from .data import ResidueClassificationDataset, ResidueCRFDataset
from .metrics import compute_token_metrics, compute_crf_metrics
from .models import EsmTokenClassifier, EsmCrfTagger


class WeightedTrainer(Trainer):
    """Trainer that applies class weights for CrossEntropy."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        self.class_weights = class_weights
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(model.device)
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        logits = outputs.logits
        loss_f = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(model.device) if self.class_weights is not None else None,
            ignore_index=-100,
        )
        loss = loss_f(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_stage1(cfg: Stage1Config, labels: LabelConfig) -> Path:
    """Train stage 1 token classifier."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = ResidueClassificationDataset(cfg.train_fasta, labels.label_map, tokenizer, labels.max_len)
    val_ds = ResidueClassificationDataset(cfg.val_fasta, labels.label_map, tokenizer, labels.max_len)

    model = EsmTokenClassifier(cfg.model_name, num_labels=labels.num_labels())

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.save_steps,
        save_steps=cfg.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_M",
        greater_is_better=True,
        report_to="none",
        logging_steps=cfg.logging_steps,
    )

    class_weights = torch.tensor([0.0, 1.0, 1.5, 1.0]) if cfg.use_class_weights else None
    trainer_cls = WeightedTrainer if class_weights is not None else Trainer

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_token_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    best_path = Path(cfg.output_dir) / "best_model"
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    return best_path


def train_stage2(cfg: Stage2Config, labels: LabelConfig) -> Path:
    """Train stage 2 ESM+CRF model."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    train_ds = ResidueCRFDataset(
        cfg.train_fasta, labels.label_map, tokenizer, labels.max_len, pad_label=labels.pad_label
    )
    val_ds = ResidueCRFDataset(
        cfg.val_fasta, labels.label_map, tokenizer, labels.max_len, pad_label=labels.pad_label
    )

    model = EsmCrfTagger(str(cfg.base_model_path), num_labels=labels.num_labels())

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate_esm,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_M",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        report_to="none",
        fp16=cfg.fp16 and torch.cuda.is_available(),
        include_inputs_for_metrics=True,
    )

    esm_lr = cfg.learning_rate_esm
    crf_lr_mult = cfg.crf_head_lr_multiplier
    crf_params = list(model.crf.parameters())
    classifier_params = list(model.classifier.parameters())
    special_ids = {id(p) for p in crf_params + classifier_params}
    esm_params = [p for _, p in model.named_parameters() if p.requires_grad and id(p) not in special_ids]
    head_params = [p for _, p in model.named_parameters() if p.requires_grad and id(p) in special_ids]
    optimizer = AdamW(
        [
            {"params": esm_params, "lr": esm_lr},
            {"params": head_params, "lr": esm_lr * crf_lr_mult},
        ]
    )

    def collate_fn(features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_crf_metrics(eval_pred, labels.label_map, labels.max_len),
        data_collator=collate_fn,
        optimizers=(optimizer, None),
    )

    trainer.train()
    best_path = Path(cfg.output_dir) / "best_model"
    best_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    # Save base config for reconstruction
    model.esm.config.save_pretrained(best_path)
    return best_path
