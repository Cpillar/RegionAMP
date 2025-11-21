"""Evaluation utilities."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from safetensors.torch import load_file

from .config import EvalConfig, LabelConfig
from .data import ResidueCRFDataset
from .metrics import compute_crf_metrics, compute_iou
from .models import EsmCrfTagger, load_crf_structure_from_config


def _load_weights(model: EsmCrfTagger, model_dir: Path):
    """Load model weights from a directory containing model.safetensors or pytorch_model.bin."""
    sf = model_dir / "model.safetensors"
    pt = model_dir / "pytorch_model.bin"
    if sf.is_file():
        state = load_file(sf, device="cpu")
    elif pt.is_file():
        state = torch.load(pt, map_location=torch.device("cpu"))
    else:
        raise FileNotFoundError(f"No model weights found in {model_dir}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys during load: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys during load: {unexpected}")


def evaluate_model(cfg: EvalConfig, labels: LabelConfig) -> Dict[str, float]:
    """Evaluate model on test set using standard token metrics."""
    base_cfg = cfg.base_config_dir or cfg.model_dir
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = load_crf_structure_from_config(str(base_cfg), labels.num_labels())
    _load_weights(model, cfg.model_dir)

    test_ds = ResidueCRFDataset(
        cfg.test_fasta, labels.label_map, tokenizer, labels.max_len, pad_label=labels.pad_label
    )

    args = TrainingArguments(
        output_dir=str(cfg.output_dir / "temp_eval"),
        per_device_eval_batch_size=cfg.batch_size,
        do_train=False,
        do_predict=True,
        report_to="none",
        include_inputs_for_metrics=True,
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
        compute_metrics=lambda eval_pred: compute_crf_metrics(eval_pred, labels.label_map, labels.max_len),
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    predictions = trainer.predict(test_ds)
    metrics = predictions.metrics or {}
    (cfg.output_dir).mkdir(parents=True, exist_ok=True)
    import json

    with open(cfg.output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    return metrics


def evaluate_model_iou_threshold(
    cfg: EvalConfig, labels: LabelConfig, target_label: str = "M"
) -> Dict[str, object]:
    """Evaluate per-sequence IoU threshold for a target label."""
    base_cfg = cfg.base_config_dir or cfg.model_dir
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = load_crf_structure_from_config(str(base_cfg), labels.num_labels())
    _load_weights(model, cfg.model_dir)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    test_ds = ResidueCRFDataset(
        cfg.test_fasta, labels.label_map, tokenizer, labels.max_len, pad_label=labels.pad_label
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=data_collator)

    t_id = labels.label_map[target_label]
    device = next(model.parameters()).device
    all_true: List[int] = []
    all_pred: List[int] = []
    per_seq = []
    matched = 0
    true_with_label = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            logits = model(ids, mask)
            decoded = model.crf.decode(logits, mask=mask)
            for i in range(ids.shape[0]):
                true_len = int(mask[i].sum()) - 2
                true_len = max(0, min(true_len, logits.shape[1] - 2))
                true_seq = lbls[i, 1 : 1 + true_len].cpu().tolist()
                pred_seq = decoded[i][1 : 1 + true_len]
                all_true.extend(true_seq)
                all_pred.extend(pred_seq)
                if t_id in true_seq:
                    true_with_label += 1
                    iou = compute_iou(true_seq, pred_seq, t_id)
                    matched += int(iou >= cfg.iou_threshold) if not np.isnan(iou) else 0
                    per_seq.append({"iou": iou, "matched": iou >= cfg.iou_threshold if not np.isnan(iou) else False})

    from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score

    metrics = {}
    if all_true:
        valid_labels = list(labels.label_map.values())
        acc = accuracy_score(all_true, all_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_true, all_pred, labels=valid_labels, average="macro", zero_division=0
        )
        try:
            mcc = matthews_corrcoef(all_true, all_pred)
        except ValueError:
            mcc = 0.0
        metrics.update(
            {
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "mcc": mcc,
            }
        )

    summary = {
        "num_sequences": len(test_ds),
        "num_sequences_with_target": true_with_label,
        "num_sequences_matched_iou": matched,
        "iou_threshold": cfg.iou_threshold,
    }

    (cfg.output_dir).mkdir(parents=True, exist_ok=True)
    import json

    with open(cfg.output_dir / f"iou_thresh_{cfg.iou_threshold}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "metrics_all_tokens": metrics, "per_sequence": per_seq}, f, indent=4)
    return {"summary": summary, "metrics_all_tokens": metrics}
