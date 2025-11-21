"""Metrics utilities."""

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_token_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for stage 1 token classification (ignore_index=-100)."""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    preds = predictions.argmax(-1).flatten()
    labs = labels.flatten()
    mask = labs != -100
    preds, labs = preds[mask], labs[mask]

    acc = accuracy_score(labs, preds)
    rec = recall_score(labs, preds, average="macro", zero_division=0)
    f1 = f1_score(labs, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labs, preds)
    pM, rM, fM, _ = precision_recall_fscore_support(
        labs, preds, labels=[2], average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "recall_macro": rec,
        "f1_macro": f1,
        "mcc": mcc,
        "precision_M": pM,
        "recall_M": rM,
        "f1_M": fM,
    }


def _flatten_predictions_with_mask(
    logits: np.ndarray,
    labels: np.ndarray,
    attention_mask: np.ndarray,
    valid_label_ids: Iterable[int],
) -> Tuple[List[int], List[int]]:
    """Flatten predictions and labels using attention mask to recover true length."""
    pred_indices = logits.argmax(axis=2)
    y_true, y_pred = [], []
    valid_set = set(valid_label_ids)

    for i in range(labels.shape[0]):
        mask_seq = attention_mask[i]
        true_len = int(mask_seq.sum()) - 2  # exclude BOS/EOS/SEP
        true_len = max(0, min(true_len, logits.shape[1] - 2))
        t_seq = labels[i, 1 : 1 + true_len].tolist()
        p_seq = pred_indices[i, 1 : 1 + true_len].tolist()
        if len(t_seq) != len(p_seq):
            continue
        for t, p in zip(t_seq, p_seq):
            if t in valid_set:
                y_true.append(t)
                y_pred.append(p)
    return y_true, y_pred


def compute_crf_metrics(eval_pred, label_map: Dict[str, int], max_len: int) -> Dict[str, float]:
    """Compute metrics for CRF model outputs."""
    pred_input = eval_pred.predictions
    labels_true = eval_pred.label_ids

    # unwrap tuple if (loss, logits)
    logits = pred_input[0] if isinstance(pred_input, tuple) else pred_input
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels_true, torch.Tensor):
        labels_true = labels_true.cpu().numpy()

    attention_mask = None
    if hasattr(eval_pred, "inputs") and isinstance(eval_pred.inputs, dict):
        am = eval_pred.inputs.get("attention_mask")
        if isinstance(am, torch.Tensor):
            attention_mask = am.cpu().numpy()
        elif isinstance(am, np.ndarray):
            attention_mask = am

    if attention_mask is None:
        # Fallback: build mask based on non-zero labels length
        attention_mask = np.ones_like(labels_true, dtype=np.int64)

    valid_labels = list(label_map.values())
    y_true, y_pred = _flatten_predictions_with_mask(logits, labels_true, attention_mask, valid_labels)

    if not y_true:
        return {
            "accuracy": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "mcc": 0.0,
            "precision_M": 0.0,
            "recall_M": 0.0,
            "f1_M": 0.0,
        }

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=valid_labels, average="macro", zero_division=0
    )
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = 0.0

    label_m = label_map.get("M")
    pM = rM = fM = 0.0
    if label_m is not None:
        pM, rM, fM, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[label_m], average="macro", zero_division=0, warn_for=()
        )

    return {
        "accuracy": acc,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "mcc": mcc,
        "precision_M": pM,
        "recall_M": rM,
        "f1_M": fM,
    }


def compute_iou(true_labels: List[int], pred_labels: List[int], target_label_id: int) -> float:
    """Compute IoU for a specific label id."""
    true_idx = {i for i, l in enumerate(true_labels) if l == target_label_id}
    pred_idx = {i for i, l in enumerate(pred_labels) if l == target_label_id}
    union = len(true_idx | pred_idx)
    if union == 0:
        return float("nan")
    return len(true_idx & pred_idx) / union


def confusion(y_true: List[int], y_pred: List[int], labels: List[int]) -> np.ndarray:
    """Return confusion matrix helper."""
    return confusion_matrix(y_true, y_pred, labels=labels)
