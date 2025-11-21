"""Prediction utilities for FASTA inputs."""

from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .config import LabelConfig, PredictionConfig
from .models import load_crf_structure_from_config


def _read_fasta_sequences(path: Path) -> List[Tuple[str, str]]:
    """Read FASTA where each record is two lines (header, sequence)."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pairs = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines) or not lines[i].startswith(">"):
            continue
        pairs.append((lines[i][1:], lines[i + 1]))
    return pairs


def _load_weights(model, model_dir: Path):
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


def predict_fasta(cfg: PredictionConfig, labels: LabelConfig) -> Path:
    """Predict labels for sequences in a FASTA file using the best stage 2 model."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    base_cfg_dir = cfg.model_dir
    model = load_crf_structure_from_config(str(base_cfg_dir), labels.num_labels())
    _load_weights(model, cfg.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sequences = _read_fasta_sequences(cfg.fasta_path)
    max_len = cfg.max_len or labels.max_len

    outputs = []
    inv_map = {v: k for k, v in labels.label_map.items()}
    inv_map[labels.pad_label] = "X"
    with torch.no_grad():
        for sid, seq in sequences:
            enc = tokenizer(
                seq,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            logits = model(ids, mask)
            pred_path = model.crf.decode(logits, mask=mask)[0]
            true_len = min(len(seq), max_len - 2)
            pred_labels = pred_path[1 : 1 + true_len]
            pred_str = "".join(inv_map.get(i, "X") for i in pred_labels)
            outputs.append((sid, seq, pred_str))

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("id\tsequence\tpredicted_labels\n")
        for sid, seq, lab in outputs:
            f.write(f"{sid}\t{seq}\t{lab}\n")
    return out_path
