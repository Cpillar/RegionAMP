"""FASTA parsing and dataset utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class FastaSample:
    """Single FASTA sample containing header, sequence, and label string."""

    header: str
    sequence: str
    labels: str


def read_fasta_three_line(path: Path) -> List[FastaSample]:
    """Parse a 3-line FASTA file (header, sequence, labels).

    Lines are expected to repeat in blocks of 3:
    >id
    SEQUENCE
    LABELS
    """
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    samples: List[FastaSample] = []
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines) or not lines[i].startswith(">"):
            continue
        seq, lab = lines[i + 1], lines[i + 2]
        if len(seq) != len(lab):
            # Skip invalid sample; length mismatch
            continue
        samples.append(FastaSample(header=lines[i], sequence=seq, labels=lab))
    return samples


class ResidueClassificationDataset(Dataset):
    """Dataset for stage 1 token classification (uses ignore_index=-100 for padding)."""

    def __init__(
        self,
        fasta_path: Path,
        label_map: Dict[str, int],
        tokenizer,
        max_len: int = 200,
        samples: Optional[Iterable[FastaSample]] = None,
    ):
        self.samples = list(samples) if samples is not None else read_fasta_three_line(fasta_path)
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample.sequence,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        labels = torch.full_like(input_ids, fill_value=-100)
        lab_ids = [self.label_map[c] for c in sample.labels]
        usable = min(len(lab_ids), self.max_len - 2)  # reserve BOS/EOS/CLS/SEP
        labels[1 : 1 + usable] = torch.tensor(lab_ids[:usable])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ResidueCRFDataset(Dataset):
    """Dataset for stage 2 CRF training/eval with padding label at index 0."""

    def __init__(
        self,
        fasta_path: Path,
        label_map: Dict[str, int],
        tokenizer,
        max_len: int = 200,
        pad_label: int = 0,
        samples: Optional[Iterable[FastaSample]] = None,
    ):
        self.samples = list(samples) if samples is not None else read_fasta_three_line(fasta_path)
        self.label_map = dict(label_map)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_label = pad_label
        if "<pad>" not in self.label_map:
            self.label_map["<pad>"] = pad_label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample.sequence,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )
        input_ids = torch.tensor(enc["input_ids"])
        attention_mask = torch.tensor(enc["attention_mask"])

        labels = torch.full_like(input_ids, fill_value=self.pad_label, dtype=torch.long)
        # Exclude special tokens; reserve positions
        max_seq_token_len = self.max_len - self.tokenizer.num_special_tokens_to_add(pair=False)
        label_ids = [self.label_map.get(c, self.pad_label) for c in sample.labels[:max_seq_token_len]]

        if label_ids:
            start = 1  # assume single leading special token before labels
            end = start + len(label_ids)
            usable_end = min(end, self.max_len - 1)
            usable_len = usable_end - start
            labels[start:usable_end] = torch.tensor(label_ids[:usable_len], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
