"""Model definitions for AMP region tasks."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import EsmForTokenClassification, EsmModel, EsmConfig


class EsmTokenClassifier(nn.Module):
    """Stage 1 token classifier wrapper around Hugging Face EsmForTokenClassification."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.model = EsmForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, **inputs):
        return self.model(**inputs)


class EsmCrfTagger(nn.Module):
    """Stage 2 ESM + CRF tagger."""

    def __init__(self, base_model_path: str, num_labels: int):
        super().__init__()
        try:
            self.esm = EsmModel.from_pretrained(base_model_path)
            self.config = self.esm.config
        except Exception as exc:
            raise RuntimeError(f"Failed to load base ESM model from {base_model_path}") from exc

        self.classifier = nn.Linear(self.esm.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)

        # Align shapes just in case
        mask = attention_mask.bool()
        if mask.shape[1] != logits.shape[1]:
            min_len = min(mask.shape[1], logits.shape[1])
            mask = mask[:, :min_len]
            logits = logits[:, :min_len, :]
            if labels is not None:
                labels = labels[:, :min_len]

        if labels is not None:
            clamped_labels = torch.clamp(labels.long(), 0, self.num_labels - 1)
            loss = -self.crf(logits, clamped_labels, mask=mask, reduction="mean")
            return loss, logits

        # Inference: return logits only (Trainer.predict) or decode manually elsewhere
        return logits


def load_crf_structure_from_config(config_dir: str, num_labels: int) -> EsmCrfTagger:
    """Instantiate CRF tagger only from config (weights loaded separately)."""
    cfg = EsmConfig.from_pretrained(config_dir)
    model = EsmCrfTagger.__new__(EsmCrfTagger)
    nn.Module.__init__(model)
    model.esm = EsmModel(cfg)
    model.config = cfg
    model.classifier = nn.Linear(cfg.hidden_size, num_labels)
    model.crf = CRF(num_labels, batch_first=True)
    model.num_labels = num_labels
    return model
