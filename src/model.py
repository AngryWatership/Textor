"""Encoder + classification head, with a partial/full fine-tune toggle.

partial (default): freeze all encoder layers except the top N (default 2)
                    plus the classification head. Safer given only ~2,093
                    labeled examples and a minority class of 218.
full:               fine-tune every encoder parameter. Higher capacity,
                    higher overfitting risk -- compare against partial's
                    held-out numbers before trusting this is actually better.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

N_CLASSES = 3


class VerdictClassifier(nn.Module):
    def __init__(self, base_model="allenai/specter2_base", finetune_mode="partial",
                 n_unfrozen_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, N_CLASSES)
        self.finetune_mode = finetune_mode

        if finetune_mode == "partial":
            self._freeze_all_but_top(n_unfrozen_layers)
        elif finetune_mode == "full":
            pass  # everything trainable
        else:
            raise ValueError(f"unknown finetune_mode: {finetune_mode}")

    def _freeze_all_but_top(self, n_unfrozen_layers):
        for p in self.encoder.parameters():
            p.requires_grad = False
        # unfreeze embeddings' layer norm + top N transformer layers;
        # exact module path depends on the base model's architecture
        # (BERT-family models expose .encoder.layer)
        layers = self.encoder.encoder.layer
        for layer in layers[-n_unfrozen_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over tokens, masked -- works for any BERT-family encoder
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(self.dropout(pooled))
        return logits, pooled  # return pooled embedding too, for silhouette/purity eval


def load_tokenizer(base_model):
    return AutoTokenizer.from_pretrained(base_model)
