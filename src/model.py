"""A Transformer encoder built entirely from scratch with torch.nn --
no pretrained weights, no transformers library, no Hugging Face, no
download of any kind. Randomly initialized and trained end-to-end on
this project's own 2,093 papers (via train.py). A deterministic, fully
trained function from paper text to label -- nothing queried, nothing
prompted, no external model fetched at runtime.
"""
import math

import torch
import torch.nn as nn

N_CLASSES = 3


class PositionalEncoding(nn.Module):
    """Standard sinusoidal position encoding (Vaswani et al.), computed
    directly -- no table or weights to load from anywhere."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class VerdictTransformer(nn.Module):
    """Embedding -> positional encoding -> nn.TransformerEncoder stack ->
    masked mean pool -> linear classifier. Every parameter is randomly
    initialized here and trained from zero; there is nothing frozen and
    nothing pretrained, so model_size controls capacity directly rather
    than choosing which of someone else's layers to unfreeze."""

    def __init__(self, vocab_size, embed_dim=256, num_layers=4, num_heads=8,
                 ffn_dim=1024, dropout=0.2, max_length=384, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos = PositionalEncoding(embed_dim, max_len=max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, N_CLASSES)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        x = self.pos(x)
        # TransformerEncoder's padding mask wants True where tokens should
        # be ignored
        key_padding_mask = attention_mask == 0
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(self.dropout(pooled))
        return logits, pooled  # pooled embedding too, for silhouette/purity eval
