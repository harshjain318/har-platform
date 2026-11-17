"""1-D CNN for HAR with explicit layer groups (block1..N, embed, head) for layer-wise adaptation."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(cout)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class HARNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 6, widths=(64, 128, 128), kernel: int = 5,
                 embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.config = dict(in_channels=int(in_channels), n_classes=int(n_classes), widths=[int(w) for w in widths],
                           kernel=int(kernel), embed_dim=int(embed_dim), dropout=float(dropout))
        chans = [in_channels, *widths]
        self.blocks = nn.ModuleList([ConvBlock(chans[i], chans[i + 1], kernel) for i in range(len(widths))])
        self.embed = nn.Sequential(nn.Linear(widths[-1], embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(embed_dim, n_classes)

    @property
    def layer_names(self) -> List[str]:
        return [f"block{i + 1}" for i in range(len(self.blocks))] + ["embed"]

    def forward(self, x: torch.Tensor, return_layers: bool = False):
        """x: (B, T, C). With return_layers=True also returns per-layer pooled features (B, D)."""
        h = x.transpose(1, 2)
        layers: Dict[str, torch.Tensor] = {}
        for i, b in enumerate(self.blocks):
            h = b(h)
            layers[f"block{i + 1}"] = h.mean(dim=2)
        z = self.embed(h.mean(dim=2))
        layers["embed"] = z
        logits = self.head(z)
        return (logits, layers) if return_layers else logits

    def layer_parameters(self) -> Dict[str, List[nn.Parameter]]:
        d = {f"block{i + 1}": list(b.parameters()) for i, b in enumerate(self.blocks)}
        d["embed"] = list(self.embed.parameters())
        d["head"] = list(self.head.parameters())
        return d


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
