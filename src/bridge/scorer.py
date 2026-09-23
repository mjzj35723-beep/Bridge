"""Lightweight BRIDGE scorer definition for CPU smoke tests and inspection."""
from __future__ import annotations

import torch
import torch.nn as nn


class EarlyBranchScorer(nn.Module):
    def __init__(self, dim: int = 8, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim + 1, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        zbar = (p.unsqueeze(-1) * z).sum(dim=-2, keepdim=True).expand_as(z)
        features = torch.cat([z, zbar, z - zbar, torch.log(p.unsqueeze(-1) + 1e-8)], dim=-1)
        return self.net(features).squeeze(-1)
