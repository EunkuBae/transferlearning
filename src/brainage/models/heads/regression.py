"""Regression head for scalar outputs."""

from __future__ import annotations

from brainage.models.backbones.cnn3d import require_torch

try:
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    nn = None


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        require_torch()
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)
