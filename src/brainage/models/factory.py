"""Factory helpers for baseline models."""

from __future__ import annotations

from brainage.models.backbones.cnn3d import CNN3DBackbone, require_torch
from brainage.models.heads.regression import RegressionHead

try:
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    nn = None


class CNN3DRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        head_hidden_dim: int = 128,
        head_dropout: float = 0.2,
    ) -> None:
        require_torch()
        super().__init__()
        self.backbone = CNN3DBackbone(in_channels=in_channels, channels=channels)
        self.head = RegressionHead(
            input_dim=self.backbone.output_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_hcp_mmse_model(config: dict) -> CNN3DRegressor:
    model_config = config.get("model", {})
    channels = tuple(model_config.get("channels", [16, 32, 64, 128]))
    return CNN3DRegressor(
        in_channels=int(model_config.get("in_channels", 1)),
        channels=channels,
        head_hidden_dim=int(model_config.get("head_hidden_dim", 128)),
        head_dropout=float(model_config.get("head_dropout", 0.2)),
    )
