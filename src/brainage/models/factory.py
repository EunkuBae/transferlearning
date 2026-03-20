"""Factory helpers for baseline models."""

from __future__ import annotations

from brainage.models.backbones.cnn3d import CNN3DBackbone, require_torch
from brainage.models.heads.regression import RegressionHead

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    torch = None
    nn = None


class CNN3DRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        head_hidden_dim: int = 128,
        head_dropout: float = 0.2,
        tabular_dim: int = 0,
        tabular_hidden_dim: int = 16,
    ) -> None:
        require_torch()
        super().__init__()
        self.backbone = CNN3DBackbone(in_channels=in_channels, channels=channels)
        self.tabular_dim = tabular_dim
        self.tabular_encoder = None
        fused_dim = self.backbone.output_dim
        if tabular_dim > 0:
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, tabular_hidden_dim),
                nn.ReLU(inplace=True),
            )
            fused_dim += tabular_hidden_dim
        self.head = RegressionHead(
            input_dim=fused_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def forward(self, x, tabular=None):
        features = self.backbone(x)
        if self.tabular_encoder is not None:
            if tabular is None:
                raise ValueError("tabular features are required for this model configuration")
            tabular_features = self.tabular_encoder(tabular)
            features = torch.cat([features, tabular_features], dim=1)
        return self.head(features)


def build_hcp_mmse_model(config: dict) -> CNN3DRegressor:
    model_config = config.get("model", {})
    channels = tuple(model_config.get("channels", [16, 32, 64, 128]))
    use_demographics = bool(model_config.get("use_demographics", False))
    return CNN3DRegressor(
        in_channels=int(model_config.get("in_channels", 1)),
        channels=channels,
        head_hidden_dim=int(model_config.get("head_hidden_dim", 128)),
        head_dropout=float(model_config.get("head_dropout", 0.2)),
        tabular_dim=2 if use_demographics else 0,
        tabular_hidden_dim=int(model_config.get("tabular_hidden_dim", 16)),
    )
