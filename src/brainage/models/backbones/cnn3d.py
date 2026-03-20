"""Simple 3D CNN backbone for MRI regression baselines."""

from __future__ import annotations

_IMPORT_ERROR: Exception | None = None

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    _IMPORT_ERROR = exc
    torch = None
    nn = None


def require_torch() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Model construction requires `torch`. Install it with `pip install -r requirements.txt`."
        ) from _IMPORT_ERROR


class CNN3DBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, channels: tuple[int, ...] = (16, 32, 64, 128)) -> None:
        require_torch()
        super().__init__()

        blocks: list[nn.Module] = []
        current_in = in_channels
        for current_out in channels:
            blocks.extend(
                [
                    nn.Conv3d(current_in, current_out, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(current_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2),
                ]
            )
            current_in = current_out

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.output_dim = channels[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
