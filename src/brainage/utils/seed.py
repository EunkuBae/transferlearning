"""Seed helpers."""

from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - optional
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - optional
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
