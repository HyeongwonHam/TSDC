from __future__ import annotations

import torch
import torch.nn as nn


# 68 -> 192 -> 192 -> 96 -> 2. Output 0 is the mean correction mu (m/s),
# output 1 is log b of the Laplace scale.
class TSDCNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        mu = out[:, 0]
        log_b = out[:, 1].clamp(min=-5.0, max=2.0)
        return torch.stack([mu, log_b], dim=-1)


def laplace_nll(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = pred[:, :1]
    log_b = pred[:, 1:2]
    b = torch.exp(log_b).clamp_min(1e-4)
    return (log_b + torch.abs(target - mu) / b).mean()
