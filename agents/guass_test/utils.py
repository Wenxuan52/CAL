"""Utility helpers for the temporary Gauss-policy based SSM experiments."""

from __future__ import annotations

import torch


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Perform Polyak averaging between ``source`` and ``target`` modules."""

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def soft_gate(qc_value: torch.Tensor, kappa: float = 0.0, alpha: float = 5.0) -> torch.Tensor:
    """Smooth gate that interpolates between reward maximisation and safety guidance."""

    return torch.sigmoid(alpha * (kappa - qc_value))

