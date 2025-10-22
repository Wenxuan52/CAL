import torch


def soft_update(target, source, tau):
    """Polyak averaging for target network updates."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def soft_gate(qc_value, kappa=0.0, alpha=5.0):
    """Differentiable approximation of the safety indicator."""
    return torch.sigmoid(alpha * (kappa - qc_value))
