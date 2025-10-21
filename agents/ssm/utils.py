import torch

def soft_update(target, source, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def compute_vh(safety_critic, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Compute the approximated Hamilton-Jacobi value V_h(s)."""
    qh1, qh2 = safety_critic(state, action)
    vh = torch.min(qh1, qh2)
    return vh.squeeze(-1)


def safe_ddpm_sample(
    policy,
    critic,
    safety_critic,
    state: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 3.0,
    step_size: float = 0.05,
    noise_scale: float = 1e-2,
    safe_threshold: float = 0.0,
):
    state = state.detach()
    device = state.device
    batch_size = state.size(0)
    action_dim = policy.action_dim
    x_t = torch.randn(batch_size, action_dim, device=device)

    for step in reversed(range(policy.T)):
        t_tensor = torch.full((batch_size, 1), float(step), device=device)
        eps = policy(state, x_t, t_tensor)
        alpha_hat = policy.alpha_hats[step]
        sqrt_alpha_hat = torch.sqrt(alpha_hat)
        sqrt_one_minus = torch.sqrt(1 - alpha_hat)

        x0 = (x_t - sqrt_one_minus * eps) / (sqrt_alpha_hat + 1e-8)
        x0 = x0.clamp(-1.0, 1.0)

        with torch.enable_grad():
            x0 = x0.detach().requires_grad_(True)
            q1, q2 = critic(state, x0)
            q = torch.min(q1, q2)
            grad_q = torch.autograd.grad(q.sum(), x0, retain_graph=True, create_graph=False)[0]

            qh1, qh2 = safety_critic(state, x0)
            vh = torch.min(qh1, qh2)
            safe_mask = (vh <= safe_threshold).float()
            qh_mean = 0.5 * (qh1 + qh2)
            grad_qh = torch.autograd.grad(qh_mean.sum(), x0, retain_graph=False, create_graph=False)[0]

        guidance = safe_mask * alpha * grad_q - (1.0 - safe_mask) * beta * grad_qh
        mu = sqrt_alpha_hat * x0.detach() + sqrt_one_minus * eps
        x_t = (mu + step_size * guidance + noise_scale * torch.randn_like(x_t)).detach()

    return x_t.clamp(-1.0, 1.0)
