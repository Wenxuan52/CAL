from typing import Optional

import torch

def soft_update(target, source, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def safe_ddpm_sample(
    policy,
    critic,
    safety_critic,
    state: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 3.0,
    step_size: float = 0.05,
    noise_scale: float = 1.0,
    safe_threshold: float = 0.0,
    guidance: bool = True,
    num_steps: Optional[int] = None,
):
    """Sample actions via DDPM with optional safety guidance.

    When ``guidance`` is ``False`` the function avoids any autograd calls and
    simply performs unconditional denoising, which is significantly faster. The
    ``num_steps`` argument allows using fewer denoising steps than the training
    horizon for cheaper sampling during target computation.
    """

    state = state.detach()
    device = state.device
    batch_size = state.size(0)
    action_dim = policy.action_dim

    step_sequence = torch.arange(policy.T - 1, -1, -1, device=device)
    if num_steps is not None and num_steps < policy.T:
        num_steps = max(1, num_steps)
        step_sequence = step_sequence[:num_steps]

    x_t = torch.randn(batch_size, action_dim, device=device)
    last_eps = None
    last_alpha_hat = None

    for step in step_sequence:
        t_value = int(step.item())
        t_tensor = torch.full((batch_size, 1), float(t_value), device=device)
        eps = policy(state, x_t, t_tensor)
        last_eps = eps
        last_alpha_hat = policy.alpha_hats[t_value]

        sqrt_alpha_hat = torch.sqrt(last_alpha_hat + 1e-8)
        sqrt_one_minus_hat = policy.sqrt_one_minus_alpha_hats[t_value]
        x0 = (x_t - sqrt_one_minus_hat * eps) / (sqrt_alpha_hat + 1e-8)
        x0 = x0.clamp(-1.0, 1.0)

        if guidance and critic is not None and safety_critic is not None:
            with torch.enable_grad():
                x0_for_grad = x0.detach().requires_grad_(True)
                q1, q2 = critic(state, x0_for_grad)
                q = torch.min(q1, q2)
                grad_q = torch.autograd.grad(
                    q.sum(), x0_for_grad, retain_graph=True, create_graph=False
                )[0]

                qh1, qh2 = safety_critic(state, x0_for_grad)
                vh = torch.min(qh1, qh2).squeeze(-1)
                safe_mask = (vh <= safe_threshold).float().unsqueeze(-1)
                qh_mean = 0.5 * (qh1 + qh2)
                grad_qh = torch.autograd.grad(
                    qh_mean.sum(), x0_for_grad, retain_graph=False, create_graph=False
                )[0]

                guidance_term = safe_mask * alpha * grad_q - (1.0 - safe_mask) * beta * grad_qh
                guided_x0 = (x0_for_grad + step_size * guidance_term).detach()
        else:
            guided_x0 = x0

        coef1 = policy.posterior_mean_coef1[t_value]
        coef2 = policy.posterior_mean_coef2[t_value]
        posterior_mean = coef1 * guided_x0 + coef2 * x_t
        variance = policy.posterior_variances[t_value]
        variance = torch.clamp(variance, min=1e-20)
        if t_value > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        x_t = posterior_mean + torch.sqrt(variance) * (noise_scale * noise)

    if step_sequence[-1] > 0 and last_eps is not None and last_alpha_hat is not None:
        final_alpha_hat = last_alpha_hat
        final_eps = last_eps
        sqrt_final_alpha_hat = torch.sqrt(final_alpha_hat + 1e-8)
        sqrt_final_one_minus = torch.sqrt(1.0 - final_alpha_hat + 1e-8)
        x0 = (x_t - sqrt_final_one_minus * final_eps) / (sqrt_final_alpha_hat + 1e-8)
    else:
        x0 = x_t

    return x0.clamp(-1.0, 1.0)