from typing import Tuple
import torch


@torch.no_grad()
def clone_and_prune(model, tau_max: float, tau_min: float, jitter_scale_factor: float,
                    max_new: int = 10000):
    """Clone gaussians with large |∂L/∂μ| and prune with small.
    model: GaussianPrimitives
    """
    mu = model.mu
    if mu.grad is None:
        return 0, 0
    grad = mu.grad.detach() # [N,3]
    gnorm = grad.norm(dim=-1) # [N]
    N = mu.shape[0]
    device = mu.device


    # indices to clone / prune
    clone_idx = torch.nonzero(gnorm > tau_max, as_tuple=False).squeeze(-1)
    prune_idx = torch.nonzero(gnorm < tau_min, as_tuple=False).squeeze(-1)


    # prune first
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)
    keep_mask[prune_idx] = False
    model.remove_mask(keep_mask)


    # clone (cap to avoid explosion)
    clone_idx = clone_idx[clone_idx < keep_mask.sum()]
    clone_idx = clone_idx[:max_new]
    n_clone = clone_idx.numel()
    if n_clone == 0:
        return 0, prune_idx.numel()


    mu_new = model.mu[clone_idx] + torch.randn_like(model.mu[clone_idx]) * (model.log_s[clone_idx].exp() * jitter_scale_factor)
    log_s_new = model.log_s[clone_idx].clone()
    q_new = model.q[clone_idx].clone()
    r_new = model.r[clone_idx].clone()
    t_new = model.t[clone_idx].clone()


    model.add_gaussians(mu_new, log_s_new, q_new, r_new, t_new)
    return n_clone, prune_idx.numel()