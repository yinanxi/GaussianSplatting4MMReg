# training/loop.py
from typing import Dict, Tuple
import math
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.gaussian import GaussianPrimitives
from ops.knn import knn_indices
from ops.alpha_blend import alpha_blend_weights
from losses.ncc import ncc_loss
from losses.tv import minibatch_tv_loss
from training.adaptive_density import clone_and_prune


# -------------------------
# Helpers
# -------------------------
def grid_init_centers(shape: Tuple[int, int, int], device, frac: float) -> torch.Tensor:
    """
    Initialize μ on a coarse 3D grid in canonical [-1,1]^3 space.
    frac controls total count relative to |Ω| (num_voxels * frac).
    """
    Z, Y, X = shape
    total = max(1, int(Z * Y * X * max(0.0, float(frac))))
    side = max(1, int(round(total ** (1.0 / 3.0))))
    gz = torch.linspace(-1, 1, side, device=device)
    gy = torch.linspace(-1, 1, side, device=device)
    gx = torch.linspace(-1, 1, side, device=device)
    Gz, Gy, Gx = torch.meshgrid(gz, gy, gx, indexing="ij")
    centers = torch.stack([Gx.reshape(-1), Gy.reshape(-1), Gz.reshape(-1)], dim=-1)  # [S^3,3]
    return centers[:total]


def _indices_to_canonical(coords: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    """(z,y,x) indices -> canonical xyz in [-1,1], keep device of coords."""
    Z, Y, X = shape
    z = coords[:, 0].float() / max(1, (Z - 1)) * 2 - 1
    y = coords[:, 1].float() / max(1, (Y - 1)) * 2 - 1
    x = coords[:, 2].float() / max(1, (X - 1)) * 2 - 1
    return torch.stack([x, y, z], dim=-1)  # [M,3]


def sample_dvf_field(
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    Rloc: torch.Tensor,
    tloc: torch.Tensor,
    vox_coords: torch.Tensor,
    shape: Tuple[int, int, int],
    k: int,
) -> torch.Tensor:
    """
    Compute φ(x_j) via α-transformation-blending:
      1) map indices -> canonical xyz in [-1,1]
      2) KNN on μ
      3) α-blend local SE(3) transforms
    Return: [M,3]
    """
    xyz = _indices_to_canonical(vox_coords, shape)              # [M,3]
    _, idx = knn_indices(xyz, mu, k)                            # [M,k]

    mu_knn = mu[idx]                                            # [M,k,3]
    Sigma_knn = Sigma[idx]                                      # [M,k,3,3]
    R_knn = Rloc[idx]                                           # [M,k,3,3]
    t_knn = tloc[idx]                                           # [M,k,3]

    w = alpha_blend_weights(xyz, mu_knn, Sigma_knn)             # [M,k]

    disp_list = []
    for kk in range(idx.shape[1]):
        Rk = R_knn[:, kk, :, :]
        tk = t_knn[:, kk, :]
        diff = (xyz - mu_knn[:, kk, :])
        contrib = (Rk @ diff.unsqueeze(-1)).squeeze(-1) + tk    # [M,3]
        disp_list.append(contrib)

    disp = torch.stack(disp_list, dim=-1)                       # [M,3,k]
    phi = (disp * w.unsqueeze(1)).sum(dim=-1)                   # [M,3]
    return phi


# -------------------------
# Training loop
# -------------------------
def train_pair(If: torch.Tensor, Im: torch.Tensor, cfg: Dict, device: str = "cuda") -> Dict:
    """
    Train on a single fixed/moving pair.
    If, Im: [Z,Y,X] float32 in [0,1]
    """
    # ---- config with safe defaults ----
    total_iters = int(cfg.get("total_iters", 2000))
    batch_voxels = int(cfg.get("batch_voxels", cfg.get("samples_per_iter", 20000)))
    knn_k = int(cfg.get("knn_k", cfg.get("k", 4)))
    lambda_tv = float(cfg.get("lambda_tv", 15.0))
    ncc_eps = float(cfg.get("ncc_epsilon", 1e-5))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    warmup_iters = int(cfg.get("warmup_iters", 0))
    clone_every = int(cfg.get("clone_every", 100))
    tau_max = float(cfg.get("tau_max", 2e-3))
    tau_min = float(cfg.get("tau_min", 1e-7))  # 兼容你之前的键名问题
    jitter_scale_factor = float(cfg.get("jitter_scale_factor", 0.25))
    max_new_per_step = int(cfg.get("max_new_per_step", 10000))

    iter_switch = int(cfg.get("iter_switch", total_iters // 2))
    stage1_factor = float(cfg.get("max_gaussians_stage1_factor", 0.125))
    stage2_factor = float(cfg.get("max_gaussians_stage2_factor", 0.25))
    init_frac = float(cfg.get("init_frac", 0.01))
    init_scale = float(cfg.get("init_scale", 0.05))

    # ---- move volumes to target device ----
    If, Im = If.to(device, copy=True), Im.to(device, copy=True)
    shape: Tuple[int, int, int] = tuple(If.shape)
    dev = If.device  # anchor device for all tensors from now on

    # ---- model & optim ----
    centers = grid_init_centers(shape, dev, init_frac)
    model = GaussianPrimitives(centers, init_scale=init_scale).to(dev)

    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=total_iters)

    def max_gaussians_cap(iter_idx: int) -> int:
        factor = stage2_factor if iter_idx >= iter_switch else stage1_factor
        return max(1, int((shape[0] * shape[1] * shape[2]) * factor))

    # ---- train ----
    for it in range(total_iters):
        opt.zero_grad(set_to_none=True)

        # sample voxels on the SAME device as If
        z = torch.randint(0, shape[0], (batch_voxels,), device=dev)
        y = torch.randint(0, shape[1], (batch_voxels,), device=dev)
        x = torch.randint(0, shape[2], (batch_voxels,), device=dev)
        coords = torch.stack([z, y, x], dim=-1)  # [B,3]

        # gaussian params
        Sigma = model.covariance_matrices()
        Rloc = model.local_rotations()
        tloc = model.local_translations()

        # φ at sampled coords
        phi = sample_dvf_field(model.mu, Sigma, Rloc, tloc, coords, shape, knn_k)  # [B,3]

        # canonical disp -> index-space disp
        dz = phi[:, 2] * (shape[0] - 1) / 2.0
        dy = phi[:, 1] * (shape[1] - 1) / 2.0
        dx = phi[:, 0] * (shape[2] - 1) / 2.0
        z_w = (z.float() + dz).round().clamp(0, shape[0] - 1).long()
        y_w = (y.float() + dy).round().clamp(0, shape[1] - 1).long()
        x_w = (x.float() + dx).round().clamp(0, shape[2] - 1).long()

        # sample intensities
        If_s = If[z, y, x]
        Iw_s = Im[z_w, y_w, x_w]

        # losses
        Ls = ncc_loss(If_s, Iw_s, eps=ncc_eps)

        # mini-batch TV: take ~1/(D+1) * B points (D=3 -> /4)
        S = max(1, batch_voxels // 4)
        zt = torch.randint(0, shape[0], (S,), device=dev)
        yt = torch.randint(0, shape[1], (S,), device=dev)
        xt = torch.randint(0, shape[2], (S,), device=dev)
        coords_tv = torch.stack([zt, yt, xt], dim=-1)

        phi_grid = torch.zeros(shape + (3,), device=dev)
        phi_grid[z, y, x] = phi
        Lr = minibatch_tv_loss(phi_grid, coords_tv, shape)

        L = Ls + lambda_tv * Lr
        L.backward()
        opt.step()
        sched.step()

        # adaptive density
        if it >= warmup_iters and (it % clone_every == 0):
            n_clone, n_prune = clone_and_prune(
                model,
                tau_max=tau_max,
                tau_min=tau_min,
                jitter_scale_factor=jitter_scale_factor,
                max_new=max_new_per_step,
            )
            # enforce cap (stage-wise)
            cap = max_gaussians_cap(it)
            if model.N > cap:
                keep = torch.arange(model.N, device=dev)[:cap]
                mask = torch.zeros(model.N, dtype=torch.bool, device=dev)
                mask[keep] = True
                model.remove_mask(mask)

        if (it + 1) % int(cfg.get("save_every", 200)) == 0:
            print(
                f"[iter {it+1:4d}/{total_iters}] "
                f"L={L.item():.4f} (Ls={Ls.item():.4f}, Lr={Lr.item():.4f})  N={model.N}"
            )

    return {"model": model, "shape": shape}
