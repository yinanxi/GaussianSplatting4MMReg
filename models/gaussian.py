import math
import torch
import torch.nn as nn


from ops.se3 import quat_to_rotmat, normalize_quat
from ops.knn import knn_indices
from ops.alpha_blend import alpha_blend_weights 

class GaussianPrimitives(nn.Module):
    """
    Parameterization per paper:
    G_i has center μ_i ∈ R^3, covariance Σ_i = Q_i S_i S_i^T Q_i^T,
    with S_i = diag(s_i) (scale vector) and Q_i ∈ SO(3) from quaternion q_i.
    Local transform T_i ∈ SE(3) decomposed into rotation R_i (from r_i quaternion) and translation t_i.
    We store learnable tensors (requires_grad=True) for μ, s, q, r, t.
    """
    def __init__(self, centers: torch.Tensor, init_scale: float = 0.05):
        super().__init__()
        N = centers.shape[0]
        device = centers.device
        dtype = centers.dtype  # 保持和 centers 一致的精度

        # μ_i
        self.mu = nn.Parameter(centers.clone())  # [N,3]

        # s_i > 0; 用 log_s 确保正数尺度
        self.log_s = nn.Parameter(torch.full((N, 3), math.log(init_scale), device=device, dtype=dtype))

        # q_i, r_i 四元数（注意不要用 expand！）
        base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.q = nn.Parameter(base_quat.repeat(N, 1).clone().contiguous())  # [N,4] 独立存储
        self.r = nn.Parameter(base_quat.repeat(N, 1).clone().contiguous())  # [N,4] 独立存储

        # t_i 局部平移
        self.t = nn.Parameter(torch.zeros(N, 3, device=device, dtype=dtype))


    @property
    def N(self):
        return self.mu.shape[0]


    def covariance_matrices(self) -> torch.Tensor:
        """Return Σ_i for all gaussians as [N,3,3]."""
        Q = quat_to_rotmat(normalize_quat(self.q)) # [N,3,3]
        S = torch.diag_embed(self.log_s.exp()) # [N,3,3]
        Sigma = Q @ (S @ S) @ Q.transpose(-1,-2)
        return Sigma


    def local_rotations(self) -> torch.Tensor:
        return quat_to_rotmat(normalize_quat(self.r)) # [N,3,3]


    def local_translations(self) -> torch.Tensor:
        return self.t # [N,3]


    def add_gaussians(self, new_mu: torch.Tensor, new_log_s: torch.Tensor,
                      new_q: torch.Tensor, new_r: torch.Tensor, new_t: torch.Tensor):
        """Append new primitives (used by adaptive cloning)."""
        self.mu = nn.Parameter(torch.cat([self.mu, new_mu], dim=0))
        self.log_s = nn.Parameter(torch.cat([self.log_s, new_log_s], dim=0))
        self.q = nn.Parameter(torch.cat([self.q, new_q], dim=0))
        self.r = nn.Parameter(torch.cat([self.r, new_r], dim=0))
        self.t = nn.Parameter(torch.cat([self.t, new_t], dim=0))


    def remove_mask(self, keep_mask: torch.Tensor):
        """Prune primitives by boolean keep_mask [N]."""
        self.mu = nn.Parameter(self.mu[keep_mask])
        self.log_s = nn.Parameter(self.log_s[keep_mask])
        self.q = nn.Parameter(self.q[keep_mask])
        self.r = nn.Parameter(self.r[keep_mask])
        self.t = nn.Parameter(self.t[keep_mask])