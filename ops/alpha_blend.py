import torch


def gaussian_pdf_weight(x: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """Compute unnormalized weight \tilde{w} = 1/((2π)^{3/2}|Σ|^{1/2}) * exp(-1/2 (x-μ)^T Σ^{-1} (x-μ))
    We omit constants since normalized later. x: [M,3], mu:[M,3], Sigma:[M,3,3]
    Returns weights [M].
    """
    diff = (x - mu).unsqueeze(-1) # [M,3,1]
    invSigma = torch.inverse(Sigma) # [M,3,3]
    m = torch.matmul(torch.matmul(diff.transpose(-1,-2), invSigma), diff).squeeze(-1).squeeze(-1) # [M]
    w = torch.exp(-0.5 * m)
    return w


def alpha_blend_weights(vox_xyz: torch.Tensor, mu_knn: torch.Tensor, Sigma_knn: torch.Tensor) -> torch.Tensor:
    """Given K nearest gaussians for each voxel, compute normalized weights w_jk.
    vox_xyz: [M,3]; mu_knn: [M,K,3]; Sigma_knn:[M,K,3,3]
    Return w: [M,K] where w_jk = \tilde{w}_{jk} / sum_k \tilde{w}_{jk}
    """
    M, K, _ = mu_knn.shape
    wtilde = []
    for k in range(K):
        w_k = gaussian_pdf_weight(vox_xyz, mu_knn[:,k,:], Sigma_knn[:,k,:,:])
        wtilde.append(w_k)
    wtilde = torch.stack(wtilde, dim=-1) # [M,K]
    w = wtilde / (wtilde.sum(dim=-1, keepdim=True) + 1e-8)
    return w