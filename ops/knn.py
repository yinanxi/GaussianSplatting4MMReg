from typing import Tuple
import torch


def knn_indices(query_xyz: torch.Tensor, ref_xyz: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (dists, idx) of KNN using torch.cdist (batchless).
    query_xyz: [M,3], ref_xyz: [N,3]
    dists: [M,k], idx: [M,k]
    """
    with torch.no_grad():
        D = torch.cdist(query_xyz, ref_xyz) # [M,N]
        dists, idx = torch.topk(D, k, dim=-1, largest=False, sorted=True)
    return dists, idx