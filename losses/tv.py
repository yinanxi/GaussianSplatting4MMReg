import torch


def minibatch_tv_loss(phi: torch.Tensor, coords_idx: torch.Tensor, img_shape: torch.Size) -> torch.Tensor:
    """Mini-batch TV regularization over sampled voxels with orthogonal neighbors.
    phi: DVF field sampled on the full grid [D,H,W,3] or [Z,Y,X,3].
    coords_idx: [S,3] integer indices of sampled voxels
    img_shape: (Z,Y,X)
    """
    Z, Y, X = img_shape
    S = coords_idx.shape[0]
    device = phi.device


    def safe_get(z,y,x):
        z = z.clamp(0, Z-1); y = y.clamp(0, Y-1); x = x.clamp(0, X-1)
        return phi[z, y, x] # [3]


    tv = 0.0
    for i in range(S):
        z,y,x = coords_idx[i]
        p = safe_get(z,y,x)
        # neighbors +/- in 3 orthogonal directions
        for dz,dy,dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            q = safe_get(z+dz,y+dy,x+dx)
            tv = tv + (p - q).abs().sum()
    return tv / (S * 6.0)