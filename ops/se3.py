import torch


def normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    # q = [w,x,y,z]
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        ww + xx - yy - zz, 2*(xy - wz), 2*(xz + wy),
        2*(xy + wz), ww - xx + yy - zz, 2*(yz - wx),
        2*(xz - wy), 2*(yz + wx), ww - xx - yy + zz
    ], dim=-1)
    return R.reshape(q.shape[:-1] + (3,3))