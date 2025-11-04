import torch


def ncc_loss(I_f: torch.Tensor, I_w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Global NCC loss (negative NCC for minimization).
    I_f, I_w flattened: [M]
    """
    I_fm = I_f - I_f.mean()
    I_wm = I_w - I_w.mean()
    num = (I_fm * I_wm).sum()
    den = torch.sqrt((I_fm.square().sum() + eps) * (I_wm.square().sum() + eps))
    ncc = num / den
    return -ncc