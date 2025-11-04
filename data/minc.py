from typing import Tuple

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError as e:
    raise ImportError(
        "nibabel is required to read .mnc/.mnc.gz files. "
        "Install with: pip install nibabel"
    ) from e


def load_minc(path: str) -> torch.Tensor:
    """
    Load a MINC volume and return torch.FloatTensor [Z, Y, X] normalized to [0,1].

    Notes
    -----
    - nibabel 会按文件的存储顺序返回 ndarray，一般可视为 [Z, Y, X]。
    - 这里只做强度归一化；如果不同模态的物理坐标（spacing/origin/direction）
      不一致，建议先做物理空间重采样/配准（可用 SimpleITK），
      或在 main.py 里用 --resample 仅做体素形状对齐。
    """
    img = nib.load(path)                       # 支持 .mnc 和 .mnc.gz
    vol = img.get_fdata(dtype=np.float32)      # numpy: [Z, Y, X]
    vol = np.nan_to_num(vol)

    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)

    return torch.from_numpy(vol)               # torch.FloatTensor [Z, Y, X]


def resample_to_shape(vol: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Resample a [Z, Y, X] tensor to `target_shape` using trilinear interpolation.

    Parameters
    ----------
    vol : torch.Tensor
        输入体积，形状 [Z, Y, X]，值域通常在 [0,1]。
    target_shape : (int, int, int)
        目标体素尺寸 (Z, Y, X)。

    Returns
    -------
    torch.Tensor
        形状为 target_shape 的重采样结果。
    """
    if tuple(vol.shape) == tuple(target_shape):
        return vol

    v = vol[None, None]  # [1, 1, Z, Y, X]
    v = torch.nn.functional.interpolate(
        v, size=target_shape, mode="trilinear", align_corners=False
    )
    return v[0, 0]