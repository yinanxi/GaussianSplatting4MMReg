from typing import Tuple
import torch


class PairwiseVolume(torch.utils.data.Dataset):
    """Simple holder for fixed & moving volumes already loaded as tensors in [-1,1] coords.
    Assumes volumes are [Z,Y,X] float32 in [0,1].
    """
    def __init__(self, If: torch.Tensor, Im: torch.Tensor):
        assert If.shape == Im.shape
        self.If = If
        self.Im = Im
        self.shape = If.shape


    def __len__(self):
        # number of available voxel coordinates
        return int(self.shape[0] * self.shape[1] * self.shape[2])


    def random_voxel_coords(self, M: int) -> torch.Tensor:
        Z,Y,X = self.shape
        z = torch.randint(0, Z, (M,))
        y = torch.randint(0, Y, (M,))
        x = torch.randint(0, X, (M,))
        return torch.stack([z,y,x], dim=-1)