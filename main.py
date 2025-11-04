import os
import argparse
import yaml
import torch

from training.loop import train_pair
from data.minc import load_minc, resample_to_shape


def load_volume(path: str) -> torch.Tensor:
    """Load volume from .pt/.npy/.mnc/.mnc.gz and return [Z,Y,X] float32 in [0,1]."""
    path_lower = path.lower()
    if path_lower.endswith(('.mnc', '.mnc.gz')):
        vol = load_minc(path)
    elif path_lower.endswith('.pt'):
        vol = torch.load(path).float()
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
    else:
        import numpy as np
        vol = torch.from_numpy(np.load(path)).float()
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
    return vol


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fixed', required=True)
    p.add_argument('--moving', required=True)
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--device', default='auto')  # auto / cuda / cpu
    p.add_argument('--resample', action='store_true',
                   help='If set, resample moving to fixed shape')
    return p.parse_args()


def main():
    args = parse_args()

    # Load config (safe defaults)
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    out_dir = cfg.get('out_dir', 'outputs/exp_default')
    exp_name = cfg.get('exp_name', 'default')

    # Device selection (auto -> cuda if available)
    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'
    elif dev == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA not available, falling back to CPU.')
        dev = 'cpu'
    print(f'[INFO] Using device: {dev}')

    # Load volumes
    If = load_volume(args.fixed)
    Im = load_volume(args.moving)
    print(f'[INFO] Fixed  shape: {tuple(If.shape)}   ({args.fixed})')
    print(f'[INFO] Moving shape: {tuple(Im.shape)}   ({args.moving})')

    # Optional: resample moving to fixed shape
    if args.resample and Im.shape != If.shape:
        print('[INFO] Resampling moving -> fixed shape ...')
        Im = resample_to_shape(Im, tuple(If.shape))

    os.makedirs(out_dir, exist_ok=True)

    # Train
    result = train_pair(If, Im, cfg, device=dev)  # 这里传 dev，而不是 args.device
    model = result['model']

    # Save checkpoint
    out_path = os.path.join(out_dir, f'{exp_name}_model.pt')
    torch.save({'state_dict': model.state_dict(), 'shape': result['shape']}, out_path)
    print('[INFO] Saved:', out_path)


if __name__ == '__main__':
    main()
