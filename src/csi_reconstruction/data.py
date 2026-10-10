"""Loading and scaling CSI arrays stored under the MATLAB key ``x``."""

from pathlib import Path

import numpy as np
import scipy.io
import torch


def load_mat(path: Path) -> torch.Tensor:
    contents = scipy.io.loadmat(path)
    if "x" not in contents:
        raise ValueError(f"{path} has no 'x' array")
    values = np.asarray(contents["x"])
    if values.ndim != 4 or tuple(values.shape[1:]) != (2, 32, 32):
        raise ValueError(f"{path}: expected (N, 2, 32, 32), got {values.shape}")
    if len(values) == 0:
        raise ValueError(f"{path}: x has no samples")
    if np.iscomplexobj(values):
        raise ValueError(f"{path}: x must have separate real and imaginary channels")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{path}: x must contain numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: x contains values outside the float32 range")
    return torch.from_numpy(values)


def load_directory(path: Path) -> torch.Tensor:
    files = sorted(path.glob("*.mat"))
    if not files:
        raise ValueError(f"No .mat files found in {path}")
    return torch.cat([load_mat(file) for file in files], dim=0)


def split_train_validation(data: torch.Tensor, fraction: float, seed: int):
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(data), generator=generator)
    n_val = max(1, round(len(data) * fraction))
    if n_val == len(data):
        raise ValueError("dataset needs at least one training sample")
    return data[indices[n_val:]], data[indices[:n_val]]


def training_scale(train: torch.Tensor) -> float:
    scale = float(train.abs().max())
    if scale <= 0:
        raise ValueError("training data is all zero")
    return scale


def normalize(data: torch.Tensor, scale: float) -> torch.Tensor:
    return data / scale * 0.5 + 0.5
