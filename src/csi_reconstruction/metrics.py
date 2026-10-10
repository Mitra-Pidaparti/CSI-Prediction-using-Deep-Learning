"""CSI reconstruction metrics."""

import torch


def nmse_db(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average per-sample NMSE, converted to dB after averaging."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must both have shape (N, 2, H, W)")
    pred = torch.complex(prediction[:, 0] - 0.5, prediction[:, 1] - 0.5)
    truth = torch.complex(target[:, 0] - 0.5, target[:, 1] - 0.5)
    error = (pred - truth).abs().square().flatten(1).mean(1)
    power = truth.abs().square().flatten(1).mean(1)
    if torch.any(power == 0):
        raise ValueError("NMSE is undefined for zero-power samples")
    return 10 * torch.log10((error / power).mean())
