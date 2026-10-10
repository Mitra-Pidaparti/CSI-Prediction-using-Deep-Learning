"""Evaluate a trained checkpoint on held-out CSI data."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import load_directory, normalize
from .metrics import nmse_db
from .model import LightweightCNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = LightweightCNN(checkpoint["compression_ratio"], sigmoid=checkpoint["sigmoid"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    data = normalize(load_directory(args.test_dir), checkpoint["normalization_scale"])
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=args.batch_size):
            predictions.append(model(batch))
    prediction = torch.cat(predictions)
    result = {
        "samples": len(data),
        "compression_ratio": checkpoint["compression_ratio"],
        "mse": torch.mean((prediction - data).square()).item(),
        "nmse_db": nmse_db(prediction, data).item(),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
