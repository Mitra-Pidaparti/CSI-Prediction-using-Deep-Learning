"""Train a CSI autoencoder using a fixed train/validation split."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import load_directory, normalize, split_train_validation, training_scale
from .model import LightweightCNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True, help="Directory of training .mat files")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_train, raw_val = split_train_validation(
        load_directory(args.train_dir), args.validation_fraction, args.seed
    )
    scale = training_scale(raw_train)
    train = DataLoader(normalize(raw_train, scale), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(normalize(raw_val, scale), batch_size=args.batch_size)
    model = LightweightCNN(args.compression_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val:
                batch = batch.to(device)
                loss = criterion(model(batch), batch)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite validation loss at epoch {epoch + 1}")
                val_loss += loss.item() * len(batch)
        record = {"epoch": epoch + 1, "train_mse": train_loss / len(raw_train), "val_mse": val_loss / len(raw_val)}
        history.append(record)
        print(json.dumps(record))
        if record["val_mse"] < best_val_loss:
            best_val_loss = record["val_mse"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": {name: tensor.cpu() for name, tensor in best_state.items()},
        "compression_ratio": args.compression_ratio,
        "normalization_scale": scale,
        "seed": args.seed,
        "sigmoid": True,
        "best_epoch": best_epoch,
        "validation_mse": best_val_loss,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    config = {
        "train_files": [p.name for p in sorted(args.train_dir.glob("*.mat"))],
        "compression_ratio": args.compression_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
