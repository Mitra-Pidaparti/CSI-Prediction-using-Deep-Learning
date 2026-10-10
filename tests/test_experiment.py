"""Exercise a small training run and its saved checkpoint."""

import json
import sys

import numpy as np
import scipy.io
import torch

from csi_reconstruction import evaluate, train


def test_train_then_evaluate(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(42)
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    output_dir = tmp_path / "run"
    train_dir.mkdir()
    test_dir.mkdir()
    scipy.io.savemat(
        train_dir / "train.mat",
        {"x": rng.normal(size=(4, 2, 32, 32)).astype("float32")},
    )
    scipy.io.savemat(
        test_dir / "test.mat",
        {"x": rng.normal(size=(2, 2, 32, 32)).astype("float32")},
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["csi-train", "--train-dir", str(train_dir), "--output-dir", str(output_dir),
         "--epochs", "1", "--batch-size", "2"],
    )
    train.main()
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["best_epoch"] == 1
    assert (output_dir / "history.json").is_file()
    assert json.loads((output_dir / "config.json").read_text())["train_files"] == ["train.mat"]

    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        ["csi-evaluate", "--test-dir", str(test_dir),
         "--checkpoint", str(output_dir / "checkpoint.pt")],
    )
    evaluate.main()
    result = json.loads(capsys.readouterr().out)
    assert result["samples"] == 2
    assert result["compression_ratio"] == 4
    assert np.isfinite(result["nmse_db"])
