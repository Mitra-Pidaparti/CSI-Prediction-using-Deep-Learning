import numpy as np
import pytest
import scipy.io
import torch

from csi_reconstruction.data import load_mat, normalize, split_train_validation, training_scale
from csi_reconstruction.metrics import nmse_db
from csi_reconstruction.model import LightweightCNN


def test_model_shape_and_compression():
    model = LightweightCNN(4)
    sample = torch.rand(2, 2, 32, 32)
    assert model.encoder(sample).shape == (2, 512)
    assert model(sample).shape == sample.shape


def test_data_scale_comes_from_training_split():
    data = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    train, val = split_train_validation(data, 0.2, 42)
    assert len(train) == 8 and len(val) == 2
    scale = training_scale(train)
    assert torch.allclose(normalize(train, scale), train / scale * 0.5 + 0.5)


def test_mat_shape_validation(tmp_path):
    good = tmp_path / "good.mat"
    bad = tmp_path / "bad.mat"
    scipy.io.savemat(good, {"x": np.zeros((2, 2, 32, 32), dtype=np.float32)})
    scipy.io.savemat(bad, {"x": np.zeros((2, 32, 32), dtype=np.float32)})
    assert load_mat(good).shape == (2, 2, 32, 32)
    with pytest.raises(ValueError, match="expected"):
        load_mat(bad)
    complex_data = tmp_path / "complex.mat"
    scipy.io.savemat(complex_data, {"x": np.ones((1, 2, 32, 32), dtype=np.complex64)})
    with pytest.raises(ValueError, match="real and imaginary"):
        load_mat(complex_data)
    too_large = tmp_path / "too_large.mat"
    scipy.io.savemat(too_large, {"x": np.full((1, 2, 32, 32), 1e100)})
    with pytest.raises(ValueError, match="float32 range"):
        load_mat(too_large)


def test_nmse_known_ratio():
    target = torch.full((1, 2, 2, 2), 1.0)
    prediction = torch.full_like(target, 0.75)
    assert nmse_db(prediction, target).item() == pytest.approx(-6.0206, abs=0.001)
