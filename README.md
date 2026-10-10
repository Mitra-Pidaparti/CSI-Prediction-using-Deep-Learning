# Lightweight CNN for CSI feedback

This project compresses channel state information (CSI) with an encoder and reconstructs it with a decoder. The input is a 32 x 32 matrix with two channels for its real and imaginary parts. The compressed vector has 1/CR as many values as the input, where CR is the compression ratio.

```text
CSI (2 x 32 x 32) -> encoder -> compressed vector -> decoder -> reconstructed CSI
```

Despite the repository name, the notebooks do not predict future CSI. They train an autoencoder for reconstruction. The original notebooks remain in [`Lightweight_CNN/`](Lightweight_CNN/) for reference; the model and experiment code now live in `src/csi_reconstruction/`.

## Data

The dataset is not included. The notebooks load local MATLAB files, but do not identify a public download or record how the files were generated. To run the code, put training files in `data/train/` and held-out test files in `data/test/`. Each file needs a real-valued `x` array shaped `(samples, 2, 32, 32)`.

Training uses an 80/20 train/validation split with seed 42. One scale calculated from the training split is used for train, validation, and test data. The notebooks calculate a separate scale for each split, so results from this code need a fresh run and should not be mixed with old notebook results.

## Setup and run

With Conda and Python 3.11:

```bash
conda create -n csi-reconstruction python=3.11
conda activate csi-reconstruction
python -m pip install -e ".[test]"
```

Train one compression ratio:

```bash
csi-train --train-dir data/train --output-dir runs/cr4 --compression-ratio 4 --epochs 200
```

Evaluate the resulting checkpoint:

```bash
csi-evaluate --test-dir data/test --checkpoint runs/cr4/checkpoint.pt
```

The checkpoint uses the epoch with the lowest validation MSE and stores the training scale. Each run also saves `config.json` and `history.json` with its settings and losses. Repeat with ratios `8`, `16`, `32`, and `64` to compare compression levels. `data/` and `runs/` are ignored by Git.

## What is measured

Evaluation prints MSE on normalized inputs, NMSE in dB, test sample count, and parameter count. NMSE is the mean of the per-sample error-to-signal-power ratios, converted to dB. The channels are shifted by 0.5 before forming complex values, as in the notebooks.

There is no results table yet because the dataset is unavailable here and the existing weight files lack the scale needed for a comparable test. The five saved weights under `Lightweight_CNN/Lightweight_CNN/Saved Model Parameters/` are retained, but `csi-evaluate` expects a new checkpoint made by `csi-train`.

The next run should record the dataset source and split, NMSE for each compression ratio, a simple baseline, and the hardware used for any latency measurement.

## Model notes

The decoder has two convolution branches. Its last operation adds a tensor to itself before the sigmoid, as in the combined-dataset notebook. That may be an unintended residual connection, but changing it would make this a different model. The other notebook omits the sigmoid. The training command follows the combined-dataset version.

The code assumes a fixed 32 x 32 input and a MATLAB `x` key. It does not include dataset acquisition, temporal prediction, baselines, or validated results.

## Tests

```bash
pytest -q
```
