# Lightweight CNN for CSI Feedback

This repository contains a lightweight convolutional autoencoder for compressing and reconstructing channel state information (CSI). The work uses CSI data generated in the **5G+/6G Wireless Research and Standardization Lab at IIT Kanpur**.

Each CSI sample is represented as a `2 x 32 x 32` tensor, with separate channels for its real and imaginary components. The encoder maps the input to a lower-dimensional representation, and the decoder reconstructs the original CSI.

```text
CSI input -> encoder -> compressed representation -> decoder -> reconstructed CSI
```

The experiments compare compression ratios of `4`, `8`, `16`, `32`, and `64`. Although the repository retains its original name, the current model performs CSI compression and reconstruction rather than future-channel prediction. The original notebooks remain in [`Lightweight_CNN/`](Lightweight_CNN/) for reference, while the reusable model and experiment code lives in `src/csi_reconstruction/`.

## Data

The dataset was generated in the **5G+/6G Wireless Research and Standardization Lab at IIT Kanpur** and is not included in this repository.

To run the code, place the training files in `data/train/` and the held-out test files in `data/test/`. Each MATLAB file must contain a real-valued array named `x` with shape `(samples, 2, 32, 32)`.

Training uses an 80/20 train/validation split with seed 42. A scale calculated from the training split is used consistently for training, validation, and testing.

## Installation

Create a Conda environment with Python 3.11 and install the project:

```bash
conda create -n csi-reconstruction python=3.11
conda activate csi-reconstruction
python -m pip install -e ".[test]"
```

## Training

Train the model for one compression ratio:

```bash
csi-train --train-dir data/train --output-dir runs/cr4 --compression-ratio 4 --epochs 200
```

Change `--compression-ratio` and the output directory to run experiments for the other compression ratios.

Each run saves the checkpoint from the epoch with the lowest validation MSE, along with `config.json` for the experiment settings and `history.json` for the training and validation losses.

## Evaluation

Evaluate the resulting checkpoint on the test data:

```bash
csi-evaluate --test-dir data/test --checkpoint runs/cr4/checkpoint.pt
```

Evaluation reports MSE on normalized inputs, NMSE in dB, the number of test samples, and the model parameter count. NMSE is calculated from the per-sample error-to-signal-power ratio and converted to dB.

The saved parameters under `Lightweight_CNN/Lightweight_CNN/Saved Model Parameters/` come from the original notebook experiments. The evaluation command expects a checkpoint produced by the current `csi-train` workflow because that checkpoint also stores the training scale.

## Model notes

The decoder uses two convolutional branches and follows the architecture in the combined-dataset notebook. The current implementation keeps the original final self-addition before the sigmoid so that the reorganized code remains consistent with that experiment.

The implementation currently assumes a fixed `32 x 32` CSI input and a MATLAB variable named `x`.

## Tests

```bash
pytest -q
```
