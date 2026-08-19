# Environmental Sound Classifier with PyTorch

A convolutional neural network that classifies five environmental sound recordings into five target categories and an additional "other" category. 

This project includes audio processing, data augmentation, class weight-imbalancing handling, checkpoint, early stopping, five fold cross validation evaluation and inference on new WAV recordings.

## Supported classes
- Crying Baby
- Door Knock
- Siren
- Clock Alarm
- Glass Breaking
- Other

The CNN model uses the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50), which contains 2000 recordings, each lasting five seconds long across 50 total categories. From these categories, five categories are selected as targets, while the remaining 45 categories are put into "other" category.

## Results

The model was evaluated using five rotating train/validation/test runs based on ESC-50's predefined folds. Each fold was used only once as the test fold.

| Metric | Five-fold result |
|---|---:|
| Balanced accuracy | **81.35% ± 3.75%** |
| Overall accuracy | **85.10% ± 2.26%** |

## Per-class results

| Class | Precision | Recall |
|---|---:|---:|
| Other | 98.31% ± 1.12% | 85.61% ± 3.16% |
| Crying baby | 54.35% ± 31.69% | 75.00% ± 26.52% |
| Door knock | 41.86% ± 16.30% | 95.00% ± 11.18% |
| Siren | 38.30% ± 18.95% | 70.00% ± 25.92% |
| Clock alarm | 61.14% ± 19.60% | 85.00% ± 13.69% |
| Glass breaking | 34.97% ± 17.98% | 77.50% ± 22.36% |

## Audio preprocessing pipeline

1. Load the WAV recordings
2. Convert sterep recordings into mono
3. Resample to 44.1 kHz when necessary
4. Add or reduce the recording length to exactly 5 seconds
5. Convert the waveform into a 64-bin log-Mel spectrogram
6. Standardize the spectrogram to zero mean and unit variance
7. Pass the spectrogram through the CNN

Training augmentations include:
- Random time shifting
- Frequency masking
- Weight sampling to reduce the effect of class imbalance on the target classes

## Model architecture

The model uses three convolutional blocks. Each block contains:
- 2D convolution
- Batch normalization
- ReLU activation function
- Max pooling

The convolutional blocks are then followed by:
- Adaptive global average pooling
- Dropout
- Linear six-class output layer

# Training and evaluation

For each of the five evaluation runs:

- Three folds are used for training.
- One fold is used for validation and early stopping.
- One fold is held out for testing.
- A new model and optimizer are initialized.
- The best checkpoint is selected using validation balanced accuracy.
- The learning rate is adjusted with `ReduceLROnPlateau`.

## Installation

Clone the repository:

```bash
git clone https://github.com/ggbond06/environmental-sound-classifier-pytorch.git
cd environmental-sound-classifier-pytorch
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the five-fold evaluation from the repository root:

```bash
python environment_sound.py
```

## Technologies

- Python
- PyTorch
- torchaudio
- SoundFile
- pandas
- NumPy
- scikit-learn
- Matplotlib
