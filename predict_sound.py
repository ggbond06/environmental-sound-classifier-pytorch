from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio

import pandas as pd

import random

to_spectrogram = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=44_100,
        n_fft=1_024,
        hop_length=512,
        n_mels=64,
    ),
    torchaudio.transforms.AmplitudeToDB(
        top_db=80,
        stype="power",
    ),
)

TARGET_CLASSES = [
    "crying_baby",
    "door_wood_knock",
    "siren",
    "clock_alarm",
    "glass_breaking",
]

def waveform_to_model_input(waveform):
    spectrogram = to_spectrogram(waveform)

    # print("Before normalization:", spectrogram.shape,)

    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)

    model_input = spectrogram.unsqueeze(0)
    return model_input

class SoundCNN(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units * 2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 2,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.block_2(self.block_1(x))


metadata_file_path = Path("data/environment_sound_dataset/ESC-50/meta/esc50.csv")

metadata_df = pd.read_csv(metadata_file_path)

fold_5_target = metadata_df[
    (metadata_df["fold"] == 5)
    & (metadata_df["category"].isin(TARGET_CLASSES))
]

random_index = random.randint(0, len(fold_5_target) - 1)

row = fold_5_target.iloc[random_index]
audio_file_path = Path("data/environment_sound_dataset/ESC-50/audio") / row["filename"]

audio, sample_rate = sf.read(
    audio_file_path,
    dtype="float32",
    always_2d=True,
)

# print(f"Audio shape: {audio.shape}")
# print(f"Sample rate: {sample_rate}")

waveform = torch.from_numpy(audio).transpose(0, 1)
waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

duration_seconds = waveform.shape[-1] / sample_rate
# print("Tensor shape:", waveform.shape)
# print("Tensor dtype:", waveform.dtype)
# print("Duration:", duration_seconds, "seconds")
# print("Minimum value:", waveform.min().item())
# print("Maximum value:", waveform.max().item())

target_sample_rate = 44_100
target_duration = 5
target_sample_num = (target_sample_rate * target_duration)

if sample_rate != target_sample_rate:
    waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=sample_rate,
        new_freq=target_sample_rate,
    )

current_sample_num = waveform.shape[-1]
if current_sample_num < target_sample_num:
    missing_samples = target_sample_num - current_sample_num
    waveform = F.pad(waveform, (0, missing_samples))

elif current_sample_num > target_sample_num:
    waveform = waveform[:, :target_sample_num]

model_input = waveform_to_model_input(waveform)

device = "mps" if torch.backends.mps.is_available() else "cpu"

checkpoint_path = Path("models/best_soundCNN_time_and_frequency_masking.pt")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

classes = checkpoint["classes"]

model = SoundCNN(
    input_shape=1,
    hidden_units=32,
    output_shape=len(classes),
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# print("Loaded checkpoint epoch:", checkpoint["epoch"])
# print(
#     "Checkpoint balanced accuracy:",
#     checkpoint["balanced_accuracy"],
# )

model_input = model_input.to(device)

with torch.inference_mode():
    logits = model(model_input)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class_index = probabilities.argmax(dim=1).item()

predicted_class = classes[predicted_class_index]
predicted_score = probabilities[0, predicted_class_index].item()

# print("Logits shape:", logits.shape)
print("Predicted class:", predicted_class)
print(
    f"Prediction score: "
    f"{predicted_score * 100:.2f}%"
)

for class_name, probability in zip(
    classes,
    probabilities[0].cpu(),
):
    print(
        f"{class_name}: "
        f"{probability.item() * 100:.2f}%"
    )

print("Filename:", row["filename"])
print("Expected class:", row["category"])
print("Fold:", row["fold"])
print("Predicted class:", predicted_class)


