from pathlib import Path
import pandas as pd

import torch
import torch.nn.functional as F

import soundfile as sf
import torchaudio

from sound_model import SoundCNN

TARGET_CLASSES = [
    "crying_baby",
    "door_wood_knock",
    "siren",
    "clock_alarm",
    "glass_breaking",
]

def sound_prediction(audio_file_path):

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    sample_rate = 44_100

    audio, original_sample_rate = sf.read(
        audio_file_path, 
        dtype="float32", 
        always_2d=True
    )

    waveform = torch.from_numpy(audio).transpose(0, 1)  # Convert to tensor and transpose to (channels, samples)
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

    duration_seconds = waveform.shape[-1] / original_sample_rate

    target_sample_rate = 44_100
    target_duration = 5
    target_sample_num = (target_sample_rate * target_duration)

    if original_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate,
        )

    current_sample_num = waveform.shape[-1]
    if current_sample_num < target_sample_num:
        missing_samples = target_sample_num - current_sample_num
        waveform = F.pad(waveform, (0, missing_samples))

    elif current_sample_num > target_sample_num:
        waveform = waveform[:, :target_sample_num]

    to_spectrogram = torch.nn.Sequential(
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

    spectrogram = to_spectrogram(waveform)
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
        
    model_input = spectrogram.unsqueeze(0)
    model_input = model_input.to(device)

    checkpoint_path = Path("models/best_soundCNN_time_and_frequency_masking.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]

    model = SoundCNN(input_shape=1, hidden_units=32, output_shape=len(classes)).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    with torch.inference_mode():
        logits = model(model_input)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_index = probabilities.argmax(dim=1).item()

    predicted_class = classes[predicted_class_index]
    predicted_score = probabilities[0, predicted_class_index].item()

    class_probabilities = probabilities[0].cpu()

    return (predicted_class, predicted_score, class_probabilities)