import pandas as pd
from pathlib import Path
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


DATA_ROOT = Path("data/environment_sound_dataset/ESC-50")
AUDIO_DIR = DATA_ROOT / "audio"
METADATA_PATH = DATA_ROOT / "meta" / "esc50.csv"

TARGET_CLASSES = [
    "crying_baby",
    "door_wood_knock",
    "siren",
    "clock_alarm",
    "glass_breaking",
]

CLASSES = ["other"] + TARGET_CLASSES
CLASSES_TO_IDX = {
    class_name: index for index, class_name in enumerate(CLASSES)
}

class ESC50Dataset(Dataset):
    def __init__(self, dataframe, folds, augment=False):
        self.dataframe = dataframe[dataframe["fold"].isin(folds)].reset_index(drop=True)

        self.to_spectrogram = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=44100,
                n_fft=1024,
                hop_length=512,
                n_mels=64,
            ),
            torchaudio.transforms.AmplitudeToDB(
                top_db=80,
                stype="power",
            )
        )

        self.augment = augment

        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=25)

        self.frequency_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=6)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # Load audio
        audio_path = AUDIO_DIR / row["filename"]
        audio, _ = sf.read(audio_path, dtype="float32")

        # [samples] -> [channel, samples]
        waveform = torch.from_numpy(audio).unsqueeze(0)

        if self.augment and torch.rand(1).item() < 0.5:
            maximum_shift = int(44_100 * 0.5)
        
            shift = torch.randint(
                -maximum_shift,
                maximum_shift + 1,
                size=(1,),
            ).item()
        
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        # Convert audio into a spectrogram
        spectrogram = self.to_spectrogram(waveform)

        spectrogram = (
            spectrogram - spectrogram.mean()
        ) / (spectrogram.std() + 1e-6)  # Normalize the spectrogram

        if self.augment and torch.rand(1).item() < 0.5:
            spectrogram = self.time_masking(spectrogram)

        if self.augment and torch.rand(1).item() < 0.5:
            spectrogram = self.frequency_masking(spectrogram)
        
        # Determine the label
        category = row["category"]

        if category not in TARGET_CLASSES:
            category = "other"

        label = CLASSES_TO_IDX[category]

        return spectrogram, label