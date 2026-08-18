import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from torchvision import datasets

import torchaudio

import soundfile as sf

from torchvision.transforms import ToTensor

from pathlib import Path

from tqdm import tqdm

from environment_sound_helper_functions import evaluate_model, get_predictions

from sound_model import SoundCNNBatchNorm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = "mps" if torch.backends.mps.is_available() else "cpu"

DATA_ROOT = Path("data/environment_sound_dataset/ESC-50")
AUDIO_DIR = DATA_ROOT / "audio"
METADATA_PATH = DATA_ROOT / "meta" / "esc50.csv"
metadata_df = pd.read_csv(METADATA_PATH)

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

training_dataset = ESC50Dataset(
    dataframe=metadata_df,
    folds=[1, 2, 3],
    augment=True
)
training_evaluation_dataset = ESC50Dataset(
    dataframe=metadata_df,
    folds=[1, 2, 3],
    augment=False
)
validation_dataset = ESC50Dataset(
    dataframe=metadata_df,
    folds=[4],
    augment=False
)
testing_dataset = ESC50Dataset(
    dataframe=metadata_df,
    folds=[5],
    augment=False
)

training_labels = []

for category in training_dataset.dataframe["category"]:
    if category not in TARGET_CLASSES:
        category = "other"
    training_labels.append(CLASSES_TO_IDX[category])

training_labels = torch.tensor(training_labels)

torch.manual_seed(42)

model = SoundCNNBatchNorm(input_shape=1, hidden_units=32, output_shape=len(CLASSES), dropout=0.4).to(device)

class_counts = torch.bincount(
    training_labels,
    minlength=len(CLASSES)
).float()
print("Class counts:", class_counts)

sample_weight_per_class = 1.0 / torch.sqrt(class_counts)
sample_weight_per_example = (sample_weight_per_class[training_labels])

training_sampler = WeightedRandomSampler(
    weights=sample_weight_per_example,
    num_samples=len(training_dataset),
    replacement=True,
)

train_dataloader = DataLoader(training_dataset, batch_size=32, sampler=training_sampler, num_workers=0)
train_evaluation_dataloader = DataLoader(training_evaluation_dataset, batch_size=32, shuffle=False, num_workers=0)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
test_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False, num_workers=0)

X, y = next(iter(train_dataloader))
with torch.inference_mode():
    output = model(X.to(device))

print(output.shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=1e-4)

torch.manual_seed(42)

checkpoint_directory = Path("models")
checkpoint_directory.mkdir(exist_ok=True)

best_model_path = (checkpoint_directory / "best_SoundCNNBatchNorm.pt")

best_balanced_accuracy = -1.0
history = []

epochs = 40
patience = 8
epochs_without_improvement = 0

for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    correct_predictions = 0
    number_of_processed_samples = 0
    prediction_counts = torch.zeros(len(CLASSES), dtype=torch.long)
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == y).sum().item()
        number_of_processed_samples += y.size(0)
        prediction_counts += torch.bincount(predictions.detach().cpu(), minlength=len(CLASSES))
        
    avg_loss = total_loss / len(train_dataloader)
    accuracy = (correct_predictions / number_of_processed_samples) * 100

    for class_name, count in zip(CLASSES, prediction_counts):
        print(f"Predictions for class '{class_name}': {count.item()}")
    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {avg_loss:.4f} "
        f"- Accuracy: {accuracy:.2f}%"
    )

    validation_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        loss_fn=loss_fn,
        device=device,
        num_classes=len(CLASSES)
    )

    history.append({
        "epoch": epoch + 1,
        "training_loss": avg_loss,
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_balanced_accuracy": validation_metrics[
            "balanced_accuracy"
        ],
    })

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Train loss: {avg_loss:.4f} "
        f"- Validation loss: "
        f"{validation_metrics['loss']:.4f} "
        f"- Validation balanced accuracy: "
        f"{validation_metrics['balanced_accuracy']:.2f}%"
    )

    current_score = validation_metrics[
        "balanced_accuracy"
    ]

    if current_score > best_balanced_accuracy:
        best_balanced_accuracy = current_score
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "balanced_accuracy": current_score,
                "classes": CLASSES,
            },
            best_model_path,
        )

        print(f"Saved new best model: {current_score:.2f}%") 

    else:
        epochs_without_improvement += 1
        print(
            f"No validation improvement for "
            f"{epochs_without_improvement}/{patience} epochs"
        )

    if epochs_without_improvement >= patience:
        print(
            f"Early stopping at epoch {epoch + 1}. "
            f"Best balanced accuracy: "
            f"{best_balanced_accuracy:.2f}%"
        )
        break 

if history:
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        checkpoint_directory / "training_history.csv",
        index=False,
    )

checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

assert checkpoint["classes"] == CLASSES

test_metrics = evaluate_model(
    model=model,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    device=device,
    num_classes=len(CLASSES),
) 

print(
    "Loaded best epoch:",
    checkpoint["epoch"],
)

print(
    "Best balanced accuracy:",
    checkpoint["balanced_accuracy"],
)

print(
    f"Test results - "
    f"Loss: {test_metrics['loss']:.4f} - "
    f"Accuracy: {test_metrics['accuracy']:.2f}% - "
    f"Balanced accuracy: "
    f"{test_metrics['balanced_accuracy']:.2f}%"
)

for index, class_name in enumerate(CLASSES):
    print(
        f"{class_name}: "
        f"precision="
        f"{test_metrics['precision'][index].item() * 100:.2f}%, "
        f"recall="
        f"{test_metrics['recall'][index].item() * 100:.2f}%"
    )

y_pred, y_true = get_predictions(
    model=model,
    dataloader=test_dataloader,
    device=device
)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig(checkpoint_directory / "confusion_matrix.png")
plt.show()

results_df = test_dataloader.dataset.dataframe.copy()
results_df["true_label"] = [CLASSES[label] for label in y_true]
results_df["predicted_label"] = [CLASSES[label] for label in y_pred]

assert (
    results_df["true_label"]
    == results_df["category"].apply(lambda c: c if c in TARGET_CLASSES else "other")
).all()

glass_confusions = results_df[
    (results_df["true_label"] == "other") &
    (results_df["predicted_label"] == "glass_breaking")
]
print(glass_confusions["category"].value_counts())

siren_confusions = results_df[
    (results_df["true_label"] == "other") &
    (results_df["predicted_label"] == "siren")
]
print(siren_confusions["category"].value_counts())