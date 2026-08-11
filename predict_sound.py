from pathlib import Path

import pandas as pd

import random

from audio_preprocessing import sound_prediction

TARGET_CLASSES = [
    "crying_baby",
    "door_wood_knock"
    "siren",
    "clock_alarm",
    "glass_breaking",
]

ALL_CLASSES = ["other"] + TARGET_CLASSES

metadata_file_path = Path("data/environment_sound_dataset/ESC-50/meta/esc50.csv")

metadata_df = pd.read_csv(metadata_file_path)

fold_5_target = metadata_df[
    (metadata_df["fold"] == 5)
    & (metadata_df["category"].isin(TARGET_CLASSES))
]

random_index = random.randint(0, len(fold_5_target) - 1)

row = fold_5_target.iloc[random_index]
audio_file_path = Path("data/environment_sound_dataset/ESC-50/audio") / row["filename"]

predicted_class, predicted_score, probabilities = sound_prediction(audio_file_path)

# print("Logits shape:", logits.shape)
print("Predicted class:", predicted_class)
print(
    f"Prediction score: "
    f"{predicted_score * 100:.2f}%"
)

for class_name, probability in zip(
    ALL_CLASSES,
    probabilities,
):
    print(
        f"{class_name}: "
        f"{probability.item() * 100:.2f}%"
    )

print("Filename:", row["filename"])
print("Expected class:", row["category"])
print("Fold:", row["fold"])
print("Predicted class:", predicted_class)