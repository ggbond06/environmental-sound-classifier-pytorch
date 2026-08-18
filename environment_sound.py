import pandas as pd
import numpy as np

import torch

from pathlib import Path

from environment_sound_helper_functions import train_and_evaluate_fold

from environment_sound_dataset import (
    METADATA_PATH,
    CLASSES,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

metadata_df = pd.read_csv(METADATA_PATH)

all_folds = [1, 2, 3, 4, 5]
cv_results = []

for i, test_fold in enumerate(all_folds):
    val_fold = all_folds[(i + 1) % 5]
    train_folds = [f for f in all_folds if f not in (test_fold, val_fold)]

    print(f"\n=== Fold run {i+1}/5: train={train_folds}, val={val_fold}, test={test_fold} ===")
    result = train_and_evaluate_fold(train_folds, val_fold, test_fold, metadata_df, run_id=i, device=device)
    cv_results.append(result)
    print(f"Test balanced accuracy: {result['test_balanced_accuracy']:.2f}%")

test_bal_accs = [r["test_balanced_accuracy"] for r in cv_results]
test_accs = [r["test_accuracy"] for r in cv_results]

print(f"\nMean test balanced accuracy: {np.mean(test_bal_accs):.2f}% ± {np.std(test_bal_accs):.2f}%")
print(f"Mean test accuracy: {np.mean(test_accs):.2f}% ± {np.std(test_accs):.2f}%")

precisions = torch.stack([r["test_precision"] for r in cv_results])
recalls = torch.stack([r["test_recall"] for r in cv_results])

for idx, class_name in enumerate(CLASSES):
    print(
        f"{class_name}: "
        f"precision={precisions[:, idx].mean()*100:.2f}% ± {precisions[:, idx].std()*100:.2f}%, "
        f"recall={recalls[:, idx].mean()*100:.2f}% ± {recalls[:, idx].std()*100:.2f}%"
    )

checkpoint_directory = Path("models")
checkpoint_directory.mkdir(exist_ok=True)

pd.DataFrame(cv_results).drop(columns=["y_pred", "y_true", "test_categories"]).to_csv(
    checkpoint_directory / "cv_results.csv", index=False
)