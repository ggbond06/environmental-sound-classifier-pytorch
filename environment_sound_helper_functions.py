from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sound_model import SoundCNNBatchNorm
from environment_sound_dataset import (
    ESC50Dataset,
    TARGET_CLASSES,
    CLASSES,
    CLASSES_TO_IDX,
)

def evaluate_model(model, dataloader, loss_fn, device, num_classes):
    model.eval()

    total_loss = 0
    correct_predictions = 0
    processed_samples = 0

    true_counts = torch.zeros(num_classes, dtype=torch.long)
    predicted_counts = torch.zeros(num_classes, dtype=torch.long)
    correct_counts = torch.zeros(num_classes, dtype=torch.long)

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item()
            correct_predictions += (predictions == y).sum().item()
            processed_samples += y.size(0)

            labels_cpu = y.cpu()
            predictions_cpu = predictions.cpu()

            true_counts += torch.bincount(
                labels_cpu,
                minlength=num_classes,
            )

            predicted_counts += torch.bincount(
                predictions_cpu,
                minlength=num_classes,
            )

            correct_mask = predictions_cpu == labels_cpu

            correct_counts += torch.bincount(
                labels_cpu[correct_mask],
                minlength=num_classes,
            )

    recall = correct_counts.float() / true_counts.clamp(min=1)
    precision = (
        correct_counts.float()
        / predicted_counts.clamp(min=1)
    )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": (
            correct_predictions / processed_samples
        ) * 100,
        "balanced_accuracy": recall.mean().item() * 100,
        "macro_precision": precision.mean().item() * 100,
        "recall": recall,
        "precision": precision,
    }

@torch.inference_mode()
def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

def train_and_evaluate_fold(train_folds, val_fold, test_fold, metadata_df, run_id, device):
    training_dataset = ESC50Dataset(metadata_df, folds=train_folds, augment=True)
    training_evaluation_dataset = ESC50Dataset(metadata_df, folds=train_folds, augment=False)
    validation_dataset = ESC50Dataset(metadata_df, folds=[val_fold], augment=False)
    testing_dataset = ESC50Dataset(metadata_df, folds=[test_fold], augment=False)

    training_labels = []
    for category in training_dataset.dataframe["category"]:
        if category not in TARGET_CLASSES:
            category = "other"
        training_labels.append(CLASSES_TO_IDX[category])
    training_labels = torch.tensor(training_labels)

    torch.manual_seed(42)
    model = SoundCNNBatchNorm(input_shape=1, hidden_units=32, output_shape=len(CLASSES)).to(device)

    class_counts = torch.bincount(training_labels, minlength=len(CLASSES)).float()
    sample_weight_per_class = 1.0 / torch.sqrt(class_counts)
    sample_weight_per_example = sample_weight_per_class[training_labels]

    training_sampler = WeightedRandomSampler(
        weights=sample_weight_per_example,
        num_samples=len(training_dataset),
        replacement=True,
    )

    train_dataloader = DataLoader(training_dataset, batch_size=32, sampler=training_sampler, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False, num_workers=0)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    checkpoint_directory = Path("models")
    checkpoint_directory.mkdir(exist_ok=True)
    best_model_path = checkpoint_directory / f"best_model_fold_{run_id}.pt"

    best_balanced_accuracy = -1.0
    epochs_without_improvement = 0
    min_epochs = 15
    epochs = 40
    patience = 8

    torch.manual_seed(42)

    for epoch in range(epochs):
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

        validation_metrics = evaluate_model(
            model=model, dataloader=validation_dataloader,
            loss_fn=loss_fn, device=device, num_classes=len(CLASSES)
        )
        scheduler.step(validation_metrics["balanced_accuracy"])

        current_score = validation_metrics["balanced_accuracy"]
        if current_score > best_balanced_accuracy:
            best_balanced_accuracy = current_score
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "balanced_accuracy": current_score,
                "classes": CLASSES,
            }, best_model_path)
        else:
            epochs_without_improvement += 1

        if epoch + 1 >= min_epochs and epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model=model, dataloader=test_dataloader,
        loss_fn=loss_fn, device=device, num_classes=len(CLASSES)
    )

    y_pred, y_true = get_predictions(model, test_dataloader, device)

    return {
        "test_fold": test_fold,
        "val_fold": val_fold,
        "best_epoch": checkpoint["epoch"],
        "val_balanced_accuracy": best_balanced_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "y_pred": y_pred,
        "y_true": y_true,
        "test_categories": testing_dataset.dataframe["category"].tolist(),
    }