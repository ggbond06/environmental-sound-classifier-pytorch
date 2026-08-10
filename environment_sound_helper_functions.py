import torch

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