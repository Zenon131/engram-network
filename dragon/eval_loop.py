import torch
import torch.nn as nn
from typing import Tuple
from .data import TextDataLoader


def evaluate_model(model: nn.Module, data_loader: TextDataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on the given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (TextDataLoader): The data loader.
        device (torch.device): The device to use for evaluation.

    Returns:
        Tuple[float, float]: Average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = torch.tensor(inputs, dtype=torch.long).to(device)
            targets = torch.tensor(targets, dtype=torch.long).to(device)

            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            correct += (preds.view(-1) == targets.view(-1)).sum().item()
            total += targets.numel()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def print_evaluation_results(model: nn.Module, data_loader: TextDataLoader, device: torch.device) -> None:
    """
    Prints the evaluation results.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (TextDataLoader): The data loader.
        device (torch.device): The device to use for evaluation.
    """
    avg_loss, accuracy = evaluate_model(model, data_loader, device)
    print(f"Evaluation Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")