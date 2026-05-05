import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any
from .data import TextDataLoader


def train_step(model: nn.Module, data_loader: TextDataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The model to train.
        data_loader (TextDataLoader): The data loader.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device to use for training.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = torch.tensor(inputs, dtype=torch.long).to(device)
        targets = torch.tensor(targets, dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        # Reshape logits and targets to match
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def train_model(model: nn.Module, data_loader: TextDataLoader, num_epochs: int, learning_rate: float, device: torch.device) -> None:
    """
    Trains the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to train.
        data_loader (TextDataLoader): The data loader.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): The device to use for training.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        avg_loss = train_step(model, data_loader, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")