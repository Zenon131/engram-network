import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .tokenizer import ByteTokenizer, SubwordTokenizer


def generate_text(model: nn.Module, tokenizer: ByteTokenizer or SubwordTokenizer, prompt: str, max_length: int, device: torch.device, temperature: float = 1.0) -> str:
    """
    Generates text using the trained model.

    Args:
        model (nn.Module): The trained model.
        tokenizer (ByteTokenizer or SubwordTokenizer): The tokenizer used for encoding/decoding.
        prompt (str): The initial prompt for generation.
        max_length (int): Maximum length of the generated text.
        device (torch.device): The device to use for generation.
        temperature (float): Temperature for sampling (lower = more deterministic).

    Returns:
        str: The generated text.
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    
    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            # Prepare input tensor
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Get model predictions
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()
            
            # Add token to sequence
            tokens.append(next_token)
            
            # Stop if end token (assuming 0 is end token)
            if next_token == 0:
                break
    
    return tokenizer.decode(tokens)


def generate_text_greedy(model: nn.Module, tokenizer: ByteTokenizer or SubwordTokenizer, prompt: str, max_length: int, device: torch.device) -> str:
    """
    Generates text using greedy decoding.

    Args:
        model (nn.Module): The trained model.
        tokenizer (ByteTokenizer or SubwordTokenizer): The tokenizer used for encoding/decoding.
        prompt (str): The initial prompt for generation.
        max_length (int): Maximum length of the generated text.
        device (torch.device): The device to use for generation.

    Returns:
        str: The generated text.
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    
    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            # Prepare input tensor
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Get model predictions
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            
            # Select the token with the highest probability
            next_token = torch.argmax(next_token_logits).item()
            
            # Add token to sequence
            tokens.append(next_token)
            
            # Stop if end token (assuming 0 is end token)
            if next_token == 0:
                break
    
    return tokenizer.decode(tokens)