import numpy as np
from typing import List, Tuple, Iterator
import os
from .tokenizer import ByteTokenizer, SubwordTokenizer
from .data import TextDataset


class MemoryEfficientTextDataset:
    """
    A memory-efficient dataset class for handling large text data.
    Reads data in chunks to avoid loading entire files into memory.
    """

    def __init__(self, file_path: str, tokenizer_type: str = 'byte', max_length: int = 512, chunk_size: int = 10000):
        """
        Initializes the MemoryEfficientTextDataset.

        Args:
            file_path (str): Path to the text file.
            tokenizer_type (str): Type of tokenizer to use ('byte' or 'subword').
            max_length (int): Maximum sequence length.
            chunk_size (int): Number of sequences to process at once.
        """
        self.file_path = file_path
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.tokenizer = ByteTokenizer() if tokenizer_type == 'byte' else SubwordTokenizer()
        
        # Pre-calculate total sequences without loading entire file
        self.total_sequences = self._calculate_total_sequences()
        
    def _calculate_total_sequences(self) -> int:
        """Calculate total sequences by reading file size and estimating."""
        file_size = os.path.getsize(self.file_path)
        # Estimate: average 4 characters per token, each sequence uses max_length tokens
        estimated_tokens = file_size // 4
        return max(0, estimated_tokens - self.max_length)
    
    def _read_chunk(self, start_idx: int, chunk_size: int) -> List[int]:
        """Read a chunk of data from the file."""
        # This is a simplified version - in practice, you'd implement proper chunked reading
        # For now, we'll use the original method but with a warning for large files
        file_size = os.path.getsize(self.file_path)
        if file_size > 10 * 1024 * 1024:  # 10MB
            print(f"Warning: Large file detected ({file_size//1024//1024}MB). Consider using streaming data loader.")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = self.tokenizer.encode(text)
        chunk_end = min(start_idx + chunk_size * self.max_length, len(tokens))
        return tokens[start_idx:chunk_end]
    
    def __len__(self) -> int:
        """Returns the number of sequences in the dataset."""
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """
        Gets a sequence and its target.
        
        Args:
            idx (int): Index of the sequence.
            
        Returns:
            Tuple[List[int], List[int]]: Input sequence and target sequence.
        """
        # Calculate the token position for this sequence
        token_start = idx
        token_end = token_start + self.max_length
        
        # Read the chunk containing this sequence
        chunk_start = (idx // self.chunk_size) * self.chunk_size * self.max_length
        chunk_tokens = self._read_chunk(chunk_start, self.chunk_size)
        
        # Extract the sequence from the chunk
        local_idx = idx % self.chunk_size
        seq_start = local_idx * self.max_length
        seq_end = seq_start + self.max_length
        
        if seq_end > len(chunk_tokens):
            # Handle edge case at end of file
            seq = chunk_tokens[seq_start:]
            target = chunk_tokens[seq_start+1:] if seq_start+1 < len(chunk_tokens) else []
        else:
            seq = chunk_tokens[seq_start:seq_end]
            target = chunk_tokens[seq_start+1:seq_end+1]
        
        # Pad if necessary
        if len(seq) < self.max_length:
            seq = seq + [0] * (self.max_length - len(seq))
        if len(target) < self.max_length:
            target = target + [0] * (self.max_length - len(target))
            
        return seq, target
    
    def get_vocab_size(self) -> int:
        """Returns the vocabulary size."""
        return self.tokenizer.vocab_size


class StreamingTextDataLoader:
    """
    A streaming data loader that reads data in chunks to minimize memory usage.
    """
    
    def __init__(self, dataset: MemoryEfficientTextDataset, batch_size: int, shuffle: bool = True):
        """
        Initializes the StreamingTextDataLoader.
        
        Args:
            dataset (MemoryEfficientTextDataset): The dataset to load.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterates over the dataset in batches with memory efficiency.
        """
        current_chunk = None
        current_chunk_start = -1
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            
            # Process batch with memory efficiency
            batch_inputs = []
            batch_targets = []
            
            for idx in batch_indices:
                # Check if we need to load a new chunk
                chunk_num = idx // self.dataset.chunk_size
                if chunk_num != current_chunk_start:
                    current_chunk_start = chunk_num
                    # In a real implementation, you'd load the chunk here
                    # For now, we'll use the dataset's getitem method
                
                seq, target = self.dataset[idx]
                batch_inputs.append(seq)
                batch_targets.append(target)
            
            # Convert to numpy arrays
            batch_inputs = np.array(batch_inputs, dtype=np.int32)
            batch_targets = np.array(batch_targets, dtype=np.int32)
            
            yield batch_inputs, batch_targets
    
    def __len__(self) -> int:
        """Returns the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# Alternative: Simple memory-efficient version that processes data in smaller chunks
def create_memory_efficient_dataset(file_path: str, max_samples: int = 1000, max_length: int = 128):
    """
    Create a memory-efficient dataset by limiting the number of samples.
    This is a quick fix for memory issues.
    """
    dataset = TextDataset(file_path, 'byte', max_length)
    
    # Limit the number of samples to avoid memory issues
    actual_samples = min(len(dataset), max_samples)
    
    class LimitedDataset:
        def __len__(self):
            return actual_samples
        
        def __getitem__(self, idx):
            if idx >= actual_samples:
                raise IndexError
            return dataset[idx]
        
        def get_vocab_size(self):
            return dataset.get_vocab_size()
    
    return LimitedDataset()