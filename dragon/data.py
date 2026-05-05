import numpy as np
from typing import List, Tuple, Iterator
from .tokenizer import ByteTokenizer, SubwordTokenizer


class TextDataset:
    """
    A dataset class for handling text data and tokenization.
    """

    def __init__(self, file_path: str, tokenizer_type: str = 'byte', max_length: int = 512):
        """
        Initializes the TextDataset.

        Args:
            file_path (str): Path to the text file.
            tokenizer_type (str): Type of tokenizer to use ('byte' or 'subword').
            max_length (int): Maximum sequence length.
        """
        self.file_path = file_path
        self.max_length = max_length
        self.tokenizer = ByteTokenizer() if tokenizer_type == 'byte' else SubwordTokenizer()
        self.data = self._load_data()

    def _load_data(self) -> List[int]:
        """
        Loads and tokenizes the text data.

        Returns:
            List[int]: The tokenized data.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.tokenizer.encode(text)

    def __len__(self) -> int:
        """
        Returns the number of sequences in the dataset.

        Returns:
            int: Number of sequences.
        """
        return max(0, len(self.data) - self.max_length)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """
        Gets a sequence and its target.

        Args:
            idx (int): Index of the sequence.

        Returns:
            Tuple[List[int], List[int]]: Input sequence and target sequence.
        """
        seq = self.data[idx:idx+self.max_length]
        target = self.data[idx+1:idx+self.max_length+1]
        return seq, target

    def get_vocab_size(self) -> int:
        """
        Returns the vocabulary size.

        Returns:
            int: Vocabulary size.
        """
        return self.tokenizer.vocab_size


class TextDataLoader:
    """
    A data loader for batching and iterating over the TextDataset.
    """

    def __init__(self, dataset: TextDataset, batch_size: int, shuffle: bool = True):
        """
        Initializes the TextDataLoader.

        Args:
            dataset (TextDataset): The dataset to load.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterates over the dataset in batches.

        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Batches of input and target sequences.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            batch_inputs, batch_targets = zip(*batch_data)
            
            # Convert to numpy arrays
            batch_inputs = np.array(batch_inputs, dtype=np.int32)
            batch_targets = np.array(batch_targets, dtype=np.int32)
            
            yield batch_inputs, batch_targets

    def __len__(self) -> int:
        """
        Returns the number of batches.

        Returns:
            int: Number of batches.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size