import numpy as np
from typing import List, Union


class ByteTokenizer:
    """
    A simple byte-level tokenizer that converts text to bytes and vice versa.
    """

    def __init__(self):
        self.vocab_size = 256  # Byte-level vocabulary

    def encode(self, text: str) -> List[int]:
        """
        Encodes text into a list of byte integers.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: The encoded byte integers.
        """
        return list(text.encode('utf-8'))

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of byte integers back into text.

        Args:
            tokens (List[int]): The tokens to decode.

        Returns:
            str: The decoded text.
        """
        return bytes(tokens).decode('utf-8', errors='ignore')

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encodes a batch of texts.

        Args:
            texts (List[str]): The texts to encode.

        Returns:
            List[List[int]]: The encoded tokens for each text.
        """
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_tokens: List[List[int]]) -> List[str]:
        """
        Decodes a batch of tokens.

        Args:
            batch_tokens (List[List[int]]): The tokens to decode.

        Returns:
            List[str]: The decoded texts.
        """
        return [self.decode(tokens) for tokens in batch_tokens]


class SubwordTokenizer:
    """
    A simple subword tokenizer using a vocabulary.
    """

    def __init__(self, vocab: dict = None):
        if vocab is None:
            # Default vocab for demonstration
            self.vocab = {
                '<unk>': 0, '<pad>': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9,
                'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19,
                's': 20, 't': 21, 'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27, ' ': 28, '.': 29,
                ',': 30, '!': 31, '?': 32, 'th': 33, 'he': 34, 'in': 35, 'er': 36, 'an': 37, 're': 38,
                'on': 39, 'at': 40, 'en': 41, 'nd': 42, 'ti': 43, 'es': 44, 'or': 45, 'te': 46, 'of': 47,
                'ed': 48, 'is': 49, 'it': 50, 'al': 51, 'ar': 52, 'st': 53, 'to': 54, 'nt': 55, 'ng': 56,
                'se': 57, 'ha': 58, 'as': 59, 'ou': 60, 'io': 61, 'le': 62, 've': 63, 'co': 64, 'me': 65,
                'de': 66, 'hi': 67, 'ri': 68, 'ro': 69, 'ic': 70, 'ne': 71, 'ea': 72, 'ra': 73, 'ce': 74,
                'li': 75, 'ch': 76, 'll': 77, 'be': 78, 'ma': 79, 'si': 80, 'om': 81, 'ur': 82, 'th': 83,
                'op': 84, 'el': 85, 'so': 86, 'se': 87, 'or': 88, 'rn': 89, 'rs': 90, 'rt': 91, 'ns': 92,
                'tr': 93, 'cr': 94, 'pr': 95, 'ey': 96, 'ay': 97, 'oy': 98, 'uy': 99, 'ly': 100, 'by': 101,
                'my': 102, 'py': 103, 'dy': 104, 'fy': 105, 'gy': 106, 'hy': 107, 'jy': 108, 'ky': 109,
                'ny': 110, 'ry': 111, 'sy': 112, 'ty': 113, 'vy': 114, 'wy': 115, 'xy': 116, 'zy': 117,
                'qu': 118, 'wh': 119, 'sh': 120, 'ph': 121, 'gh': 122, 'th': 123, 'ch': 124, 'ck': 125,
                'ng': 126, 'st': 127
            }
        else:
            self.vocab = vocab

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """
        Encodes text into subword tokens.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: The encoded subword tokens.
        """
        tokens = []
        i = 0
        while i < len(text):
            # Try to match the longest subword first
            matched = False
            for length in range(min(3, len(text) - i), 0, -1):
                subword = text[i:i+length].lower()
                if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                    i += length
                    matched = True
                    break
            if not matched:
                # Use character-level encoding for unknown subwords
                tokens.append(self.vocab.get(text[i].lower(), self.vocab['<unk>']))
                i += 1
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes subword tokens back into text.

        Args:
            tokens (List[int]): The tokens to decode.

        Returns:
            str: The decoded text.
        """
        return ''.join([self.inv_vocab.get(token, '<unk>') for token in tokens])

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encodes a batch of texts.

        Args:
            texts (List[str]): The texts to encode.

        Returns:
            List[List[int]]: The encoded tokens for each text.
        """
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_tokens: List[List[int]]) -> List[str]:
        """
        Decodes a batch of tokens.

        Args:
            batch_tokens (List[List[int]]): The tokens to decode.

        Returns:
            List[str]: The decoded texts.
        """
        return [self.decode(tokens) for tokens in batch_tokens]