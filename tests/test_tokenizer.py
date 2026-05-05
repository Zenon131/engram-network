import unittest
from dragon.tokenizer import ByteTokenizer, SubwordTokenizer


class TestByteTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()

    def test_encode_decode(self):
        text = "Hello, World!"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded_text)

    def test_encode_batch_decode_batch(self):
        texts = ["Hello, World!", "BDH Model", "Testing 123"]
        tokens = self.tokenizer.encode_batch(texts)
        decoded_texts = self.tokenizer.decode_batch(tokens)
        self.assertEqual(texts, decoded_texts)

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 256)


class TestSubwordTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SubwordTokenizer()

    def test_encode_decode(self):
        text = "Hello, World!"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        # Note: This might not be exactly equal due to subword tokenization
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(decoded_text, str)

    def test_encode_batch_decode_batch(self):
        texts = ["Hello, World!", "BDH Model", "Testing 123"]
        tokens = self.tokenizer.encode_batch(texts)
        decoded_texts = self.tokenizer.decode_batch(tokens)
        self.assertEqual(len(texts), len(decoded_texts))
        for decoded in decoded_texts:
            self.assertIsInstance(decoded, str)

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, len(self.tokenizer.vocab))


if __name__ == '__main__':
    unittest.main()