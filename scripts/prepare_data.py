import os
import sys
import argparse

# Add the parent directory to Python path to find the dragon module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dragon.tokenizer import ByteTokenizer, SubwordTokenizer
from dragon.data import TextDataset


def prepare_data(input_file: str, output_dir: str, tokenizer_type: str = 'byte', max_length: int = 512):
    """
    Prepares data by tokenizing and saving to disk.

    Args:
        input_file (str): Path to the input text file.
        output_dir (str): Directory to save the processed data.
        tokenizer_type (str): Type of tokenizer to use ('byte' or 'subword').
        max_length (int): Maximum sequence length.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tokenizer
    if tokenizer_type == 'byte':
        tokenizer = ByteTokenizer()
    else:
        tokenizer = SubwordTokenizer()
    
    # Load and tokenize data
    print(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Tokenizing data...")
    tokens = tokenizer.encode(text)
    
    # Save tokenized data
    output_file = os.path.join(output_dir, f"tokenized_{tokenizer_type}.txt")
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, tokens)))
    
    print(f"Tokenized data saved to {output_file}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(tokens)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for BDH model training")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--tokenizer", type=str, choices=['byte', 'subword'], default='byte', help="Tokenizer type")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.tokenizer, args.max_length)


if __name__ == "__main__":
    main()