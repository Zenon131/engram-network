#!/usr/bin/env python3
import os
import sys
from collections import Counter

def analyze_text_file(filepath):
    """Analyze a text file for basic statistics"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        line_count = text.count('\n') + 1
        unique_chars = len(set(text))
        
        # Character frequency
        char_freq = Counter(text)
        most_common_chars = char_freq.most_common(20)
        
        # Byte analysis (for tokenization)
        byte_tokens = list(text.encode('utf-8'))
        unique_bytes = len(set(byte_tokens))
        
        print(f"File: {filepath}")
        print(f"Size: {os.path.getsize(filepath):,} bytes")
        print(f"Characters: {char_count:,}")
        print(f"Words: {word_count:,}")
        print(f"Lines: {line_count:,}")
        print(f"Unique characters: {unique_chars}")
        print(f"Unique bytes: {unique_bytes}")
        print(f"Vocabulary size (byte-level): 256")
        
        print(f"\nMost common characters:")
        for char, count in most_common_chars:
            if char == '\n':
                char_repr = '\\n'
            elif char == ' ':
                char_repr = 'SPACE'
            elif char == '\t':
                char_repr = '\\t'
            else:
                char_repr = char
            print(f"  '{char_repr}': {count:,}")
        
        # Sample of text
        print(f"\nFirst 500 characters:")
        print(repr(text[:500]))
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'unique_chars': unique_chars,
            'unique_bytes': unique_bytes
        }
    
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def main():
    # Analyze current sample data
    sample_path = 'data/raw/sample.txt'
    print("=" * 80)
    print("CURRENT DATASET ANALYSIS")
    print("=" * 80)
    sample_stats = analyze_text_file(sample_path)
    
    # Check if processed data exists
    processed_path = 'data/processed/tokenized_byte.txt'
    if os.path.exists(processed_path):
        print("\n" + "=" * 80)
        print("PROCESSED DATA ANALYSIS")
        print("=" * 80)
        with open(processed_path, 'r') as f:
            processed_text = f.read()
        
        tokens = [int(x) for x in processed_text.split()]
        print(f"Processed tokens: {len(tokens):,}")
        print(f"Unique token values: {len(set(tokens))}")
        print(f"Token range: {min(tokens)} - {max(tokens)}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    if sample_stats and sample_stats['char_count'] < 100000:
        print("❌ Dataset is too small for effective training.")
        print("💡 Recommended: At least 1-10 million characters for a language model.")
    else:
        print("✅ Dataset size is reasonable.")
    
    print("\n💡 Consider downloading larger datasets like:")
    print("   - Wikipedia dumps")
    print("   - Project Gutenberg books")
    print("   - OpenWebText")
    print("   - C4 dataset")

if __name__ == "__main__":
    main()