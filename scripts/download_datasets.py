#!/usr/bin/env python3
"""
Script to download and prepare larger training datasets for the BDH model.
This script provides multiple options for obtaining training data.
"""

import os
import sys
import argparse
import requests
import gzip
import shutil
from pathlib import Path
from urllib.parse import urlparse
import tempfile

def download_file(url, output_path, chunk_size=8192):
    """Download a file from a URL with progress tracking."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded_size += len(chunk)
            if total_size > 0:
                progress = (downloaded_size / total_size) * 100
                print(f"Downloaded: {downloaded_size}/{total_size} bytes ({progress:.1f}%)", end='\r')
    
    print(f"\nDownloaded: {output_path}")
    return output_path

def download_wikipedia_dump(output_dir, language='en', max_size_mb=100):
    """Download a Wikipedia dump for training."""
    # This is a sample of recent Wikipedia pages
    # For a full dump, you'd need to download from https://dumps.wikimedia.org/
    print("Downloading Wikipedia sample data...")
    
    # Download a small sample from recent Wikipedia pages
    sample_urls = [
        "https://en.wikipedia.org/api/rest_v1/page/summary/Artificial_intelligence",
        "https://en.wikipedia.org/api/rest_v1/page/summary/Machine_learning",
        "https://en.wikipedia.org/api/rest_v1/page/summary/Computer_science",
        "https://en.wikipedia.org/api/rest_v1/page/summary/Mathematics",
        "https://en.wikipedia.org/api/rest_v1/page/summary/Physics",
    ]
    
    all_text = []
    for url in sample_urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data:
                    all_text.append(data['extract'])
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    output_path = os.path.join(output_dir, "wikipedia_sample.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    print(f"Downloaded Wikipedia sample: {len(all_text)} articles")
    return output_path

def download_gutenberg_books(output_dir, num_books=5):
    """Download books from Project Gutenberg."""
    print("Downloading Project Gutenberg books...")
    
    # Public domain books from Gutenberg
    gutenberg_books = {
        'alice': 'https://www.gutenberg.org/files/11/11-0.txt',
        'sherlock': 'https://www.gutenberg.org/files/1661/1661-0.txt',
        'pride': 'https://www.gutenberg.org/files/1342/1342-0.txt',
        'moby': 'https://www.gutenberg.org/files/2701/2701-0.txt',
        'dracula': 'https://www.gutenberg.org/files/345/345-0.txt'
    }
    
    all_text = []
    for name, url in list(gutenberg_books.items())[:num_books]:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Skip Gutenberg headers/footers
                text = response.text
                lines = text.split('\n')
                content_lines = []
                in_content = False
                
                for line in lines:
                    if '*** START' in line or '***START' in line:
                        in_content = True
                        continue
                    if '*** END' in line or '***END' in line:
                        break
                    if in_content:
                        content_lines.append(line)
                
                if content_lines:
                    book_text = '\n'.join(content_lines)
                    all_text.append(f"=== {name.upper()} ===\n\n{book_text}")
                    print(f"Downloaded: {name}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    output_path = os.path.join(output_dir, "gutenberg_books.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    return output_path

def download_opensubtitles(output_dir):
    """Download OpenSubtitles data."""
    print("Downloading OpenSubtitles sample...")
    
    # Sample of OpenSubtitles data (you'd typically download the full dataset)
    sample_dialogues = [
        "Hello, how are you today?",
        "I'm doing well, thank you! How about you?",
        "I'm great! The weather is lovely today.",
        "Yes, it's perfect for a walk in the park.",
        "That sounds like a wonderful idea!",
        "Shall we go now?",
        "Yes, let's go! I'll get my coat.",
        "Don't forget your umbrella, just in case.",
        "Good thinking! You never know with this weather.",
        "Exactly! Better safe than sorry."
    ]
    
    output_path = os.path.join(output_dir, "dialogues.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_dialogues))
    
    return output_path

def create_synthetic_data(output_dir, num_samples=1000):
    """Create synthetic training data with various patterns."""
    print("Creating synthetic training data...")
    
    import random
    import string
    
    patterns = [
        # Common English patterns
        "The {adjective} {noun} {verb} {adverb}.",
        "In the {place}, the {noun} {verb} {adverb}.",
        "When the {time} comes, we shall {verb}.",
        "{Name} went to the {place} to {verb}.",
        "It was a {adjective} day for {activity}.",
        "The {color} {animal} {verb} over the {object}.",
        "{Name} said: '{quote}'",
        "After {time}, the {noun} began to {verb}.",
        "The {adjective} {object} was found in the {place}.",
        "Despite the {obstacle}, they managed to {verb}."
    ]
    
    adjectives = ["quick", "lazy", "sleepy", "noisy", "serious", "happy", "sad", "brave", "clever", "wise"]
    nouns = ["fox", "dog", "cat", "rabbit", "turtle", "bear", "wolf", "eagle", "lion", "tiger"]
    verbs = ["jumps", "runs", "sleeps", "eats", "drinks", "reads", "writes", "sings", "dances", "flies"]
    adverbs = ["quickly", "slowly", "quietly", "loudly", "happily", "sadly", "bravely", "wisely", "carefully", "eagerly"]
    places = ["park", "forest", "city", "village", "house", "garden", "beach", "mountain", "river", "lake"]
    times = ["morning", "afternoon", "evening", "night", "day", "week", "month", "year", "century", "moment"]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"]
    animals = ["fox", "dog", "cat", "rabbit", "bird", "fish", "deer", "squirrel", "mouse", "frog"]
    objects = ["table", "chair", "book", "computer", "phone", "car", "tree", "flower", "rock", "cloud"]
    obstacles = ["rain", "storm", "wind", "snow", "heat", "cold", "darkness", "distance", "time", "challenge"]
    activities = ["reading", "writing", "running", "swimming", "dancing", "singing", "learning", "exploring", "discovering", "creating"]
    quotes = [
        "Hello world!",
        "How are you?",
        "What time is it?",
        "Let's go!",
        "I love programming.",
        "The weather is nice.",
        "This is amazing!",
        "I can't believe it!",
        "What a wonderful day!",
        "Let me think about that."
    ]
    
    def replace_placeholders(pattern):
        result = pattern
        if '{adjective}' in result:
            result = result.replace('{adjective}', random.choice(adjectives), 1)
        if '{noun}' in result:
            result = result.replace('{noun}', random.choice(nouns), 1)
        if '{verb}' in result:
            result = result.replace('{verb}', random.choice(verbs), 1)
        if '{adverb}' in result:
            result = result.replace('{adverb}', random.choice(adverbs), 1)
        if '{place}' in result:
            result = result.replace('{place}', random.choice(places), 1)
        if '{time}' in result:
            result = result.replace('{time}', random.choice(times), 1)
        if '{Name}' in result:
            result = result.replace('{Name}', random.choice(names), 1)
        if '{color}' in result:
            result = result.replace('{color}', random.choice(colors), 1)
        if '{animal}' in result:
            result = result.replace('{animal}', random.choice(animals), 1)
        if '{object}' in result:
            result = result.replace('{object}', random.choice(objects), 1)
        if '{obstacle}' in result:
            result = result.replace('{obstacle}', random.choice(obstacles), 1)
        if '{activity}' in result:
            result = result.replace('{activity}', random.choice(activities), 1)
        if '{quote}' in result:
            result = result.replace('{quote}', random.choice(quotes), 1)
        return result
    
    synthetic_data = []
    for i in range(num_samples):
        pattern = random.choice(patterns)
        sentence = replace_placeholders(pattern)
        synthetic_data.append(sentence)
    
    output_path = os.path.join(output_dir, "synthetic_data.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(synthetic_data))
    
    return output_path

def combine_datasets(input_files, output_file):
    """Combine multiple dataset files into one."""
    print(f"Combining {len(input_files)} datasets...")
    
    all_text = []
    for file_path in input_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    all_text.append(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    # Analyze the combined dataset
    char_count = sum(len(text) for text in all_text)
    word_count = sum(len(text.split()) for text in all_text)
    
    print(f"Combined dataset:")
    print(f"  Files combined: {len(input_files)}")
    print(f"  Total characters: {char_count:,}")
    print(f"  Total words: {word_count:,}")
    print(f"  Output file: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Download and prepare training datasets")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for datasets")
    parser.add_argument("--datasets", type=str, nargs='+', 
                       choices=['wikipedia', 'gutenberg', 'dialogues', 'synthetic', 'all'],
                       default=['synthetic'], help="Datasets to download")
    parser.add_argument("--combine", action='store_true', help="Combine all datasets into one file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    downloaded_files = []
    
    if 'all' in args.datasets or 'wikipedia' in args.datasets:
        file_path = download_wikipedia_dump(args.output_dir)
        downloaded_files.append(file_path)
    
    if 'all' in args.datasets or 'gutenberg' in args.datasets:
        file_path = download_gutenberg_books(args.output_dir)
        downloaded_files.append(file_path)
    
    if 'all' in args.datasets or 'dialogues' in args.datasets:
        file_path = download_opensubtitles(args.output_dir)
        downloaded_files.append(file_path)
    
    if 'all' in args.datasets or 'synthetic' in args.datasets:
        file_path = create_synthetic_data(args.output_dir, num_samples=5000)
        downloaded_files.append(file_path)
    
    if args.combine and downloaded_files:
        combined_file = os.path.join(args.output_dir, "combined_dataset.txt")
        combine_datasets(downloaded_files, combined_file)
        print(f"\nCombined dataset created: {combined_file}")
    
    print(f"\nDownloaded {len(downloaded_files)} datasets to {args.output_dir}")
    print("Use 'python scripts/prepare_data.py --input data/raw/combined_dataset.txt --output data/processed' to prepare the data for training.")

if __name__ == "__main__":
    main()