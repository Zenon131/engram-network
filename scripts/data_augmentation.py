#!/usr/bin/env python3
"""
Data augmentation techniques for text data to improve model training.
"""

import random
import re
from typing import List, Callable
import numpy as np

class TextAugmenter:
    """Text data augmentation class with various techniques."""
    
    def __init__(self, augmentation_prob=0.3):
        self.augmentation_prob = augmentation_prob
        
    def synonym_replacement(self, text: str, max_replacements: int = 3) -> str:
        """Replace words with their synonyms."""
        # Simple synonym dictionary (in practice, you'd use WordNet or similar)
        synonyms = {
            'good': ['excellent', 'great', 'wonderful', 'fantastic', 'superb'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'dreadful'],
            'big': ['large', 'huge', 'enormous', 'massive', 'gigantic'],
            'small': ['tiny', 'little', 'miniature', 'petite', 'compact'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'content'],
            'sad': ['unhappy', 'depressed', 'miserable', 'sorrowful', 'gloomy'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'brisk'],
            'slow': ['sluggish', 'leisurely', 'gradual', 'unhurried', 'delayed'],
            'beautiful': ['attractive', 'gorgeous', 'stunning', 'lovely', 'pretty'],
            'ugly': ['unattractive', 'hideous', 'unsightly', 'repulsive', 'grotesque']
        }
        
        words = text.split()
        replacements = 0
        
        for i, word in enumerate(words):
            if word.lower() in synonyms and random.random() < self.augmentation_prob:
                synonym = random.choice(synonyms[word.lower()])
                words[i] = synonym
                replacements += 1
                if replacements >= max_replacements:
                    break
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, max_insertions: int = 2) -> str:
        """Insert random words into the text."""
        common_words = ['very', 'really', 'quite', 'extremely', 'somewhat', 
                       'actually', 'basically', 'essentially', 'literally']
        
        words = text.split()
        insertions = 0
        
        for i in range(len(words) - 1, 0, -1):  # Insert from end to avoid index issues
            if random.random() < self.augmentation_prob and insertions < max_insertions:
                insert_word = random.choice(common_words)
                words.insert(i, insert_word)
                insertions += 1
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, max_deletions: int = 2) -> str:
        """Randomly delete words from the text."""
        words = text.split()
        if len(words) <= 3:  # Don't delete from very short texts
            return text
        
        deletions = 0
        new_words = []
        
        for word in words:
            if random.random() < self.augmentation_prob and deletions < max_deletions:
                deletions += 1
            else:
                new_words.append(word)
        
        return ' '.join(new_words) if new_words else text
    
    def random_swap(self, text: str, max_swaps: int = 2) -> str:
        """Randomly swap adjacent words."""
        words = text.split()
        if len(words) <= 2:
            return text
        
        swaps = 0
        
        for i in range(len(words) - 1):
            if random.random() < self.augmentation_prob and swaps < max_swaps:
                words[i], words[i + 1] = words[i + 1], words[i]
                swaps += 1
        
        return ' '.join(words)
    
    def back_translation_sim(self, text: str) -> str:
        """Simulate back translation by paraphrasing."""
        # Simple paraphrasing rules (in practice, you'd use a translation API)
        paraphrases = {
            r'\bI am\b': "I'm",
            r'\byou are\b': "you're",
            r'\bhe is\b': "he's",
            r'\bshe is\b': "she's",
            r'\bit is\b': "it's",
            r'\bwe are\b': "we're",
            r'\bthey are\b': "they're",
            r'\bdo not\b': "don't",
            r'\bdoes not\b': "doesn't",
            r'\bdid not\b': "didn't",
            r'\bhave not\b': "haven't",
            r'\bhas not\b': "hasn't",
            r'\bhad not\b': "hadn't",
            r'\bwill not\b': "won't",
            r'\bwould not\b': "wouldn't",
            r'\bcould not\b': "couldn't",
            r'\bshould not\b': "shouldn't",
            r'\bmust not\b': "mustn't"
        }
        
        augmented = text
        for pattern, replacement in paraphrases.items():
            if random.random() < 0.5:  # 50% chance to apply each rule
                augmented = re.sub(pattern, replacement, augmented, flags=re.IGNORECASE)
        
        return augmented
    
    def case_augmentation(self, text: str) -> str:
        """Augment text by changing case patterns."""
        if random.random() < 0.3:
            # Randomly change to uppercase for some words
            words = text.split()
            for i in range(len(words)):
                if random.random() < 0.1 and len(words[i]) > 2:  # 10% chance per word
                    words[i] = words[i].upper()
            return ' '.join(words)
        elif random.random() < 0.3:
            # Randomly change to lowercase
            return text.lower()
        else:
            return text
    
    def punctuation_augmentation(self, text: str) -> str:
        """Augment text by adding/removing punctuation."""
        if random.random() < 0.2:
            # Add random punctuation at the end
            punctuations = ['.', '!', '?', '...']
            if text and text[-1] not in punctuations:
                return text + random.choice(punctuations)
        
        if random.random() < 0.2:
            # Remove some punctuation
            text = re.sub(r'[.,!?;:]', '', text)
        
        return text
    
    def apply_all_augmentations(self, text: str) -> List[str]:
        """Apply multiple augmentation techniques to generate variations."""
        augmentations = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_deletion,
            self.random_swap,
            self.back_translation_sim,
            self.case_augmentation,
            self.punctuation_augmentation
        ]
        
        augmented_texts = [text]  # Include original
        
        # Apply each augmentation technique
        for aug_func in augmentations:
            if random.random() < 0.7:  # 70% chance to apply each technique
                try:
                    augmented = aug_func(text)
                    if augmented != text and len(augmented.split()) >= 2:
                        augmented_texts.append(augmented)
                except:
                    pass  # Skip if augmentation fails
        
        # Also apply random combinations
        for _ in range(2):
            augmented = text
            for aug_func in random.sample(augmentations, 3):  # Random 3 techniques
                try:
                    augmented = aug_func(augmented)
                except:
                    pass
            if augmented != text and len(augmented.split()) >= 2:
                augmented_texts.append(augmented)
        
        return list(set(augmented_texts))  # Remove duplicates
    
    def augment_dataset(self, input_file: str, output_file: str, augmentation_factor: int = 3):
        """Augment an entire dataset file."""
        print(f"Augmenting dataset from {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        augmented_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line.split()) >= 3:  # Only augment meaningful lines
                variations = self.apply_all_augmentations(line)
                # Take up to augmentation_factor variations per line
                augmented_lines.extend(variations[:augmentation_factor])
            else:
                augmented_lines.append(line)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(augmented_lines))
        
        print(f"Original lines: {len(lines)}")
        print(f"Augmented lines: {len(augmented_lines)}")
        print(f"Augmentation factor: {len(augmented_lines) / len(lines):.2f}x")
        print(f"Augmented dataset saved to: {output_file}")

def main():
    """Example usage of the text augmenter."""
    augmenter = TextAugmenter(augmentation_prob=0.3)
    
    # Test with sample text
    sample_text = "The quick brown fox jumps over the lazy dog. It is a beautiful day."
    
    print("Original text:", sample_text)
    print("\nAugmented variations:")
    
    variations = augmenter.apply_all_augmentations(sample_text)
    for i, variation in enumerate(variations):
        print(f"{i+1}. {variation}")
    
    # Example of augmenting a file
    # augmenter.augment_dataset('data/raw/sample.txt', 'data/raw/augmented_sample.txt')

if __name__ == "__main__":
    main()