#!/usr/bin/env python3
"""
Data validation and cleaning utilities for text datasets.
"""

import re
import string
from typing import List, Tuple, Dict, Set
from collections import Counter
import os

class DataValidator:
    """Data validation class for text datasets."""
    
    def __init__(self):
        self.common_issues = {
            'encoding': 'Detect and fix encoding issues',
            'whitespace': 'Remove extra whitespace',
            'special_chars': 'Handle special characters',
            'duplicates': 'Remove duplicate lines',
            'length': 'Filter by length',
            'language': 'Detect language',
            'profanity': 'Filter profanity'
        }
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect the encoding of a file."""
        # Simple encoding detection - try common encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()
                return encoding
            except UnicodeDecodeError:
                continue
        return 'unknown'
    
    def fix_encoding(self, file_path: str, output_path: str, target_encoding: str = 'utf-8') -> bool:
        """Fix encoding issues in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            with open(output_path, 'w', encoding=target_encoding) as f:
                f.write(content)
            
            return True
        except Exception as e:
            print(f"Error fixing encoding: {e}")
            return False
    
    def clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace from text."""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        return text.strip()
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters from text."""
        if keep_punctuation:
            # Keep basic punctuation
            allowed_chars = string.ascii_letters + string.digits + ' .,!?;:()[]{}"\'-'
            return ''.join(c for c in text if c in allowed_chars)
        else:
            # Keep only alphanumeric and spaces
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def remove_duplicates(self, lines: List[str]) -> List[str]:
        """Remove duplicate lines while preserving order."""
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        return unique_lines
    
    def filter_by_length(self, lines: List[str], min_length: int = 10, max_length: int = 1000) -> List[str]:
        """Filter lines by character length."""
        return [line for line in lines if min_length <= len(line) <= max_length]
    
    def filter_by_word_count(self, lines: List[str], min_words: int = 2, max_words: int = 100) -> List[str]:
        """Filter lines by word count."""
        return [line for line in lines if min_words <= len(line.split()) <= max_words]
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        # Very basic language detection
        english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i'}
        spanish_words = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'por', 'con'}
        french_words = {'le', 'de', 'à', 'les', 'il', 'et', 'en', 'des', 'un', 'du'}
        
        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        
        english_score = len(words & english_words)
        spanish_score = len(words & spanish_words)
        french_score = len(words & french_words)
        
        scores = {
            'english': english_score,
            'spanish': spanish_score,
            'french': french_score
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'unknown'
    
    def contains_profanity(self, text: str, profanity_list: Set[str] = None) -> bool:
        """Check if text contains profanity."""
        if profanity_list is None:
            # Basic profanity list (you'd want a more comprehensive list)
            profanity_list = {
                'damn', 'hell', 'crap', 'suck', 'stupid', 'idiot', 'moron', 'shut up',
                'kill', 'die', 'hate', 'stupid', 'dumb', 'screw', 'screw you'
            }
        
        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        return any(profanity in words for profanity in profanity_list)
    
    def validate_dataset(self, file_path: str) -> Dict:
        """Comprehensive validation of a dataset file."""
        validation_report = {
            'file_path': file_path,
            'file_size': 0,
            'line_count': 0,
            'character_count': 0,
            'word_count': 0,
            'unique_words': 0,
            'encoding': 'unknown',
            'language_distribution': {},
            'issues': [],
            'suggestions': []
        }
        
        try:
            # File size
            validation_report['file_size'] = os.path.getsize(file_path)
            
            # Encoding detection
            validation_report['encoding'] = self.detect_encoding(file_path)
            
            # Read and analyze content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            validation_report['line_count'] = len(lines)
            
            if not lines:
                validation_report['issues'].append('File is empty')
                return validation_report
            
            # Analyze content
            all_text = ' '.join(lines)
            validation_report['character_count'] = len(all_text)
            validation_report['word_count'] = len(all_text.split())
            validation_report['unique_words'] = len(set(all_text.split()))
            
            # Language detection
            languages = []
            for line in lines[:100]:  # Sample first 100 lines
                lang = self.detect_language(line)
                languages.append(lang)
            
            validation_report['language_distribution'] = dict(Counter(languages))
            
            # Check for issues
            if validation_report['encoding'] != 'utf-8':
                validation_report['issues'].append(f'Non-UTF-8 encoding: {validation_report["encoding"]}')
            
            if validation_report['line_count'] < 10:
                validation_report['issues'].append('Very small dataset')
            
            if validation_report['unique_words'] < 100:
                validation_report['issues'].append('Limited vocabulary')
            
            # Check for duplicates
            unique_lines = self.remove_duplicates(lines)
            if len(unique_lines) < len(lines) * 0.9:
                validation_report['issues'].append(f'High duplication rate: {len(unique_lines)}/{len(lines)} unique lines')
            
            # Check line lengths
            short_lines = [line for line in lines if len(line) < 10]
            long_lines = [line for line in lines if len(line) > 1000]
            
            if short_lines:
                validation_report['issues'].append(f'{len(short_lines)} very short lines (<10 chars)')
            if long_lines:
                validation_report['issues'].append(f'{len(long_lines)} very long lines (>1000 chars)')
            
            # Suggestions
            if 'Very small dataset' in validation_report['issues']:
                validation_report['suggestions'].append('Consider adding more data')
            
            if 'High duplication rate' in validation_report['issues']:
                validation_report['suggestions'].append('Remove duplicate lines')
            
            if 'Non-UTF-8 encoding' in validation_report['issues']:
                validation_report['suggestions'].append('Convert to UTF-8 encoding')
            
            if short_lines or long_lines:
                validation_report['suggestions'].append('Filter lines by length')
        
        except Exception as e:
            validation_report['issues'].append(f'Error reading file: {e}')
        
        return validation_report
    
    def clean_dataset(self, input_file: str, output_file: str, 
                     min_length: int = 10, max_length: int = 1000,
                     min_words: int = 2, max_words: int = 100,
                     remove_duplicates: bool = True,
                     remove_profanity: bool = False) -> Dict:
        """Clean a dataset file with various filters."""
        cleaning_report = {
            'input_file': input_file,
            'output_file': output_file,
            'lines_before': 0,
            'lines_after': 0,
            'removed_duplicates': 0,
            'removed_short': 0,
            'removed_long': 0,
            'removed_profanity': 0
        }
        
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            cleaning_report['lines_before'] = len(lines)
            
            # Remove duplicates
            if remove_duplicates:
                unique_lines = self.remove_duplicates(lines)
                cleaning_report['removed_duplicates'] = len(lines) - len(unique_lines)
                lines = unique_lines
            
            # Filter by length
            filtered_lines = self.filter_by_length(lines, min_length, max_length)
            cleaning_report['removed_short'] = len([l for l in lines if len(l) < min_length])
            cleaning_report['removed_long'] = len([l for l in lines if len(l) > max_length])
            lines = filtered_lines
            
            # Filter by word count
            filtered_lines = self.filter_by_word_count(lines, min_words, max_words)
            lines = filtered_lines
            
            # Remove profanity
            if remove_profanity:
                clean_lines = []
                for line in lines:
                    if not self.contains_profanity(line):
                        clean_lines.append(line)
                    else:
                        cleaning_report['removed_profanity'] += 1
                lines = clean_lines
            
            # Apply final cleaning
            cleaned_lines = []
            for line in lines:
                line = self.clean_whitespace(line)
                line = self.remove_special_characters(line, keep_punctuation=True)
                if line:
                    cleaned_lines.append(line)
            
            cleaning_report['lines_after'] = len(cleaned_lines)
            
            # Write cleaned data
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cleaned_lines))
            
            return cleaning_report
        
        except Exception as e:
            print(f"Error cleaning dataset: {e}")
            return {'error': str(e)}

def main():
    """Example usage of the data validation and cleaning utilities."""
    validator = DataValidator()
    
    # Example validation
    sample_file = 'data/raw/sample.txt'
    if os.path.exists(sample_file):
        print("Validating sample dataset...")
        report = validator.validate_dataset(sample_file)
        
        print("\n=== VALIDATION REPORT ===")
        for key, value in report.items():
            if key not in ['issues', 'suggestions']:
                print(f"{key}: {value}")
        
        if report.get('issues'):
            print("\n=== ISSUES FOUND ===")
            for issue in report['issues']:
                print(f"• {issue}")
        
        if report.get('suggestions'):
            print("\n=== SUGGESTIONS ===")
            for suggestion in report['suggestions']:
                print(f"• {suggestion}")
    
    # Example cleaning
    print("\n=== CLEANING EXAMPLE ===")
    if os.path.exists(sample_file):
        output_file = 'data/raw/cleaned_sample.txt'
        cleaning_report = validator.clean_dataset(
            sample_file, output_file,
            min_length=20, max_length=1000,
            min_words=3, max_words=200,
            remove_duplicates=True,
            remove_profanity=False
        )
        
        print("Cleaning report:")
        for key, value in cleaning_report.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()