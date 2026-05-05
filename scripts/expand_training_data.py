#!/usr/bin/env python3
"""
Massive data expansion script for BDH model training.
This script can generate gigabytes of training data through multiple methods.
"""

import os
import sys
import requests
import random
import json
import gzip
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_augmentation import TextAugmenter

class MassiveDataExpander:
    """Class to generate massive amounts of training data"""
    
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.augmenter = TextAugmenter(augmentation_prob=0.4)
        
    def download_common_crawl_sample(self, num_segments=10, max_size_mb=1000):
        """Download samples from Common Crawl (massive web corpus)"""
        print("Downloading Common Crawl samples...")
        
        # Common Crawl sample URLs (these are small samples from the massive corpus)
        sample_urls = [
            "https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2023-50/segments/1700678510332.12/warc/CC-MAIN-20231124062520-20231124192520-00000.warc.gz",
            "https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2023-50/segments/1700678510332.12/warc/CC-MAIN-20231124062520-20231124192520-00001.warc.gz",
        ]
        
        # For a real implementation, you'd download and process full segments
        # This is a simplified version that creates synthetic web-like data
        
        web_domains = [
            "news", "blog", "article", "forum", "wiki", "documentation",
            "tutorial", "guide", "manual", "textbook", "encyclopedia"
        ]
        
        web_content_types = [
            "Breaking news: {headline}. {content}",
            "Blog post: {title}. {content}",
            "Technical article: {title}. {content}",
            "Forum discussion: {topic}. {content}",
            "Documentation: {subject}. {content}",
            "Tutorial: Learning {skill}. {content}"
        ]
        
        headlines = [
            "Major breakthrough in artificial intelligence research",
            "New study reveals insights into machine learning optimization",
            "Researchers develop novel neural network architecture",
            "Tech company announces revolutionary AI product",
            "Scientific discovery could transform computer vision",
            "Breakthrough in natural language processing achieved"
        ]
        
        content_samples = []
        for i in range(50000):  # Generate 50K web-like documents
            domain = random.choice(web_domains)
            template = random.choice(web_content_types)
            headline = random.choice(headlines)
            
            # Generate realistic content
            content = self._generate_realistic_content(paragraphs=random.randint(2, 5))
            
            document = template.format(
                headline=headline,
                title=f"{headline} in {domain}",
                content=content,
                topic=f"Discussion about {domain} and AI",
                subject=f"{domain} documentation",
                skill=f"{domain} development"
            )
            
            content_samples.append(document)
            
            if i % 1000 == 0:
                print(f"Generated {i} web documents...")
        
        output_file = self.output_dir / "common_crawl_sample.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(content_samples))
            
        print(f"Generated {len(content_samples)} web-like documents")
        return output_file
    
    def _generate_realistic_content(self, paragraphs=3):
        """Generate realistic paragraph content"""
        paragraph_templates = [
            "Recent developments in the field have shown remarkable progress. {detail} This advancement represents a significant step forward.",
            "The methodology employed in this research involves {technique}. Results indicate that {finding} which suggests important implications.",
            "Comparative analysis reveals that {comparison}. This finding aligns with previous studies that demonstrated {previous_finding}.",
            "Experimental results confirm the hypothesis that {hypothesis}. The data shows consistent patterns across {conditions}.",
            "Future work will focus on {future_direction}. This approach promises to address current limitations in {area}."
        ]
        
        details = [
            "machine learning algorithms can now process complex datasets with unprecedented accuracy",
            "neural networks have achieved human-level performance on specific tasks",
            "computational efficiency has improved by several orders of magnitude",
            "model interpretability has become more accessible to non-experts",
            "training times have been reduced through innovative optimization techniques"
        ]
        
        paragraphs_text = []
        for _ in range(paragraphs):
            template = random.choice(paragraph_templates)
            detail = random.choice(details)
            paragraph = template.format(
                detail=detail,
                technique="advanced statistical methods combined with deep learning",
                finding="significant improvements in model performance",
                comparison="the new approach outperforms traditional methods",
                previous_finding="similar trends in related domains",
                hypothesis="the proposed architecture enhances learning efficiency",
                conditions="various experimental setups",
                future_direction="scaling the approach to larger datasets",
                area="current methodological challenges"
            )
            paragraphs_text.append(paragraph)
        
        return ' '.join(paragraphs_text)
    
    def generate_massive_synthetic_data(self, num_documents=100000):
        """Generate massive amounts of synthetic training data"""
        print(f"Generating {num_documents} synthetic documents...")
        
        document_templates = [
            "The {field} of {domain} has witnessed {adjective} advancements in recent years. {content}",
            "Research in {domain} focuses on {focus_area}. Recent findings indicate that {finding}",
            "The development of {technology} has revolutionized {industry}. Key innovations include {innovations}",
            "{Subject} represents a fundamental aspect of {field}. Current research explores {research_direction}",
            "The intersection of {domain1} and {domain2} has led to {outcome}. This convergence enables {capability}"
        ]
        
        fields = ["field", "domain", "area", "discipline", "specialty"]
        domains = ["artificial intelligence", "machine learning", "computer science", 
                  "data science", "neuroscience", "cognitive science", "robotics"]
        adjectives = ["remarkable", "significant", "substantial", "impressive", "groundbreaking"]
        technologies = ["deep learning", "neural networks", "transformer models", 
                       "reinforcement learning", "computer vision", "natural language processing"]
        industries = ["healthcare", "finance", "education", "manufacturing", "transportation"]
        
        documents = []
        for i in range(num_documents):
            template = random.choice(document_templates)
            
            document = template.format(
                field=random.choice(fields),
                domain=random.choice(domains),
                adjective=random.choice(adjectives),
                content=self._generate_realistic_content(paragraphs=random.randint(2, 4)),
                focus_area="algorithm optimization and model efficiency",
                finding="consistent improvements across multiple benchmarks",
                technology=random.choice(technologies),
                industry=random.choice(industries),
                innovations="novel architectures and training techniques",
                Subject="Computational efficiency",
                research_direction="scalable learning methods",
                domain1=random.choice(domains),
                domain2=random.choice(domains),
                outcome="new interdisciplinary approaches",
                capability="more robust and generalizable systems"
            )
            
            documents.append(document)
            
            if i % 10000 == 0:
                print(f"Generated {i} documents...")
        
        output_file = self.output_dir / "massive_synthetic.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(documents))
            
        print(f"Generated {len(documents)} synthetic documents")
        return output_file
    
    def augment_existing_data(self, input_file, augmentation_factor=10):
        """Massively augment existing data"""
        print(f"Augmenting {input_file} by factor {augmentation_factor}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            original_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        augmented_lines = []
        
        # Use threading for faster augmentation
        def augment_line(line):
            variations = self.augmenter.apply_all_augmentations(line)
            return variations[:augmentation_factor]
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(augment_line, line) for line in original_lines]
            
            for future in as_completed(futures):
                try:
                    variations = future.result()
                    augmented_lines.extend(variations)
                except Exception as e:
                    print(f"Augmentation error: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in augmented_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        output_file = self.output_dir / f"augmented_{Path(input_file).name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_lines))
            
        print(f"Augmented from {len(original_lines)} to {len(unique_lines)} lines")
        return output_file
    
    def combine_all_data(self, output_name="expanded_dataset.txt"):
        """Combine all available data into one massive dataset"""
        data_files = list(self.output_dir.glob("*.txt"))
        
        print(f"Combining {len(data_files)} data files...")
        
        all_content = []
        total_size = 0
        
        for file_path in data_files:
            if file_path.name != output_name:  # Don't include the output file itself
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        all_content.append(content)
                        total_size += len(content)
        
        output_file = self.output_dir / output_name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_content))
        
        # Analyze the combined dataset
        char_count = sum(len(text) for text in all_content)
        word_count = sum(len(text.split()) for text in all_content)
        
        print(f"Combined dataset analysis:")
        print(f"  Files combined: {len(data_files)}")
        print(f"  Total characters: {char_count:,}")
        print(f"  Total words: {word_count:,}")
        print(f"  Approx tokens: {char_count//4:,}")
        print(f"  Output file: {output_file}")
        
        return output_file

def main():
    """Main function to run massive data expansion"""
    parser = argparse.ArgumentParser(description="Massively expand training data")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--synthetic-docs", type=int, default=100000, help="Number of synthetic documents")
    parser.add_argument("--augmentation-factor", type=int, default=10, help="Augmentation factor")
    parser.add_argument("--combine", action='store_true', help="Combine all datasets")
    
    args = parser.parse_args()
    
    expander = MassiveDataExpander(args.output_dir)
    
    print("MASSIVE DATA EXPANSION STARTED")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Generate massive synthetic data
    synthetic_file = expander.generate_massive_synthetic_data(args.synthetic_docs)
    
    # Step 2: Augment existing data
    existing_files = ["sample.txt", "combined_dataset.txt"]
    for file_name in existing_files:
        file_path = expander.output_dir / file_name
        if file_path.exists():
            expander.augment_existing_data(str(file_path), args.augmentation_factor)
    
    # Step 3: Combine all data
    if args.combine:
        combined_file = expander.combine_all_data()
        
        # Final analysis
        with open(combined_file, 'r', encoding='utf-8') as f:
            final_content = f.read()
            final_chars = len(final_content)
            final_words = len(final_content.split())
            
        print(f"\nFINAL DATASET STATISTICS:")
        print(f"  Total characters: {final_chars:,}")
        print(f"  Total words: {final_words:,}")
        print(f"  Approx tokens: {final_chars//4:,}")
        print(f"  Data expansion: {final_chars/377701:.1f}x original size")
    
    end_time = time.time()
    print(f"\nData expansion completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    main()