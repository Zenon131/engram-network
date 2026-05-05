# Massive Data Expansion Plan for BDH Model

## Current Data Situation
- **Current data**: ~94K tokens (377K characters)
- **Model capacity**: Up to 1-2 billion parameters
- **Data requirement**: 10-100 tokens per parameter (10B-200B tokens ideal)

## Immediate Data Expansion Strategy

### Phase 1: Massive Synthetic Generation (100x expansion)
```bash
# Generate 100,000 synthetic documents (~50M tokens)
python scripts/expand_training_data.py --synthetic-docs 100000

# Expected output: ~50M tokens (500x current size)
```

### Phase 2: Aggressive Data Augmentation (10x expansion)
```bash
# Augment existing data by 10x
python scripts/expand_training_data.py --augmentation-factor 10 --combine

# Expected output: ~500M tokens (5,000x current size)
```

### Phase 3: Web-Scale Data Collection (1,000x expansion)
```bash
# Download and process web-scale data
python scripts/download_datasets.py --datasets all --combine
```

## Data Expansion Targets

| Phase | Target Tokens | Expansion Factor | Model Capacity |
|-------|---------------|------------------|----------------|
| Current | 94K | 1x | 10M params |
| Phase 1 | 50M | 500x | 500M params |
| Phase 2 | 500M | 5,000x | 5B params |
| Phase 3 | 5B | 50,000x | 50B params |

## Running the Expansion

### Step 1: Generate Massive Synthetic Data
```bash
cd /Users/jonathanwallace/Projects/bdh
python scripts/expand_training_data.py --synthetic-docs 100000 --combine
```

### Step 2: Augment Existing Data
```bash
python scripts/data_augmentation.py
```

### Step 3: Download Additional Datasets
```bash
python scripts/download_datasets.py --datasets all --combine
```

### Step 4: Combine Everything
```bash
python scripts/expand_training_data.py --combine
```

## Expected Results

### After Phase 1 (100K synthetic docs):
- **Tokens**: ~50 million
- **Characters**: ~200 million
- **Expansion**: 500x current size
- **Model support**: Up to 500M parameters

### After Phase 2 (10x augmentation):
- **Tokens**: ~500 million
- **Characters**: ~2 billion
- **Expansion**: 5,000x current size
- **Model support**: Up to 5B parameters

### After Phase 3 (web-scale):
- **Tokens**: ~5 billion
- **Characters**: ~20 billion
- **Expansion**: 50,000x current size
- **Model support**: Up to 50B parameters

## Memory Considerations

Your M3 MacBook Air with 16GB RAM can handle:
- **Data processing**: Up to billions of tokens with streaming
- **Model training**: Up to 1-2B parameters comfortably
- **Storage**: Generated data will be ~20GB for 5B tokens

## Implementation Commands

Run these commands sequentially:

```bash
# 1. Generate massive synthetic data
python scripts/expand_training_data.py --synthetic-docs 100000

# 2. Augment everything
python scripts/expand_training_data.py --augmentation-factor 10 --combine

# 3. Check the final dataset size
python -c "
import os
file_path = 'data/raw/expanded_dataset.txt'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f'Final dataset: {len(content):,} chars, ~{len(content)//4:,} tokens')
else:
    print('Run the expansion script first')
"
```

## Monitoring Progress

The script provides real-time progress:
- Document generation status
- Augmentation progress
- Final size statistics
- Memory usage

## Scaling Further

If you need even more data:
1. Increase `--synthetic-docs` to 1,000,000+
2. Use `--augmentation-factor 50` for 50x augmentation
3. Add more data sources from Common Crawl, Wikipedia dumps, etc.

## Next Steps After Expansion

1. **Preprocess the data**:
   ```bash
   python scripts/prepare_data.py --input data/raw/expanded_dataset.txt --output data/processed
   ```

2. **Update model configuration**:
   - Increase `n_neurons` to 500K-1M
   - Increase `n_layers` to 8-12
   - Adjust `batch_size` based on memory

3. **Start training**:
   ```bash
   python scripts/train_torch.py
   ```

This expansion will give you sufficient data to train models with hundreds of millions to billions of parameters on your M3 MacBook Air.