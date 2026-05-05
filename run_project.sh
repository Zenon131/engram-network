#!/bin/zsh

# Script to run through the complete flow of the BDH project
# Training -> Evaluation -> Text Generation

echo "BDH Project Flow Execution"
echo "This script will run through the complete flow of the BDH project:"
echo "1. Data preparation"
echo "2. Model training"
echo "3. Model evaluation"
echo "4. Text generation"
echo ""

# Check if conda environment is active
if [[ -z $CONDA_DEFAULT_ENV ]]; then
    echo "Warning: No conda environment is currently active."
    echo "Please activate your environment before running this script."
    echo "Example: conda activate your_env_name"
    echo ""
    read -q "REPLY?Do you want to continue anyway? (y/n) "
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Create necessary directories
echo "Creating directories"
mkdir -p data/processed checkpoints
echo "Created data/processed and checkpoints directories"
echo ""

# Step 1: Data preparation
echo "Step 1: Data Preparation"
echo "Running data preparation script..."

# Use expanded dataset if available, otherwise use synthetic data
if [ -f "data/raw/expanded_dataset.txt" ]; then
    echo "Using expanded dataset..."
    python scripts/prepare_data.py --input data/raw/expanded_dataset.txt --output data/processed --tokenizer byte
else
    echo "Using synthetic data (run scripts/expand_training_data.py for larger dataset)..."
    python scripts/prepare_data.py --input data/raw/synthetic_data.txt --output data/processed --tokenizer byte
fi

if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed"
    exit 1
fi

echo "Data preparation completed successfully"
echo ""

# Step 2: Model training
echo "Step 2: Model Training"
echo "Running enhanced model training script..."

# Use tiny config for memory safety - larger configs may be too big for 16GB RAM
if [ -f "data/raw/expanded_dataset.txt" ]; then
    echo "Using tiny configuration with expanded dataset..."
    python scripts/train_enhanced.py --config configs/tiny.yaml --data data/raw/expanded_dataset.txt
else
    echo "Using tiny configuration with synthetic dataset..."
    python scripts/train_enhanced.py --config configs/tiny.yaml
fi

if [ $? -ne 0 ]; then
    echo "Error: Model training failed"
    echo "Tip: The model may be too large for available memory"
    echo "Consider reducing batch_size or model size in the configuration"
    exit 1
fi

echo "Enhanced model training completed successfully"
echo ""

# Step 3: Model evaluation
echo "Step 3: Model Evaluation"
echo "Running enhanced model evaluation script..."

# Use the same configuration as training for evaluation
if [ -f "data/raw/expanded_dataset.txt" ]; then
    echo "Evaluating with expanded dataset..."
    python scripts/eval_torch.py --config configs/tiny.yaml --data data/raw/expanded_dataset.txt
else
    echo "Evaluating with synthetic dataset..."
    python scripts/eval_torch.py --config configs/tiny.yaml
fi

if [ $? -ne 0 ]; then
    echo "Error: Model evaluation failed"
    exit 1
fi

echo "Enhanced model evaluation completed successfully"
echo ""

# Step 4: Text generation
echo "Step 4: Text Generation"
echo "Running text generation script..."
python scripts/generate_torch.py

if [ $? -ne 0 ]; then
    echo "Error: Text generation failed"
    exit 1
fi

echo "Text generation completed successfully"
echo ""

echo "All steps completed successfully! You have successfully run through the complete BDH project flow:"
echo "1. Data preparation - Completed"
echo "2. Enhanced model training - Completed"
echo "3. Enhanced model evaluation - Completed"
echo "4. Text generation - Completed"
echo ""
echo "Enhanced training features:"
echo "- Better monitoring and logging"
echo "- Advanced optimizer configurations"
echo "- Learning rate scheduling"
echo "- Gradient clipping"
echo ""
echo "Checkpoints are saved in the 'checkpoints' directory"
echo "Processed data is saved in the 'data/processed' directory"
echo "Training logs are saved in the 'logs' directory"