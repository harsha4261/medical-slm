# Medical Small Language Model (medical_slm)

A small language model specialized for medical domain tasks using efficient training approaches.

## Overview

This project implements a compact small language model (SLM) designed for medical applications. It includes training and evaluation scripts for building domain-specific language models with a focus on medical text understanding and generation.

## Features

- **Compact Architecture**: Optimized for medical domain
- **Training Script**: `train_medical_slm1.py` - Complete training pipeline
- **Evaluation Tools**: `evaluate_medical.py` - Model evaluation metrics
- **Interactive Chat**: `chat_medical.py` - Chat interface for model interaction
- **Visualization**: Loss plots and training metrics

## Project Structure

```
medical_slm/
├── train_medical_slm1.py      # Main training script
├── evaluate_medical.py         # Evaluation script
├── chat_medical.py             # Interactive chat interface
├── best_model_params.pt        # Best model weights
├── plot.png                    # Training visualization
├── train.bin                   # Training data
├── validation.bin              # Validation data
└── requirements.txt            # Dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/medical-slm.git
cd medical-slm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_medical_slm1.py
```

### Evaluation

```bash
python evaluate_medical.py
```

### Interactive Chat

```bash
python chat_medical.py
```

## Requirements

See `requirements.txt` for complete dependencies.

## Model Weights

The best model weights are saved in `best_model_params.pt`

## Training Data

- Training data: `train.bin`
- Validation data: `validation.bin`

## Results

Training metrics and loss plots are saved in `plot.png`

## Author

Harsha Vardhan Emani

## Contact

For questions or issues, please create an issue on GitHub.
