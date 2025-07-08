# Cat Breed Classification CNN

A Convolutional Neural Network model for classifying 20 different cat breeds with **89% accuracy**, designed for integration into a cat adoption center website.

## Model Overview
- **Architecture**: Inception-based CNN
- **Parameters**: 3+ million
- **Input**: Cat images 312x312x3
- **Output**: 20 breed classifications
- **Dataset**: 10,000+ custom-collected images
- **Key Feature**: Lightweight design for web deployment

## Performance
- **Accuracy**: 89%
- **Training**: Custom dataset
- **Inference Speed**: Optimized for web use

## Requirements

CUDA or CPU

### Python Version
- Python 3.11

### Dependencies
- torch
- torchaudio
- torchvision
- seaborn
- matplotlib
- scikit-learn

## Installation
1. Clone repository:
```bash
git clone https://github.com/erik000mm/CatBreedDetection
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Future work
- **Expand to more breeds**
- **Increase dataset size**
- **Improve accuracy**