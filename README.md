# ESC Meta - Environmental Sound Classification with KAN

A PyTorch-based implementation for environmental sound classification using Kolmogorov-Arnold Networks (KAN) and traditional CNN architectures on the FSC22 dataset.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DilshanRakshitha/ESC_Meta.git
cd ESC_Meta
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data structure:
```
ESC_Meta/
├── data/
│   └── fsc22/
│       ├── FSC22.csv          # Dataset metadata
│       ├── wav44/             # Audio files (.wav)
│       └── Pickle_Files/      # Generated features (created automatically)
└── ...
```

## Usage

### 1. Feature Generation

Generate mel spectrogram features from audio files:

```bash
# Basic usage (uses default settings)
python generate_features.py

# With custom paths (relative to project root - recommended)
python generate_features.py --csv-path data/fsc22/FSC22.csv --audio-path data/fsc22/wav44 --output data/fsc22/Pickle_Files

# Different feature types
python generate_features.py --feature-type MEL    # Mel spectrograms (default)
python generate_features.py --feature-type MFCC   # MFCC features
python generate_features.py --feature-type MIX    # Mixed features

# Augmentation levels
python generate_features.py --augmentation 0   # No augmentation
python generate_features.py --augmentation 1   # Time stretch only
python generate_features.py --augmentation 2   # Pitch shift only
python generate_features.py --augmentation 3   # Time stretch + pitch shift (default)
python generate_features.py --augmentation 4   # All augmentations

# Use preset configuration
python generate_features.py --preset aug_ts_ps_mel_features_5_20
```

**Available Arguments:**
- `--csv-path, -c`: Path to FSC22.csv file (default: data/fsc22/FSC22.csv)
- `--audio-path, -a`: Path to audio files directory (default: data/fsc22/wav44)  
- `--output, -o`: Output directory for features (default: data/fsc22/Pickle_Files)
- `--feature-type, -f`: Feature type - MEL, MFCC, MIX (default: MEL)
- `--augmentation, -aug`: Augmentation level 0-4 (default: 3)
- `--preset, -p`: Use preset configuration (aug_ts_ps_mel_features_5_20)

### 2. Model Training

Train models using cross-validation:

```bash
# Train different models
python main.py --model alexnet
python main.py --model kan  
python main.py --model ickan
python main.py --model rapid_kan_lite
```

**Available Models:**

**CNN Architectures:**
- `alexnet` - AlexNet CNN
- `densenet` - DenseNet-121
- `efficientnet` - EfficientNet-V2 B0  
- `inception` - Inception-V3
- `mobilenet` - MobileNet-V3 Small
- `mobilenetv3large` - MobileNet-V3 Large
- `resnet50` - ResNet-50 v2
- `resnet18` - ResNet-18

**KAN-Inspired Architectures (Fast, CNN-based):**
- `kan_inspired` - High-performance KAN-inspired model
- `ickan_inspired` - ICKAN-inspired model  
- `wavkan_inspired` - Wavelet KAN-inspired model

**KAN Architectures:**
- `kan` - Exact KAN implementation
- `kan_fast` - Fast KAN (balanced mode)
- `kan_memory_safe` - Memory-efficient KAN

**ICKAN Architectures:**
- `ickan` - Standard ICKAN
- `ickan_light` - Lightweight ICKAN
- `ickan_deep` - Deep ICKAN

**Rapid KAN Architectures:**
- `rapid_kan` - Efficient Rapid KAN
- `rapid_kan_lite` - Lightweight Rapid KAN  
- `rapid_kan_power` - Powerful Rapid KAN

### 3. Hyperparameter Optimization

Hyperparameter optimization for KAN models is not still working. 
Optimize model hyperparameters using Optuna:

```bash
# Quick optimization (5 trials, 10 epochs)
python optimize_models.py --model alexnet --config quick

# Standard optimization (20 trials, 30 epochs)
python optimize_models.py --model kan --config standard

# Extensive optimization (100 trials, 50 epochs)
python optimize_models.py --model ickan --config extensive

# Optimize all available models
python optimize_models.py --model all --config quick
```

**Configuration Options:**
- `quick` - Fast testing with few trials
- `standard` - Balanced optimization
- `extensive` - Thorough hyperparameter search
