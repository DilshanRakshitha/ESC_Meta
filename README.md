# ESC Meta - Clean Audio Classification Pipeline

A streamlined, modular audio classification system with hyperparameter optimization for Environmental Sound Classification (ESC).

## 🏗️ Project Structure

```
ESC_Meta/
├── 📁 config/                          # Configuration files
│   ├── fsc22.yml                       # FSC22 dataset config
│   ├── fsc_comprehensive_config.yml    # Main pipeline config
│   └── optimization_configs.yml        # Hyperparameter optimization config
│
├── 📁 models/                          # Model architectures and training
│   ├── architectures/                 # Core model implementations
│   │   ├── AlexNet.py                 # AlexNet architecture
│   │   ├── kan_models.py              # KAN (Kolmogorov-Arnold Network)
│   │   ├── ickan_models.py            # ICKAN variant
│   │   └── wavkan_models.py           # WavKAN variant
│   └── training/                      # Training utilities
│       └── trainer.py                 # Cross-validation trainer
│
├── 📁 features/                        # Feature extraction
│   ├── fsc_original_features.py       # FSC Original data loader
│   └── extractors.py                  # Feature extraction utilities
│
├── 📁 optimization/                    # Hyperparameter optimization
│   ├── hyperparameter_optimizer.py    # Main optimization interface
│   ├── optimization_config.py         # Configuration management
│   ├── objective_function.py          # Optuna objective function
│   └── README.md                      # Optimization documentation
│
├── 📁 utils/                          # Utilities
│   └── data_prep.py                   # Data preprocessing
│
├── 📁 data/                           # Data loading
│   └── dataloader.py                  # Data loading utilities
│
├── main.py                            # Main pipeline entry point
├── optimize_models.py                 # Hyperparameter optimization script
└── run_model_comparison.sh            # Model comparison script
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda activate esc_meta
```

### 2. Train a Single Model
```bash
# Train AlexNet
python main.py --model alexnet

# Train KAN
python main.py --model kan

# Train ICKAN  
python main.py --model ickan

# Train WavKAN
python main.py --model wavkan
```

### 3. Hyperparameter Optimization
```bash
# Quick test (5 trials, ~5 minutes)
python optimize_models.py --model alexnet --config quick

# Standard optimization (20 trials, ~1 hour)
python optimize_models.py --model kan --config standard

# Extensive optimization (100 trials, ~2+ hours)
python optimize_models.py --model all --config extensive
```

### 4. Compare All Models
```bash
./run_model_comparison.sh
```

## 🏆 Model Architectures

### AlexNet
- Classic CNN architecture adapted for audio spectrograms
- Proven performance for audio classification
- ~97.90% accuracy on FSC22

### KAN (Kolmogorov-Arnold Network)
- Novel architecture using learnable activation functions
- High-performance implementation with residual connections
- ~98.56% accuracy on FSC22

### ICKAN (Improved Convolutional KAN)
- Enhanced KAN with improved convolutional layers
- Optimized for audio feature extraction

### WavKAN (Wavelet-based KAN)
- KAN architecture with wavelet transforms
- Specialized for time-frequency audio analysis

## ⚙️ Configuration

### Training Configuration
- `config/fsc_comprehensive_config.yml` - Main training parameters
- Supports CPU/GPU training
- Configurable batch sizes, learning rates, epochs

### Optimization Configuration  
- `config/optimization_configs.yml` - Hyperparameter tuning settings
- Three modes: quick, standard, extensive
- Easy to modify parameter ranges and trial counts

## 📊 Hyperparameter Optimization

The system includes a sophisticated optimization pipeline using Optuna:

### Features
- **Multiple optimization modes** (quick/standard/extensive)
- **Model-agnostic** hyperparameter tuning
- **Automatic early stopping** and pruning
- **Visualization** of optimization results
- **Persistent studies** with SQLite storage

### Optimized Parameters
- Learning rate
- Batch size  
- Optimizer choice (Adam, AdamW, SGD)
- Weight decay
- Model-specific parameters (dropout, hidden dimensions)

## 📈 Results and Outputs

### Training Results
- Cross-validation scores
- Best model checkpoints
- Training logs and metrics

### Optimization Results
- Best hyperparameters in YAML format
- Complete trial history in CSV
- Optimization plots (history, parameter importance)
- SQLite database for study persistence

## 🛠️ Technical Details

### Data Pipeline
- Supports FSC22 dataset
- Automatic feature extraction (mel-spectrograms)
- Efficient data loading with augmentation

### Training Pipeline
- 5-fold cross-validation
- Early stopping with patience
- Learning rate scheduling
- Batch normalization and dropout

### Optimization Pipeline
- Bayesian optimization with Optuna
- Pruning of unpromising trials
- Parallel trial execution
- Configurable search spaces

## 📁 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Main training pipeline |
| `optimize_models.py` | Hyperparameter optimization |
| `models/architectures/AlexNet.py` | AlexNet implementation |
| `models/architectures/kan_models.py` | KAN implementation |
| `models/training/trainer.py` | Training and CV logic |
| `optimization/hyperparameter_optimizer.py` | Main optimization interface |
| `config/optimization_configs.yml` | Optimization settings |

## 🎯 Performance

Recent optimization results on FSC22:
- **AlexNet**: 97.90% ± 0.05% (optimized)
- **KAN**: 98.56% ± 0.03% (optimized)
- **ICKAN**: Testing in progress
- **WavKAN**: Testing in progress

## 💡 Usage Tips

1. **Start with quick mode** for initial testing
2. **Use standard mode** for production optimization
3. **Monitor optimization** through generated plots
4. **Save best parameters** for reproducible results
5. **Use CPU mode** if GPU memory is limited

## 🔧 Customization

The system is designed to be easily customizable:
- Add new model architectures in `models/architectures/`
- Modify hyperparameter ranges in `config/optimization_configs.yml`
- Extend the optimization with custom objective functions
- Add new feature extractors in `features/`

This clean, modular design ensures maintainability while providing state-of-the-art audio classification capabilities.
