# ESC Meta - Forest Sound Classification with Advanced Neural Networks

ESC Meta is a comprehensive machine learning framework for Environmental Sound Classification (ESC) with a focus on forest sound analysis. This project integrates multiple state-of-the-art neural network architectures including traditional CNNs, Kolmogorov-Arnold Networks (KAN), and advanced optimization techniques.

## Project Structure

```
ESC_Meta/
├── main.py                      # Main execution pipeline
├── optimize_models.py           # Hyperparameter optimization script
├── run_model_comparison.sh      # Batch model comparison script
├── requirements.txt             # Python dependencies
├── config/                      # Configuration files
│   ├── fsc_comprehensive_config.yml
│   ├── optimization_configs.yml
│   └── fsc22.yml
├── models/                      # Model architectures and training
│   ├── architectures/          # Neural network models
│   │   ├── AlexNet.py
│   │   ├── DenseNet121.py
│   │   ├── EfficientNetV2B0.py
│   │   ├── InceptionV3.py
│   │   ├── ResNet50V2.py
│   │   ├── MobileNetV3Small.py
│   │   ├── kan_models.py       # Kolmogorov-Arnold Networks
│   │   ├── ickan_models.py     # Improved KAN variants
│   │   └── wavkan_models.py    # Wavelet-based KAN
│   └── training/               # Training modules
├── features/                   # Feature extraction
├── data/                      # Dataset management
├── optimization/              # Hyperparameter optimization
└── utils/                     # Utility functions
```

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository_url>
cd ESC_Meta

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

#### Run with Default Settings
```bash
python main.py
```

#### Run Specific Model
```bash
# Run AlexNet
python main.py --model alexnet

# Run KAN model
python main.py --model kan

# Run with specific configuration
python main.py --config config/fsc_comprehensive_config.yml --model densenet
```

#### Available Models
- `alexnet` - AlexNet architecture
- `densenet` / `densenet121` - DenseNet121
- `efficientnet` / `efficientnetv2b0` - EfficientNetV2-B0
- `inception` / `inceptionv3` - InceptionV3
- `resnet` / `resnet50v2` - ResNet50V2
- `resnet18` - ResNet18
- `mobilenet` / `mobilenetv3small` - MobileNetV3-Small
- `mobilenetv3large` - MobileNetV3-Large
- `kan` - Kolmogorov-Arnold Network
- `ickan` - Improved KAN
- `wavkan` - Wavelet KAN

### 3. Model Comparison

Run multiple models automatically:

```bash
# Make script executable
chmod +x run_model_comparison.sh

# Run comparison
./run_model_comparison.sh
```

### Main Configuration (`config/fsc_comprehensive_config.yml`)

The main configuration file controls:

- **Data Processing**: Audio parameters, feature extraction methods
- **Model Settings**: Architecture-specific parameters
- **Training Parameters**: Batch size, epochs, learning rate
- **Cross-Validation**: Number of folds, random seed

## Hyperparameter Optimization

### Basic Optimization

```bash
# Optimize AlexNet
python optimize_models.py --model alexnet --trials 50

# Optimize KAN with extended search
python optimize_models.py --model kan --config extended --trials 100

# Quick optimization for testing
python optimize_models.py --model densenet --config quick --trials 20
```

### Advanced Optimization Options

```bash
# Custom optimization with specific parameters
python optimize_models.py \
  --model alexnet \
  --trials 100 \
  --timeout 7200 \
  --config extended \
  --study-name "alexnet_extensive_search"

# Resume existing study
python optimize_models.py \
  --model kan \
  --trials 50 \
  --resume-study "kan_optimization_v2"
```

## Results

### Expected Performance

| Model | Expected Accuracy | Description |
|-------|------------------|-------------|
| AlexNet | ~89.5% | Baseline CNN architecture |
| DenseNet121 | ~91-93% | Dense connections, efficient |
| EfficientNetV2 | ~92-94% | Optimized efficiency |
| ResNet50V2 | ~90-92% | Deep residual learning |
| KAN | ~85-90% | Novel Kolmogorov-Arnold approach |
| ICKAN | ~87-91% | Improved KAN variant |
| WavKAN | ~86-90% | Wavelet-based KAN |

## License

This project is licensed under the MIT License - see the LICENSE file for details.


**Note**: This framework is designed for research and educational purposes. Ensure you have appropriate data and computational resources before running extensive experiments.
