#!/bin/bash

# FSC Meta Optimization Setup Script
# Install required dependencies and test the optimization system

echo "🔍 FSC Meta Hyperparameter Optimization Setup"
echo "=============================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "esc_meta" ]]; then
    echo "⚠️ Please activate the esc_meta environment first:"
    echo "   conda activate esc_meta"
    exit 1
fi

echo "📦 Installing Optuna and visualization dependencies..."

# Install Optuna and related packages
pip install optuna
pip install optuna-dashboard
pip install plotly
pip install matplotlib
pip install seaborn

echo "✅ Dependencies installed!"

# Create optimization directory structure
echo "📁 Creating optimization directory structure..."
mkdir -p optimization/results
mkdir -p optimization/studies
mkdir -p optimization/plots

echo "🧪 Testing optimization system..."

# Quick test with AlexNet
echo "🔍 Running quick optimization test with AlexNet..."
cd "$(dirname "$0")"
python -c "
import sys
sys.path.append('.')
from optimization.optimize import FSCMetaOptimizer
from optimization.config import OptimizationPresets

print('🚀 Testing FSC Meta Optimizer...')
try:
    optimizer = FSCMetaOptimizer()
    print('✅ Optimizer initialized successfully')
    
    # Test with synthetic data
    import numpy as np
    features = np.random.randn(100, 3, 128, 196)
    labels = np.random.randint(0, 26, 100)
    print('✅ Test data created')
    
    print('🎯 Starting quick test optimization...')
    results = optimizer.optimize_single_model(
        model_name='alexnet',
        preset='quick_test',
        custom_trials=3
    )
    
    print('✅ Quick test completed successfully!')
    print(f'   Best accuracy: {results[\"best_value\"]:.4f}')
    
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "Usage examples:"
echo "  # Quick test (3 trials, ~2 minutes)"
echo "  python optimization/optimize.py --model alexnet --preset quick_test"
echo ""
echo "  # Development optimization (25 trials, ~30 minutes)"
echo "  python optimization/optimize.py --model kan --preset development"
echo ""
echo "  # Optimize all models"
echo "  python optimization/optimize.py --model all --preset development"
echo ""
echo "  # Compare results"
echo "  python optimization/optimize.py --compare"
echo ""
echo "  # Production optimization (100 trials, ~2 hours)"
echo "  python optimization/optimize.py --model all --preset production"
echo ""
echo "📊 Optuna dashboard (optional):"
echo "  optuna-dashboard sqlite:///optimization/results/optuna.db"
echo ""
