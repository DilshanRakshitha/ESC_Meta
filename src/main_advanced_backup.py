#!/usr/bin/env python3
"""
ESC Meta - PyTorch Main Pipeline with Advanced Trainer
Proper implementation using PyTorch models and advanced training infrastructure
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

print("🎵 ESC META - PYTORCH MAIN PIPELINE")
print("🔗 Using advanced trainer and PyTorch models")

# Import proper PyTorch components
try:
    from models.training.advanced_trainer import AdvancedTrainer
    print("✅ Advanced trainer imported")
    TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced trainer not available: {e}")
    TRAINER_AVAILABLE = False

# Import PyTorch model architectures (bypassing __init__.py issues)
try:
    from models.architectures.AlexNet import AlexNet
    print("✅ AlexNet imported")
    ALEXNET_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AlexNet not available: {e}")
    ALEXNET_AVAILABLE = False

try:
    from models.architectures.DenseNet121 import create_densenet121
    from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
    from models.architectures.InceptionV3 import create_inception_v3
    from models.architectures.MobileNetV3Small import create_mobilenet_v3_small
    print("✅ CNN models imported")
    CNN_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CNN models not available: {e}")
    CNN_MODELS_AVAILABLE = False

try:
    from models.architectures.ResNet50V2 import create_resnet50_v2, create_resnet18
    print("✅ ResNet models imported")
    RESNET_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ResNet models not available: {e}")
    RESNET_AVAILABLE = False

try:
    from models.architectures.kan_models import create_high_performance_kan
    print("✅ KAN model imported")
    KAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ KAN model not available: {e}")
    KAN_AVAILABLE = False

class PyTorchModelFactory:
    """Factory for creating PyTorch models with proper advanced trainer integration"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Model factory initialized on device: {self.device}")
    
    def create_model(self, model_name: str, num_classes: int = 26):
        """Create PyTorch model by name"""
        model_name = model_name.lower()
        
        if model_name in ['alexnet', 'alex'] and ALEXNET_AVAILABLE:
            return AlexNet(input_size=3, num_classes=num_classes)
        
        elif model_name in ['densenet', 'densenet121'] and CNN_MODELS_AVAILABLE:
            return create_densenet121(num_classes=num_classes, input_channels=3)
        
        elif model_name in ['kan'] and KAN_AVAILABLE:
            return create_high_performance_kan((128, 196, 3), num_classes)
        
        else:
            raise ValueError(f"Model '{model_name}' not available or not supported")
    
    def get_available_models(self):
        """Get list of available models"""
        available = []
        if ALEXNET_AVAILABLE:
            available.append('alexnet')
        if CNN_MODELS_AVAILABLE:
            available.append('densenet')
        if KAN_AVAILABLE:
            available.append('kan')
        return available

def run_advanced_training(model_name: str):
    """Run training using the advanced trainer"""
    print(f"\n🚀 Starting advanced training for {model_name}")
    
    if not TRAINER_AVAILABLE:
        print("❌ Advanced trainer not available")
        return
    
    # Create model
    factory = PyTorchModelFactory()
    model = factory.create_model(model_name)
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create training config
    from config.config import TrainingConfig
    
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        optimizer='adam'
    )
    
    # Initialize trainer
    trainer = AdvancedTrainer(config)
    
    print(f"🎯 Advanced training setup complete for {model_name}")

def main():
    parser = argparse.ArgumentParser(description='ESC Meta - PyTorch Main Pipeline')
    parser.add_argument('--model', type=str, help='Model to train/test')
    parser.add_argument('--test', action='store_true', help='Test all models')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--advanced', action='store_true', help='Use advanced trainer')
    
    args = parser.parse_args()
    
    factory = PyTorchModelFactory()
    available_models = factory.get_available_models()
    
    if args.list:
        print("📋 Available PyTorch models:")
        for model in available_models:
            print(f"  • {model}")
        return
    
    if args.test:
        print("\n📦 Testing all PyTorch models...")
        for model_name in available_models:
            try:
                model = factory.create_model(model_name, 26)
                params = sum(p.numel() for p in model.parameters())
                print(f"✅ {model_name}: {params:,} parameters")
            except Exception as e:
                print(f"❌ {model_name}: {e}")
        return
    
    if args.model:
        if args.model.lower() not in available_models:
            print(f"❌ Model '{args.model}' not available")
            print(f"Available models: {available_models}")
            return
        
        if args.train:
            if args.advanced:
                run_advanced_training(args.model)
            else:
                print(f"🔄 Basic training not implemented in PyTorch main")
                print(f"💡 Use root main.py for basic training or --advanced flag")
        else:
            # Just test model creation
            try:
                model = factory.create_model(args.model, 26)
                params = sum(p.numel() for p in model.parameters())
                print(f"✅ {args.model} created successfully: {params:,} parameters")
            except Exception as e:
                print(f"❌ {args.model} failed: {e}")
    
    else:
        print("💡 ESC Meta - PyTorch Pipeline with Advanced Trainer")
        print("💡 Use --list to see available models")
        print("💡 Use --test to test all models")
        print("💡 Use --model <name> --train --advanced for advanced training")
        print(f"💡 Available: {len(available_models)} PyTorch models")

if __name__ == "__main__":
    main()
