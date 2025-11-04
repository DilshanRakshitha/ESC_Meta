"""
ESC Meta - Simple Working Main
Clean implementation without problematic imports
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

print("🎵 FSC META - CLEAN MAIN PIPELINE")

def create_simple_cnn(input_channels=3, num_classes=26):
    """Create a simple CNN that always works"""
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 16, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )

def test_model_loading():
    """Test which models can be loaded safely"""
    print("\n📦 Testing model availability...")
    
    # Test AlexNet
    try:
        from models.architectures.AlexNet import AlexNet
        model = AlexNet(input_size=3, num_classes=26)
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ AlexNet: {params:,} parameters")
    except Exception as e:
        print(f"❌ AlexNet: {e}")
    
    # Test DenseNet
    try:
        from models.architectures.DenseNet121 import create_densenet121
        model = create_densenet121(num_classes=26, input_channels=3)
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ DenseNet121: {params:,} parameters")
    except Exception as e:
        print(f"❌ DenseNet121: {e}")
    
    # Test simple CNN fallback
    try:
        model = create_simple_cnn()
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ Simple CNN (fallback): {params:,} parameters")
    except Exception as e:
        print(f"❌ Simple CNN: {e}")

def main():
    parser = argparse.ArgumentParser(description='FSC Meta - Clean Pipeline')
    parser.add_argument('--test', action='store_true', help='Test model loading')
    parser.add_argument('--model', type=str, help='Model to test')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Running model test...")
        test_model_loading()
        return
    
    if args.model:
        print(f"🎯 Testing model: {args.model}")
        
        if args.model.lower() == 'alexnet':
            try:
                from models.architectures.AlexNet import AlexNet
                model = AlexNet(input_size=3, num_classes=26)
                print(f"✅ {args.model} created successfully")
            except Exception as e:
                print(f"❌ {args.model} failed: {e}")
        
        elif args.model.lower() == 'densenet':
            try:
                from models.architectures.DenseNet121 import create_densenet121
                model = create_densenet121(num_classes=26, input_channels=3)
                print(f"✅ {args.model} created successfully")
            except Exception as e:
                print(f"❌ {args.model} failed: {e}")
        
        else:
            print(f"❌ Model {args.model} not implemented in clean version")
            print("Available: alexnet, densenet")
    
    else:
        print("💡 Use --test to test all models")
        print("💡 Use --model <name> to test a specific model")
        print("💡 Available models: alexnet, densenet")

if __name__ == "__main__":
    main()
