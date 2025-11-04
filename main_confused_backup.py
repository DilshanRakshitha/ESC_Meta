#!/usr/bin/env python3
"""
ESC Meta - Complete Working Main Pipeline
Clean implementation with all available models
Bypasses problematic imports for stability
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("🎵 FSC META - COMPLETE MAIN PIPELINE")
print("🔗 Direct imports for maximum compatibility")

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

def create_model(model_name: str):
    """Create model by name with direct imports"""
    model_name = model_name.lower()
    
    if model_name in ['alexnet', 'alex']:
        from models.architectures.AlexNet import AlexNet
        return AlexNet(input_size=3, num_classes=26)
    
    elif model_name in ['densenet', 'densenet121']:
        from models.architectures.DenseNet121 import create_densenet121
        return create_densenet121(num_classes=26, input_channels=3)
    
    elif model_name in ['efficientnet', 'efficientnetv2', 'efficientnetv2b0']:
        from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
        return create_efficientnet_v2_b0(num_classes=26, input_channels=3)
    
    elif model_name in ['inception', 'inceptionv3']:
        from models.architectures.InceptionV3 import create_inception_v3
        return create_inception_v3(num_classes=26, input_channels=3, pretrained=False)
    
    elif model_name in ['mobilenet', 'mobilenetv3', 'mobilenetv3small']:
        from models.architectures.MobileNetV3Small import create_mobilenet_v3_small
        return create_mobilenet_v3_small(num_classes=26, input_channels=3)
    
    elif model_name in ['resnet', 'resnet50', 'resnet50v2']:
        from models.architectures.ResNet50V2 import create_resnet50_v2
        return create_resnet50_v2(num_classes=26, input_channels=3)
    
    elif model_name in ['resnet18']:
        from models.architectures.ResNet50V2 import create_resnet18
        return create_resnet18(num_classes=26, input_channels=3)
    
    elif model_name in ['kan']:
        from models.architectures.kan_models import create_high_performance_kan
        return create_high_performance_kan((128, 196, 3), 26)
    
    elif model_name in ['ickan']:
        from models.architectures.ickan_models import create_high_performance_ickan
        return create_high_performance_ickan((128, 196, 3), 26)
    
    elif model_name in ['wavkan']:
        from models.architectures.wavkan_models import create_high_performance_wavkan
        return create_high_performance_wavkan((128, 196, 3), 26)
    
    elif model_name in ['simple', 'simple_cnn', 'fallback']:
        return create_simple_cnn()
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def test_all_models():
    """Test all available models"""
    print("\n📦 Testing all model architectures...")
    
    models_to_test = [
        'alexnet', 'densenet', 'efficientnet', 'inception', 
        'mobilenet', 'resnet', 'resnet18', 'kan', 'ickan', 
        'wavkan', 'simple_cnn'
    ]
    
    working_models = []
    
    for model_name in models_to_test:
        try:
            model = create_model(model_name)
            params = sum(p.numel() for p in model.parameters())
            print(f"✅ {model_name}: {params:,} parameters")
            working_models.append(model_name)
        except Exception as e:
            print(f"❌ {model_name}: {e}")
    
    print(f"\n🎯 Working models ({len(working_models)}/{len(models_to_test)}): {working_models}")
    return working_models

def run_basic_training(model_name: str):
    """Run basic training demonstration"""
    print(f"\n�� Running basic training demo for {model_name}...")
    
    # Create synthetic data
    batch_size = 32
    features = torch.randn(batch_size, 3, 128, 196)
    labels = torch.randint(0, 26, (batch_size,))
    
    # Create model
    model = create_model(model_name)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simple training loop
    model.train()
    for epoch in range(3):  # Just 3 epochs for demo
        optimizer.zero_grad()
        
        # Handle KAN models (different input format)
        if model_name.lower() in ['kan', 'ickan', 'wavkan']:
            # KAN expects (batch, height, width, channels)
            model_input = features.permute(0, 2, 3, 1)
        else:
            # CNN expects (batch, channels, height, width)
            model_input = features
        
        outputs = model(model_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        
        print(f"Epoch {epoch+1}/3: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    print(f"✅ Training demo completed for {model_name}")

def main():
    parser = argparse.ArgumentParser(description='FSC Meta - Complete Pipeline')
    parser.add_argument('--test', action='store_true', help='Test all models')
    parser.add_argument('--model', type=str, help='Model to test/train')
    parser.add_argument('--train', action='store_true', help='Run training demo')
    parser.add_argument('--list', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    available_models = [
        'alexnet', 'densenet', 'efficientnet', 'inception', 
        'mobilenet', 'resnet', 'resnet18', 'kan', 'ickan', 
        'wavkan', 'simple_cnn'
    ]
    
    if args.list:
        print("📋 Available models:")
        for model in available_models:
            print(f"  • {model}")
        return
    
    if args.test:
        working_models = test_all_models()
        return working_models
    
    if args.model:
        print(f"🎯 Testing model: {args.model}")
        
        try:
            model = create_model(args.model)
            params = sum(p.numel() for p in model.parameters())
            print(f"✅ {args.model} created successfully: {params:,} parameters")
            
            if args.train:
                run_basic_training(args.model)
            
        except Exception as e:
            print(f"❌ {args.model} failed: {e}")
            print(f"💡 Available models: {available_models}")
    
    else:
        print("💡 ESC Meta - Complete Pipeline")
        print("�� Use --list to see available models")
        print("💡 Use --test to test all models")
        print("💡 Use --model <name> to test a specific model")
        print("💡 Use --model <name> --train to run training demo")
        print(f"💡 Available: {len(available_models)} models")

if __name__ == "__main__":
    main()
