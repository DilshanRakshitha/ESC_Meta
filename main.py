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

# Import cross-validation trainer for real FSC22 training
try:
    from models.training.trainer import FSCCrossValidator
    from utils.data_prep import DataPreprocessor
    print("✅ Cross-validation trainer imported")
    CV_TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Cross-validation trainer not available: {e}")
    CV_TRAINER_AVAILABLE = False

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
    """Create model by name with direct imports - Complete Architecture Collection"""
    model_name = model_name.lower()
    input_shape = (128, 196, 3)  # Standard ESC input shape (H, W, C)
    
    # ========== CNN ARCHITECTURES ==========
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
    
    elif model_name in ['mobilenetv3large']:
        from models.architectures.MobileNetV3Small import create_mobilenet_v3_large
        return create_mobilenet_v3_large(num_classes=26, input_channels=3)
    
    elif model_name in ['resnet', 'resnet50', 'resnet50v2']:
        from models.architectures.ResNet50V2 import create_resnet50_v2
        return create_resnet50_v2(num_classes=26, input_channels=3)
    
    elif model_name in ['resnet18']:
        from models.architectures.ResNet50V2 import create_resnet18
        return create_resnet18(num_classes=26, input_channels=3)
    
    # ========== KAN-INSPIRED ARCHITECTURES (Fast, CNN-based) ==========
    elif model_name in ['kan_inspired', 'kan_fast']:
        from models.architectures.kan_models import create_high_performance_kan
        return create_high_performance_kan(input_shape, 26)
    
    elif model_name in ['ickan_inspired', 'ickan_fast']:
        from models.architectures.ickan_models import create_high_performance_ickan
        return create_high_performance_ickan(input_shape, 26)
    
    elif model_name in ['wavkan_inspired', 'wavkan_fast']:
        from models.architectures.wavkan_models import create_high_performance_wavkan
        return create_high_performance_wavkan(input_shape, 26)
    
    # ========== TRUE KAN ARCHITECTURES (Exact implementations) ==========
    # Exact KAN models (heavy computation)
    elif model_name in ['kan', 'exact_kan']:
        from models.architectures.exact_kan_models import create_exact_kan
        return create_exact_kan(input_shape, 26)
    
    elif model_name in ['kan_pure', 'pure_kan']:
        from models.architectures.exact_kan_models import create_pure_kan
        return create_pure_kan(input_shape, 26)
    
    elif model_name in ['kan_fast_exact', 'fast_kan']:
        from models.architectures.exact_kan_models import create_fast_exact_kan
        return create_fast_exact_kan(input_shape, 26, mode='balanced')
    
    elif model_name in ['kan_memory_safe', 'memory_safe_kan']:
        from models.architectures.exact_kan_models import create_memory_safe_kan
        return create_memory_safe_kan(input_shape, 26, max_memory_gb=6)
    
    # Exact ICKAN models
    elif model_name in ['ickan', 'exact_ickan']:
        from models.architectures.exact_ickan_models import create_exact_ickan
        return create_exact_ickan(input_shape, 26, variant="standard")
    
    elif model_name in ['ickan_light', 'light_ickan']:
        from models.architectures.exact_ickan_models import create_exact_ickan
        return create_exact_ickan(input_shape, 26, variant="light")
    
    elif model_name in ['ickan_deep', 'deep_ickan']:
        from models.architectures.exact_ickan_models import create_exact_ickan
        return create_exact_ickan(input_shape, 26, variant="deep")
    
    # ========== SUPERIOR KAN ARCHITECTURES (High Performance) ==========
    elif model_name in ['superior_kan', 'superior_kan_high']:
        from models.architectures.superior_kan_models import create_superior_kan
        return create_superior_kan(input_shape, 26, performance_mode='high')
    
    elif model_name in ['superior_kan_ultra', 'superior_ultra']:
        from models.architectures.superior_kan_models import create_superior_kan
        return create_superior_kan(input_shape, 26, performance_mode='ultra')
    
    elif model_name in ['superior_kan_balanced', 'superior_balanced']:
        from models.architectures.superior_kan_models import create_superior_kan
        return create_superior_kan(input_shape, 26, performance_mode='balanced')
    
    elif model_name in ['superior_kan_safe', 'superior_safe']:
        from models.architectures.superior_kan_models import create_memory_safe_superior_kan
        return create_memory_safe_superior_kan(input_shape, 26, max_memory_gb=4)
    
    # ========== RAPID KAN ARCHITECTURES (Fast Learning) ==========
    elif model_name in ['rapid_kan', 'rapid_kan_efficient']:
        from models.architectures.rapid_kan_models import create_rapid_kan
        return create_rapid_kan(input_shape, 26, performance='efficient')
    
    elif model_name in ['rapid_kan_lite', 'rapid_lite']:
        from models.architectures.rapid_kan_models import create_rapid_kan
        return create_rapid_kan(input_shape, 26, performance='lightweight')
    
    elif model_name in ['rapid_kan_power', 'rapid_power']:
        from models.architectures.rapid_kan_models import create_rapid_kan
        return create_rapid_kan(input_shape, 26, performance='powerful')
    
    # ========== FALLBACK ==========
    elif model_name in ['simple', 'simple_cnn', 'fallback']:
        return create_simple_cnn()
    
    else:
        available_models = [
            # CNN Models
            'alexnet', 'densenet', 'efficientnet', 'inception', 'mobilenet', 'mobilenetv3large', 'resnet', 'resnet18',
            # KAN-Inspired (Fast)
            'kan_inspired', 'ickan_inspired', 'wavkan_inspired',
            # True KAN (Exact)
            'kan', 'kan_pure', 'kan_fast_exact', 'kan_memory_safe',
            # True ICKAN (Exact)
            'ickan', 'ickan_light', 'ickan_deep',
            # Superior KAN
            'superior_kan', 'superior_kan_ultra', 'superior_kan_balanced', 'superior_kan_safe',
            # Rapid KAN
            'rapid_kan', 'rapid_kan_lite', 'rapid_kan_power',
            # Fallback
            'simple_cnn'
        ]
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")

def test_all_models():
    """Test all available models"""
    print("\n📦 Testing all model architectures...")
    
    # Organized model categories for testing
    cnn_models = ['alexnet', 'densenet', 'efficientnet', 'inception', 'mobilenet', 'mobilenetv3large', 'resnet', 'resnet18']
    
    kan_inspired_models = ['kan_inspired', 'ickan_inspired', 'wavkan_inspired']
    
    exact_kan_models = ['kan', 'kan_pure', 'kan_fast_exact', 'kan_memory_safe']
    
    exact_ickan_models = ['ickan', 'ickan_light', 'ickan_deep']
    
    superior_kan_models = ['superior_kan', 'superior_kan_ultra', 'superior_kan_balanced', 'superior_kan_safe']
    
    rapid_kan_models = ['rapid_kan', 'rapid_kan_lite', 'rapid_kan_power']
    
    fallback_models = ['simple_cnn']
    
    # Test different categories
    all_categories = [
        ("CNN Models", cnn_models),
        ("KAN-Inspired (Fast)", kan_inspired_models),
        ("Exact KAN Models", exact_kan_models),
        ("Exact ICKAN Models", exact_ickan_models),
        ("Superior KAN Models", superior_kan_models),
        ("Rapid KAN Models", rapid_kan_models),
        ("Fallback Models", fallback_models)
    ]
    
    working_models = []
    total_models = 0
    
    for category_name, models_list in all_categories:
        print(f"\n🔹 {category_name}:")
        category_working = []
        
        for model_name in models_list:
            total_models += 1
            try:
                model = create_model(model_name)
                params = sum(p.numel() for p in model.parameters())
                print(f"  ✅ {model_name}: {params:,} parameters")
                working_models.append(model_name)
                category_working.append(model_name)
            except Exception as e:
                print(f"  ❌ {model_name}: {str(e)[:100]}...")
        
        print(f"  📊 Category Status: {len(category_working)}/{len(models_list)} working")
    
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Total Working Models: {len(working_models)}/{total_models}")
    print(f"   Success Rate: {len(working_models)/total_models*100:.1f}%")
    print(f"   Working Models: {working_models}")
    
    return working_models

def run_basic_training(model_name: str):
    """Run basic training demonstration"""
    print(f"\n🎯 Running basic training demo for {model_name}...")
    
    # Create synthetic data
    batch_size = 32
    features = torch.randn(batch_size, 3, 128, 196)
    labels = torch.randint(0, 26, (batch_size,))
    
    # Create model
    model = create_model(model_name)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    
    # Special learning rates for different model types
    if any(rapid in model_name.lower() for rapid in ['rapid_kan']):
        # Rapid KAN models prefer higher learning rates
        learning_rate = 0.003
    elif any(superior in model_name.lower() for superior in ['superior_kan']):
        # Superior KAN models prefer moderate learning rates
        learning_rate = 0.002
    elif model_name.lower() in ['kan', 'kan_pure', 'ickan']:
        # Exact models prefer lower learning rates
        learning_rate = 0.001
    else:
        # CNN models standard learning rate
        learning_rate = 0.001
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Simple training loop
    model.train()
    for epoch in range(3):  # Just 3 epochs for demo
        optimizer.zero_grad()
        
        # All models (CNN and KAN) use the same standard PyTorch format
        # (batch, channels, height, width) - KAN models handle format internally
        model_input = features
        
        outputs = model(model_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        
        print(f"Epoch {epoch+1}/3: Loss={loss:.4f}, Accuracy={accuracy:.4f}, LR={learning_rate}")
    
    print(f"✅ Training demo completed for {model_name}")
    return model

def load_fsc22_data(model_name: str = None):
    """Load real FSC22 dataset with proper format for the model type"""
    print("📂 Loading FSC22 dataset...")
    
    data_path = "/home/dilshan/Documents/ESC/temp1/ESC_Meta/data/fsc22/Pickle_Files/aug_ts_ps_mel_features_5_20"
    
    try:
        # Use DataPreprocessor to load FSC22 data
        preprocessor = DataPreprocessor(data_path)
        features, labels = preprocessor.prepare_fsc22_data()
        
        # Convert to numpy arrays
        if isinstance(features, list):
            features = np.array(features)
        if isinstance(labels, list):
            labels = np.array(labels)
        
        # Handle data format based on model type
        is_kan_model = model_name and model_name.lower() in ['kan', 'ickan', 'wavkan', 'exact_kan', 'exact_ickan']
        
        if len(features.shape) == 4 and features.shape[-1] == 3:
            if is_kan_model:
                # KAN models expect (batch, height, width, channels) format
                print(f"📊 Keeping original format for KAN model: {features.shape}")
            else:
                # CNN models expect (batch, channels, height, width) format
                print(f"📊 Transposing data from {features.shape} to CNN format...")
                features = np.transpose(features, (0, 3, 1, 2))  # (B,H,W,C) -> (B,C,H,W)
                print(f"📊 New shape: {features.shape}")
        
        print(f"✅ FSC22 data loaded successfully")
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Labels shape: {labels.shape}")
        print(f"📊 Number of classes: {len(np.unique(labels))}")
        
        return features, labels
        
    except Exception as e:
        print(f"❌ Error loading FSC22 data: {e}")
        return None, None

def create_model_factory(model_name: str, num_classes: int = 26):
    """Create a model factory function for cross-validation with ALL architectures"""
    def model_factory():
        # Create model with correct number of classes for FSC22
        model_name_lower = model_name.lower()
        input_shape = (128, 196, 3)  # Standard ESC input shape
        
        # ========== CNN ARCHITECTURES ==========
        if model_name_lower in ['alexnet', 'alex']:
            from models.architectures.AlexNet import AlexNet
            return AlexNet(input_size=3, num_classes=num_classes)
        elif model_name_lower in ['densenet', 'densenet121']:
            from models.architectures.DenseNet121 import create_densenet121
            return create_densenet121(num_classes=num_classes, input_channels=3)
        elif model_name_lower in ['efficientnet', 'efficientnetv2', 'efficientnetv2b0']:
            from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
            return create_efficientnet_v2_b0(num_classes=num_classes, input_channels=3)
        elif model_name_lower in ['inception', 'inceptionv3']:
            from models.architectures.InceptionV3 import create_inception_v3
            return create_inception_v3(num_classes=num_classes, input_channels=3, pretrained=False)
        elif model_name_lower in ['mobilenet', 'mobilenetv3', 'mobilenetv3small']:
            from models.architectures.MobileNetV3Small import create_mobilenet_v3_small
            return create_mobilenet_v3_small(num_classes=num_classes, input_channels=3)
        elif model_name_lower in ['mobilenetv3large']:
            from models.architectures.MobileNetV3Small import create_mobilenet_v3_large
            return create_mobilenet_v3_large(num_classes=num_classes, input_channels=3)
        elif model_name_lower in ['resnet', 'resnet50', 'resnet50v2']:
            from models.architectures.ResNet50V2 import create_resnet50_v2
            return create_resnet50_v2(num_classes=num_classes, input_channels=3)
        elif model_name_lower in ['resnet18']:
            from models.architectures.ResNet50V2 import create_resnet18
            return create_resnet18(num_classes=num_classes, input_channels=3)
        
        # ========== KAN-INSPIRED ARCHITECTURES (Fast, CNN-based) ==========
        elif model_name_lower in ['kan_inspired', 'kan_fast']:
            from models.architectures.kan_models import create_high_performance_kan
            return create_high_performance_kan(input_shape, num_classes)
        elif model_name_lower in ['ickan_inspired', 'ickan_fast']:
            from models.architectures.ickan_models import create_high_performance_ickan
            return create_high_performance_ickan(input_shape, num_classes)
        elif model_name_lower in ['wavkan_inspired', 'wavkan_fast']:
            from models.architectures.wavkan_models import create_high_performance_wavkan
            return create_high_performance_wavkan(input_shape, num_classes)
        
        # ========== TRUE KAN ARCHITECTURES (Exact implementations) ==========
        elif model_name_lower in ['kan', 'exact_kan']:
            from models.architectures.exact_kan_models import create_exact_kan
            return create_exact_kan(input_shape, num_classes)
        elif model_name_lower in ['kan_pure', 'pure_kan']:
            from models.architectures.exact_kan_models import create_pure_kan
            return create_pure_kan(input_shape, num_classes)
        elif model_name_lower in ['kan_fast_exact', 'fast_kan']:
            from models.architectures.exact_kan_models import create_fast_exact_kan
            return create_fast_exact_kan(input_shape, num_classes, mode='balanced')
        elif model_name_lower in ['kan_memory_safe', 'memory_safe_kan']:
            from models.architectures.exact_kan_models import create_memory_safe_kan
            return create_memory_safe_kan(input_shape, num_classes, max_memory_gb=6)
        
        # ========== TRUE ICKAN ARCHITECTURES ==========
        elif model_name_lower in ['ickan', 'exact_ickan']:
            from models.architectures.exact_ickan_models import create_exact_ickan
            return create_exact_ickan(input_shape, num_classes, variant="standard")
        elif model_name_lower in ['ickan_light', 'light_ickan']:
            from models.architectures.exact_ickan_models import create_exact_ickan
            return create_exact_ickan(input_shape, num_classes, variant="light")
        elif model_name_lower in ['ickan_deep', 'deep_ickan']:
            from models.architectures.exact_ickan_models import create_exact_ickan
            return create_exact_ickan(input_shape, num_classes, variant="deep")
        
        # ========== SUPERIOR KAN ARCHITECTURES (High Performance) ==========
        elif model_name_lower in ['superior_kan', 'superior_kan_high']:
            from models.architectures.superior_kan_models import create_superior_kan
            return create_superior_kan(input_shape, num_classes, performance_mode='high')
        elif model_name_lower in ['superior_kan_ultra', 'superior_ultra']:
            from models.architectures.superior_kan_models import create_superior_kan
            return create_superior_kan(input_shape, num_classes, performance_mode='ultra')
        elif model_name_lower in ['superior_kan_balanced', 'superior_balanced']:
            from models.architectures.superior_kan_models import create_superior_kan
            return create_superior_kan(input_shape, num_classes, performance_mode='balanced')
        elif model_name_lower in ['superior_kan_safe', 'superior_safe']:
            from models.architectures.superior_kan_models import create_memory_safe_superior_kan
            return create_memory_safe_superior_kan(input_shape, num_classes, max_memory_gb=4)
        
        # ========== RAPID KAN ARCHITECTURES (Fast Learning) ==========
        elif model_name_lower in ['rapid_kan', 'rapid_kan_efficient']:
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='efficient')
        elif model_name_lower in ['rapid_kan_lite', 'rapid_lite']:
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='lightweight')
        elif model_name_lower in ['rapid_kan_power', 'rapid_power']:
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='powerful')
        
        # ========== FALLBACK ==========
        elif model_name_lower in ['simple', 'simple_cnn', 'fallback']:
            return create_simple_cnn(num_classes=num_classes)
        
        else:
            raise ValueError(f"Model {model_name} not supported in cross-validation pipeline")
    
    return model_factory

def run_fsc22_cross_validation(model_name: str, n_folds: int = 5, use_gpu: bool = False):
    """Run real FSC22 training with 5-fold cross-validation"""
    print(f"\n🚀 Starting FSC22 5-fold cross-validation for {model_name}")
    
    if not CV_TRAINER_AVAILABLE:
        print("❌ Cross-validation trainer not available")
        return None
    
    # Load real FSC22 data
    features, labels = load_fsc22_data(model_name)
    if features is None or labels is None:
        print("❌ Cannot load FSC22 data. Training aborted.")
        return None
    
    # Get number of classes from the data
    num_classes = len(np.unique(labels))
    print(f"📊 Training on {len(features)} samples with {num_classes} classes")
    
    # Setup device - CPU by default for stability
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu' and use_gpu:
            print("⚠️ GPU requested but CUDA not available, falling back to CPU")
    else:
        device = torch.device('cpu')
    print(f"🖥️ Using device: {device}")
    
    # Create model factory
    model_factory = create_model_factory(model_name, num_classes)
    
    # Test model creation
    try:
        test_model = model_factory()
        params = sum(p.numel() for p in test_model.parameters())
        print(f"✅ Model created: {params:,} parameters")
    except Exception as e:
        print(f"❌ Failed to create {model_name}: {e}")
        return None
    
    # Initialize cross-validation trainer
    print("🔧 Initializing cross-validation trainer...")
    cv_trainer = FSCCrossValidator(
        model_creator_func=model_factory,
        device=device,
        random_state=42
    )
    
    # Run 5-fold cross-validation
    print(f"▶️ Starting {n_folds}-fold cross-validation training...")
    try:
        results = cv_trainer.run_kfold_training(features, labels, n_splits=n_folds)
        
        # Display results
        print(f"\n🎯 {model_name.upper()} - FSC22 5-Fold Cross-Validation Results:")
        print("=" * 70)
        
        fold_accuracies = results['individual_accuracies']
        for i, accuracy in enumerate(fold_accuracies, 1):
            print(f"Fold {i}: {accuracy:.2f}%")
        
        print("=" * 70)
        print(f"📊 Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
        print(f"📊 Best Fold: {results['best_fold_acc']:.2f}%")
        print(f"📊 Worst Fold: {results['worst_fold_acc']:.2f}%")
        print("=" * 70)
        
        # Save results
        results_file = f"fsc22_results_{model_name}.txt"
        with open(results_file, 'w') as f:
            f.write(f"{model_name.upper()} - FSC22 5-Fold Cross-Validation Results\n")
            f.write("=" * 70 + "\n")
            for i, acc in enumerate(fold_accuracies, 1):
                f.write(f"Fold {i}: {acc:.2f}%\n")
            f.write("=" * 70 + "\n")
            f.write(f"Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%\n")
            f.write(f"Best Fold: {results['best_fold_acc']:.2f}%\n")
            f.write(f"Worst Fold: {results['worst_fold_acc']:.2f}%\n")
        
        print(f"💾 Results saved to {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Cross-validation training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='FSC Meta - Complete Pipeline with FSC22 Training')
    parser.add_argument('--test', action='store_true', help='Test all models')
    parser.add_argument('--model', type=str, help='Train model on FSC22 data with 5-fold CV')
    parser.add_argument('--demo', action='store_true', help='Run synthetic training demo')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training (default: CPU)')
    
    args = parser.parse_args()
    
    # Complete list of all available models
    available_models = [
        # CNN Models
        'alexnet', 'densenet', 'efficientnet', 'inception', 
        'mobilenet', 'mobilenetv3large', 'resnet', 'resnet18',
        # KAN-Inspired (Fast)
        'kan_inspired', 'ickan_inspired', 'wavkan_inspired',
        # True KAN (Exact)
        'kan', 'kan_pure', 'kan_fast_exact', 'kan_memory_safe',
        # True ICKAN (Exact)
        'ickan', 'ickan_light', 'ickan_deep',
        # Superior KAN
        'superior_kan', 'superior_kan_ultra', 'superior_kan_balanced', 'superior_kan_safe',
        # Rapid KAN
        'rapid_kan', 'rapid_kan_lite', 'rapid_kan_power',
        # Fallback
        'simple_cnn'
    ]
    
    if args.list:
        print("📋 Available models:")
        for model in available_models:
            print(f"  • {model}")
        return
        
    if args.test:
        test_all_models()
        return
    
    if args.model:
        model_name = args.model.lower()
        
        if model_name not in available_models:
            print(f"❌ Model '{model_name}' not available. Use --list to see available models.")
            return
        
        print(f"🎯 Selected model: {model_name}")
        
        if args.demo:
            # Run synthetic demo training
            print("🚀 Running synthetic training demo")
            run_basic_training(model_name)
        else:
            # Default: Run real FSC22 training with cross-validation
            print("🚀 Starting real FSC22 training with 5-fold cross-validation")
            run_fsc22_cross_validation(model_name, use_gpu=args.gpu)
    
    else:
        print("🎵 ESC Meta - Complete Pipeline with FSC22 Training")
        print("Available commands:")
        print("  --model <name>            Train model on FSC22 data (default)")
        print("  --model <name> --demo     Run synthetic training demo")
        print("  --test                    Test all model architectures")
        print("  --list                    List available models")
        print("  --gpu                     Use GPU for training (default: CPU)")
        print("\\nExamples:")
        print("  python main.py --model alexnet      # Train on real FSC22 data (CPU)")
        print("  python main.py --model alexnet --gpu  # Train with GPU")
        print("  python main.py --model alexnet --demo  # Synthetic demo")
        print("  python main.py --test               # Test all models")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
