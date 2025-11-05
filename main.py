import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Global variable to store fold data for proper cross-validation
_folds_data_cache = None

from models.training.trainer import FSCCrossValidator
from utils.data_prep import DataPreprocessor

def create_model(model_name: str):
    model_name = model_name.lower()
    input_shape = (128, 196, 3)
    
    # ========== CNN ARCHITECTURES ==========
    if model_name == 'alexnet' :
        from models.architectures.AlexNet import AlexNet
        return AlexNet(input_size=3, num_classes=26)
    
    elif model_name == 'densenet' :
        from models.architectures.DenseNet121 import create_densenet121
        return create_densenet121(num_classes=26, input_channels=3)
    
    elif model_name == 'efficientnet' :
        from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
        return create_efficientnet_v2_b0(num_classes=26, input_channels=3)

    elif model_name == 'inception':
        from models.architectures.InceptionV3 import create_inception_v3
        return create_inception_v3(num_classes=26, input_channels=3, pretrained=False)

    elif model_name == 'mobilenet':
        from models.architectures.MobileNetV3Small import create_mobilenet_v3_small
        return create_mobilenet_v3_small(num_classes=26, input_channels=3)
    
    elif model_name == 'mobilenetv3large' :
        from models.architectures.MobileNetV3Small import create_mobilenet_v3_large
        return create_mobilenet_v3_large(num_classes=26, input_channels=3)
    
    elif model_name == 'resnet' or model_name == 'resnet50' :
        from models.architectures.ResNet50V2 import create_resnet50_v2
        return create_resnet50_v2(num_classes=26, input_channels=3)
    
    elif model_name == 'resnet18' :
        from models.architectures.ResNet50V2 import create_resnet18
        return create_resnet18(num_classes=26, input_channels=3)
    
    # ========== KAN-INSPIRED ARCHITECTURES (Fast, CNN-based) ==========
    elif model_name == 'kan_inspired' :
        from models.architectures.kan_models import create_high_performance_kan
        return create_high_performance_kan(input_shape, 26)
    
    elif model_name == 'ickan_inspired' :
        from models.architectures.ickan_models import create_high_performance_ickan
        return create_high_performance_ickan(input_shape, 26)
    
    elif model_name == 'wavkan_inspired' :
        from models.architectures.wavkan_models import create_high_performance_wavkan
        return create_high_performance_wavkan(input_shape, 26)
    
    # ========== TRUE KAN ARCHITECTURES (Exact implementations) ==========
    elif model_name == 'kan' :
        from models.architectures.exact_kan_models import create_exact_kan
        return create_exact_kan(input_shape, 26)
    
    elif model_name == 'kan_fast' :
        from models.architectures.exact_kan_models import create_fast_exact_kan
        return create_fast_exact_kan(input_shape, 26, mode='balanced')
    
    elif model_name == 'kan_memory_safe' :
        from models.architectures.exact_kan_models import create_memory_safe_kan
        return create_memory_safe_kan(input_shape, 26, max_memory_gb=6)
    
    # Exact ICKAN models
    elif model_name == 'ickan' :
        from models.architectures.exact_ickan_models import create_exact_ickan
        return create_exact_ickan(input_shape, 26, variant="standard")
    
    elif model_name == 'ickan_light' :
        from models.architectures.exact_ickan_models import create_exact_ickan
        return create_exact_ickan(input_shape, 26, variant="light")
    
    elif model_name == 'ickan_deep' :
        from models.architectures.exact_ickan_models import create_exact_ickan
        return create_exact_ickan(input_shape, 26, variant="deep")
    
    # ========== RAPID KAN ARCHITECTURES (Fast Learning) ==========
    elif model_name == 'rapid_kan' :
        from models.architectures.rapid_kan_models import create_rapid_kan
        return create_rapid_kan(input_shape, 26, performance='efficient')
    
    elif model_name == 'rapid_kan_lite' :
        from models.architectures.rapid_kan_models import create_rapid_kan
        return create_rapid_kan(input_shape, 26, performance='lightweight')
    
    elif model_name == 'rapid_kan_power' :
        from models.architectures.rapid_kan_models import create_rapid_kan
        return create_rapid_kan(input_shape, 26, performance='powerful')
    
    else:
        raise ValueError(f"Unknown model: {model_name}.")

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
    elif model_name.lower() in ['kan', 'ickan']:
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
    """Load real FSC22 dataset with proper fold separation to prevent data leakage"""
    print("📂 Loading FSC22 dataset...")
    
    data_path = "/home/dilshan/Documents/ESC/temp1/ESC_Meta/data/fsc22/Pickle_Files/aug_ts_ps_mel_features_5_20"
    
    try:
        # Use DataPreprocessor to load FSC22 data with proper fold separation
        preprocessor = DataPreprocessor(data_path)
        folds_data, _ = preprocessor.prepare_fsc22_data()
        
        if not folds_data:
            print("❌ No fold data loaded!")
            return None, None
        
        # Extract features and labels from the first fold for getting structure info
        first_fold_data = list(folds_data.values())[0]
        
        # Extract features and labels from fold data
        if isinstance(first_fold_data, list) and len(first_fold_data) > 0:
            if len(first_fold_data[0]) == 2:
                # Format: [[feature, label], [feature, label], ...]
                sample_feature = first_fold_data[0][0]
                sample_label = first_fold_data[0][1]
            else:
                print("❌ Unexpected fold data format")
                return None, None
        else:
            print("❌ Empty or invalid fold data")
            return None, None
        
        # Combine all folds for shape/class info only (for model creation)
        # The actual cross-validation will use separate folds
        all_features = []
        all_labels = []
        
        for fold_num, fold_data in folds_data.items():
            for item in fold_data:
                if len(item) == 2:
                    all_features.append(item[0])
                    all_labels.append(item[1])
        
        # Convert to numpy arrays
        features = np.array(all_features)
        labels = np.array(all_labels)
        
        # Handle data format based on model type
        is_kan_model = model_name and any(kan_type in model_name.lower() 
                                        for kan_type in ['kan', 'ickan', 'wavkan', 'exact_kan', 'exact_ickan'])
        
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
        print(f"🔒 Fold structure preserved for proper cross-validation")
        
        # Store fold structure in a global variable for cross-validation
        global _folds_data_cache
        _folds_data_cache = folds_data
        
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

def run_fsc22_cross_validation(model_name: str, n_folds: int = 5, cuda: bool = False):
    
    features, labels = load_fsc22_data(model_name)
    
    num_classes = len(np.unique(labels))
    
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu' and cuda:
            print("GPU requested but CUDA not available, falling back to CPU")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    model_factory = create_model_factory(model_name, num_classes)
    
    try:
        test_model = model_factory()
        params = sum(p.numel() for p in test_model.parameters())
        print(f"Model created: {params:,} parameters")
    except Exception as e:
        print(f"Failed to create {model_name}: {e}")
        return None
    
    print("Initializing cross-validation trainer...")
    cv_trainer = FSCCrossValidator(
        model_creator_func=model_factory,
        device=device,
        random_state=42
    )
    
    try:
        # Use the new fold-based cross-validation method to prevent data leakage
        global _folds_data_cache
        if _folds_data_cache is None:
            print("❌ No fold data available for proper cross-validation")
            return None
            
        results = cv_trainer.run_kfold_training_with_folds(_folds_data_cache, n_splits=n_folds)
        
        print(f"\n{model_name.upper()} - FSC22 5-Fold Cross-Validation Results:")
        print("=" * 70)
        
        fold_accuracies = results['individual_accuracies']
        for i, accuracy in enumerate(fold_accuracies, 1):
            print(f"Fold {i}: {accuracy:.2f}%")
        
        print("=" * 70)
        print(f"Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
        print(f"Best Fold: {results['best_fold_acc']:.2f}%")
        print("=" * 70)
        
        results_file = f"fsc22_results_{model_name}.txt"
        with open(results_file, 'w') as f:
            f.write(f"{model_name.upper()} - FSC22 5-Fold Cross-Validation Results\n")
            f.write("=" * 70 + "\n")
            for i, acc in enumerate(fold_accuracies, 1):
                f.write(f"Fold {i}: {acc:.2f}%\n")
            f.write("=" * 70 + "\n")
            f.write(f"Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%\n")
            f.write(f"Best Fold: {results['best_fold_acc']:.2f}%\n")
        
        print(f"Results saved to {results_file}")
        
        return results
        
    except Exception as e:
        print(f"Cross-validation training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='FSC Meta - Complete Pipeline with FSC22 Training')
    parser.add_argument('--model', type=str, help='Train model on FSC22 data with 5-fold CV')
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
        # KAN
        'kan', 'kan_fast', 'kan_memory_safe',
        # ICKAN
        'ickan', 'ickan_light', 'ickan_deep',
        # Rapid KAN
        'rapid_kan', 'rapid_kan_lite', 'rapid_kan_power'
    ]
    
    if args.list:
        print("Available models:")
        for model in available_models:
            print(f"  • {model}")
        return
        
    
    if args.model:
        model_name = args.model.lower()
        
        if model_name not in available_models:
            print(f"Model '{model_name}' not available.")
            return
        
        print("Starting FSC22 training with 5-fold cross-validation")
        run_fsc22_cross_validation(model_name, cuda=args.gpu)
    
    else:
        print("Environment Sound Classification")
        print("Available commands:")
        print("  --model <name>            Train model on FSC22 data (default)")
        print("  --list                    List available models")
        print("  --gpu                     Use GPU for training (default: CPU)")
        print("  python main.py --model alexnet      # Train on real FSC22 data (CPU)")
        print("  python main.py --model alexnet --gpu  # Train with GPU")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
