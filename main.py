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

def load_fsc22_data(model_name: str = None):
    
    
    data_path = "/home/dilshan/Documents/ESC/temp1/ESC_Meta/data/fsc22/Pickle_Files/aug_ts_ps_mel_features_5_20"
    
    try:
        
        preprocessor = DataPreprocessor(data_path)
        folds_data, _ = preprocessor.prepare_fsc22_data()
        
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
    
    def model_factory():
    
        model_name_lower = model_name.lower()
        input_shape = (128, 196, 3)  
        
        # ========== CNN ARCHITECTURES ==========
        if model_name_lower == 'alexnet' :
            from models.architectures.AlexNet import AlexNet
            return AlexNet(input_size=3, num_classes=num_classes)
        elif model_name_lower == 'densenet' :
            from models.architectures.DenseNet121 import create_densenet121
            return create_densenet121(num_classes=num_classes, input_channels=3)
        elif model_name_lower == 'efficientnet' :
            from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
            return create_efficientnet_v2_b0(num_classes=num_classes, input_channels=3)
        elif model_name_lower == 'inception' :
            from models.architectures.InceptionV3 import create_inception_v3
            return create_inception_v3(num_classes=num_classes, input_channels=3, pretrained=False)
        elif model_name_lower == 'mobilenet' :
            from models.architectures.MobileNetV3Small import create_mobilenet_v3_small
            return create_mobilenet_v3_small(num_classes=num_classes, input_channels=3)
        elif model_name_lower == 'mobilenetv3large' :
            from models.architectures.MobileNetV3Small import create_mobilenet_v3_large
            return create_mobilenet_v3_large(num_classes=num_classes, input_channels=3)
        elif model_name_lower == 'resnet50' :
            from models.architectures.ResNet50V2 import create_resnet50_v2
            return create_resnet50_v2(num_classes=num_classes, input_channels=3)
        elif model_name_lower == 'resnet18' :
            from models.architectures.ResNet50V2 import create_resnet18
            return create_resnet18(num_classes=num_classes, input_channels=3)
        
        # ========== KAN-INSPIRED ARCHITECTURES (Fast, CNN-based) ==========
        elif model_name_lower == 'kan_inspired' :
            from models.architectures.kan_models import create_high_performance_kan
            return create_high_performance_kan(input_shape, num_classes)
        elif model_name_lower == 'ickan_inspired' :
            from models.architectures.ickan_models import create_high_performance_ickan
            return create_high_performance_ickan(input_shape, num_classes)
        elif model_name_lower == 'wavkan_inspired' :
            from models.architectures.wavkan_models import create_high_performance_wavkan
            return create_high_performance_wavkan(input_shape, num_classes)
        
        # ========== KAN ARCHITECTURES ==========
        elif model_name_lower == 'kan' :
            from models.architectures.exact_kan_models import create_exact_kan
            return create_exact_kan(input_shape, num_classes)
        elif model_name_lower == 'kan_fast' :
            from models.architectures.exact_kan_models import create_fast_exact_kan
            return create_fast_exact_kan(input_shape, num_classes, mode='balanced')
        elif model_name_lower == 'kan_memory_safe' :
            from models.architectures.exact_kan_models import create_memory_safe_kan
            return create_memory_safe_kan(input_shape, num_classes, max_memory_gb=6)
        
        # ========== ICKAN ARCHITECTURES ==========
        elif model_name_lower == 'ickan':
            from models.architectures.exact_ickan_models import create_exact_ickan
            return create_exact_ickan(input_shape, num_classes, variant="standard")
        elif model_name_lower == 'ickan_light':
            from models.architectures.exact_ickan_models import create_exact_ickan
            return create_exact_ickan(input_shape, num_classes, variant="light")
        elif model_name_lower == 'ickan_deep' :
            from models.architectures.exact_ickan_models import create_exact_ickan
            return create_exact_ickan(input_shape, num_classes, variant="deep")
        
        # ========== RAPID KAN ARCHITECTURES (Fast Learning) ==========
        elif model_name_lower == 'rapid_kan':
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='efficient')
        elif model_name_lower == 'rapid_kan_lite' :
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='lightweight')
        elif model_name_lower == 'rapid_kan_power' :
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='powerful')
        
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
        global _folds_data_cache
        if _folds_data_cache is None:
            print("No fold data available for proper cross-validation")
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
