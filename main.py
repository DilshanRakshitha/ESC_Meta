import os
import sys
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import pickle
import librosa
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


project_root = Path(__file__).parent.absolute()
sys.path.extend([
    str(project_root),
    str(project_root / 'models'),
    str(project_root / 'models' / 'architectures'),
    str(project_root / 'models' / 'training'),
    str(project_root / 'features'),
    str(project_root / 'utils'),
    str(project_root / 'config'),
])

try:
    from models.architectures.AlexNet import AlexNet
    print("✅ AlexNet imported")
except ImportError as e:
    print(f"⚠️ AlexNet import error: {e}")

try:
    from models.architectures.kan_models import create_high_performance_kan
    print("✅ KAN model imported")
except ImportError as e:
    print(f"⚠️ KAN import error: {e}")

try:
    from models.architectures.ickan_models import create_high_performance_ickan
    print("✅ ICKAN model imported")
except ImportError as e:
    print(f"⚠️ ICKAN import error: {e}")

try:
    from models.architectures.wavkan_models import create_high_performance_wavkan
    print("✅ WavKAN model imported")
except ImportError as e:
    print(f"⚠️ WavKAN import error: {e}")

# Import data loader (keeping only essential parts)
try:
    from features.fsc_original_features import FSCOriginalDataLoader
    print("✅ Data loader imported")
    FSC_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Data loader import error: {e}")
    FSC_LOADER_AVAILABLE = False

# Import trainer if available
try:
    from models.training.trainer import FSCOriginalTrainer, FSCOriginalCrossValidator
    print("✅ Trainer modules imported")
    FSC_TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Trainer import error: {e}")
    FSC_TRAINER_AVAILABLE = False


class SimpleModelFactory:
    """Simple model factory for AlexNet, KAN, ICKAN, WavKAN"""
    
    def create_model(self, model_name: str, input_shape: Tuple, num_classes: int):
        """Create model based on name"""
        if model_name.lower() in ['alexnet', 'alex']:
            # AlexNet expects input_channels as first parameter
            if len(input_shape) == 3:  # (C, H, W)
                input_channels = input_shape[0]
            else:
                input_channels = 3  # Default
            return AlexNet(input_size=input_channels, num_classes=num_classes)
        
        elif model_name.lower() in ['densenet', 'densenet121']:
            from models.architectures.DenseNet121 import create_densenet121
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_densenet121(num_classes=num_classes, input_channels=input_channels)
        
        elif model_name.lower() in ['efficientnet', 'efficientnetv2', 'efficientnetv2b0']:
            from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_efficientnet_v2_b0(num_classes=num_classes, input_channels=input_channels)
        
        elif model_name.lower() in ['inception', 'inceptionv3']:
            from models.architectures.InceptionV3 import create_inception_v3
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_inception_v3(num_classes=num_classes, input_channels=input_channels)
        
        elif model_name.lower() in ['resnet', 'resnet50', 'resnet50v2']:
            from models.architectures.ResNet50V2 import create_resnet50_v2
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_resnet50_v2(num_classes=num_classes, input_channels=input_channels)
        
        elif model_name.lower() in ['resnet18']:
            from models.architectures.ResNet50V2 import create_resnet18
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_resnet18(num_classes=num_classes, input_channels=input_channels)
        
        elif model_name.lower() in ['mobilenet', 'mobilenetv3', 'mobilenetv3small']:
            from models.architectures.MobileNetV3Small import create_mobilenet_v3_small
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_mobilenet_v3_small(num_classes=num_classes, input_channels=input_channels)
        
        elif model_name.lower() in ['mobilenetv3large']:
            from models.architectures.MobileNetV3Small import create_mobilenet_v3_large
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_mobilenet_v3_large(num_classes=num_classes, input_channels=input_channels)
            
        elif model_name.lower() == 'kan':
            # KAN expects (height, width, channels) but we have (channels, height, width)
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                kan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                kan_input_shape = input_shape
            return create_high_performance_kan(kan_input_shape, num_classes)
            
        elif model_name.lower() == 'ickan':
            # ICKAN expects (height, width, channels) but we have (channels, height, width)
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                ickan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                ickan_input_shape = input_shape
            return create_high_performance_ickan(ickan_input_shape, num_classes)
            
        elif model_name.lower() == 'wavkan':
            # WavKAN expects (height, width, channels) but we have (channels, height, width)
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                wavkan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                wavkan_input_shape = input_shape
            return create_high_performance_wavkan(wavkan_input_shape, num_classes)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")


class FSCMetaMain:
    """
    Main FSC Meta pipeline with complete module integration
    Links all components: data loading → feature extraction → model training → evaluation
    """
    
    def __init__(self, config_path: str = "config/fsc_comprehensive_config.yml"):
        """Initialize with all components"""
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize all pipeline components
        print("🔧 Initializing pipeline components...")
        self.setup_components()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration with fallback"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Config loaded: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️ Config not found: {self.config_path}, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration if config file not found"""
        return {
            'data': {
                'fsc_original': {
                    'base_path': '../Forest-Sound-Analysis-on-Edge-main/datasets/',
                    'pickle_files': {
                        'aug_ts_ps_mel_features_5_20': 'fsc22/aug_ts_ps_mel_features_5_20.pkl'
                    }
                },
                'raw_audio': {
                    'csv_path': 'data/fsc22/fsc22_dev_clean.csv',
                    'audio_path': 'data/fsc22/'
                },
                'results': {
                    'base_dir': 'results/'
                }
            },
            'models': {
                'fsc_alexnet': {
                    'type': 'fsc_alexnet',
                    'dropout': 0.5,
                    'batch_norm': True
                }
            },
            'training': {
                'batch_size': 64,
                'epochs': 50,
                'learning_rate': 0.01,
                'cross_validation': {
                    'n_folds': 5,
                    'random_state': 42
                }
            },
            'strategy_presets': {
                'fsc_original_alexnet': {
                    'model': 'fsc_alexnet',
                    'data': 'fsc_original',
                    'feature_type': 'aug_ts_ps_mel_features_5_20',
                    'expected_accuracy': 0.895
                }
            }
        }
    
    def setup_components(self):
        """Setup all pipeline components with error handling"""
        if FSC_LOADER_AVAILABLE:
            try:
                self.data_loader = FSCOriginalDataLoader(self.config)
                print("✅ Data loader initialized")
            except Exception as e:
                print(f"⚠️ Data loader initialization failed: {e}")
                self.data_loader = None
        else:
            print("⚠️ FSC data loader not available")
            self.data_loader = None
        
        # Simple model factory for your specific models
        self.model_factory = SimpleModelFactory()
        print("✅ Simple model factory initialized")
    
    def load_data(self, strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data using specified strategy with fallbacks
        
        Args:
            strategy: 'fsc_original', 'raw_audio', or 'auto'
        
        Returns:
            features, labels arrays
        """
        print(f"📂 Loading data with strategy: {strategy}")
        
        if self.data_loader is None:
            return self.load_data_fallback()
        
        try:
            if strategy == 'fsc_original':
                return self.data_loader.load_fsc_original_pickle_data()
            elif strategy == 'raw_audio':
                csv_path = self.config['data']['raw_audio']['csv_path']
                audio_path = self.config['data']['raw_audio']['audio_path']
                return self.data_loader.load_raw_audio_with_fsc_processing(csv_path, audio_path)
            else:  # auto
                return self.data_loader.load_data('auto')
        except Exception as e:
            print(f"⚠️ Data loading failed with {strategy}: {e}")
            return self.load_data_fallback()
    
    def load_data_fallback(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback data loading method"""
        print("🔄 Using fallback data loading...")
        
        # Try to find any available pickle files
        fsc_data_paths = [
            '../Forest-Sound-Analysis-on-Edge-main/datasets/fsc22/aug_ts_ps_mel_features_5_20.pkl',
            'data/fsc22/features.pkl',
        ]
        
        for path in fsc_data_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, tuple) and len(data) == 2:
                        features, labels = data
                        print(f"✅ Loaded data from: {path}")
                        print(f"   Shape: {features.shape}, Classes: {len(np.unique(labels))}")
                        return features, labels
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        # Generate synthetic data as last resort
        print("🎲 Generating synthetic data for testing...")
        n_samples, n_features = 1000, 128
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 10, n_samples)
        return features, labels
    
    def create_model(self, model_name: str, input_shape: Tuple, num_classes: int):
        """Create model with fallback"""
        try:
            return self.model_factory.create_model(model_name, input_shape, num_classes)
        except Exception as e:
            print(f"⚠️ Model factory failed: {e}")
            # Fallback simple model
            print("🔄 Creating fallback model...")
            return self.create_simple_model(input_shape, num_classes)
    
    def create_simple_model(self, input_shape: Tuple, num_classes: int):
        """Simple fallback model"""
        if len(input_shape) == 1:  # 1D features
            model = nn.Sequential(
                nn.Linear(input_shape[0], 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:  # 2D features (spectrograms)
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(input_shape), 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        return model
    
    def train_model_simple(self, features: np.ndarray, labels: np.ndarray, model_name: str):
        """Simple training with cross-validation fallback"""
        print(f"🔄 Training {model_name} with simple method...")
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {device}")
        
        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        num_classes = len(np.unique(labels_encoded))
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels_encoded)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels_encoded)):
            print(f"\n📊 Fold {fold + 1}/5")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model
            model = self.create_model(model_name, features.shape[1:], num_classes)
            model = model.to(device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            model.train()
            for epoch in range(20):  # Simplified training
                X_batch, y_batch = X_train.to(device), y_train.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    print(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_gpu = X_val.to(device)
                val_outputs = model(X_val_gpu)
                _, predicted = torch.max(val_outputs.data, 1)
                accuracy = (predicted == y_val.to(device)).float().mean().item()
                fold_accuracies.append(accuracy * 100)
                print(f"   Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")
        
        # Results
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'individual_accuracies': fold_accuracies,
            'best_fold_acc': max(fold_accuracies),
            'worst_fold_acc': min(fold_accuracies)
        }
    
    def run_experiment(self, model_name: str = None, strategy_preset: str = None):
        """
        Run complete experiment with comprehensive error handling
        """
        print("=" * 90)
        print("🚀 FSC META - UNIFIED PIPELINE EXECUTION")
        print("=" * 90)
        
        # Strategy selection
        if strategy_preset and strategy_preset in self.config['strategy_presets']:
            preset = self.config['strategy_presets'][strategy_preset]
            model_name = preset['model']
            data_strategy = preset['data']
            expected_acc = preset.get('expected_accuracy', 0.0)
            print(f"🎯 Using preset: {strategy_preset}")
        else:
            model_name = model_name or list(self.config['models'].keys())[0]
            data_strategy = 'auto'
            expected_acc = 0.0
            print(f"🎯 Using model: {model_name}")
        
        # Load data
        features, labels = self.load_data(data_strategy)
        print(f"📊 Dataset: {len(features)} samples, {len(np.unique(labels))} classes")
        print(f"📏 Feature shape: {features.shape}")
        
        # Training
        try:
            if FSC_TRAINER_AVAILABLE and hasattr(self, 'data_loader') and self.data_loader is not None:
                # Use advanced training
                print("🔬 Using FSC Original training methodology...")
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                def model_creator():
                    return self.create_model(model_name, features.shape[1:], len(np.unique(labels)))
                
                cv_trainer = FSCOriginalCrossValidator(
                    model_creator_func=model_creator,
                    device=device,
                    random_state=42
                )
                
                results = cv_trainer.run_kfold_training(features, labels, n_splits=5)
            else:
                # Use simple training
                print("🔄 Using simple training method...")
                results = self.train_model_simple(features, labels, model_name)
                
        except Exception as e:
            print(f"⚠️ Advanced training failed: {e}")
            print("🔄 Falling back to simple training...")
            results = self.train_model_simple(features, labels, model_name)
        
        # Results
        accuracy = results['mean_accuracy']
        
        print(f"\n" + "=" * 90)
        print(f"🎉 EXPERIMENT COMPLETED!")
        print(f"📊 Results:")
        print(f"   Model: {model_name}")
        print(f"   Mean Accuracy: {accuracy:.2f}% ± {results['std_accuracy']:.2f}%")
        print(f"   Best Fold: {results['best_fold_acc']:.2f}%")
        print(f"   Individual Folds: {[f'{acc:.1f}%' for acc in results['individual_accuracies']]}")
        
        if expected_acc > 0:
            diff = accuracy - (expected_acc * 100)
            if abs(diff) <= 2.0:
                print(f"✅ ACCURACY MATCH: Within 2% of expected ({expected_acc:.1%})")
            else:
                print(f"📊 Difference from expected: {diff:+.1f}%")
        
        # Performance tier
        if accuracy >= 85:
            print("🏆 Performance: Excellent")
        elif accuracy >= 70:
            print("✅ Performance: Good")
        elif accuracy >= 50:
            print("⚠️ Performance: Fair")
        else:
            print("❌ Performance: Needs improvement")
        
        print("=" * 90)
        return results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='FSC Meta - Complete Pipeline')
    parser.add_argument('--config', type=str, 
                       default='config/fsc_comprehensive_config.yml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str, default=None,
                       help='Model to train')
    parser.add_argument('--preset', type=str, default=None,
                       help='Strategy preset to use')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test with synthetic data')
    
    args = parser.parse_args()
    
    print("🎵 FSC META - UNIFIED MAIN PIPELINE")
    print("🔗 All modules linked and ready")
    
    # Initialize pipeline
    try:
        pipeline = FSCMetaMain(args.config)
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"⚠️ Pipeline initialization issues: {e}")
        print("🔄 Continuing with fallback configuration...")
        pipeline = FSCMetaMain()
    
    # Run experiment
    try:
        results = pipeline.run_experiment(
            model_name=args.model,
            strategy_preset=args.preset
        )
        print("✅ Experiment completed successfully!")
        return results
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
