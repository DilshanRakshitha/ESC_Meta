"""
FSC Meta Main Training Script with FSC Original Methodologies
Adopts proven techniques from FSC Original research for superior performance
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project directories to path
project_root = Path(__file__).parent
sys.path.extend([
    str(project_root),
    str(project_root / 'models'),
    str(project_root / 'models' / 'architectures'),
    str(project_root / 'models' / 'training'),
    str(project_root / 'features'),
    str(project_root / 'data'),
    str(project_root / 'utils')
])

from models.architectures.AlexNet import AlexNet
from models.architectures.kan_models import create_kan_model
from models.architectures.wavkan_models import create_wavkan_model
from models.architectures.ickan_models import create_ickan_model
from models.training.trainer import FSCOriginalTrainer, FSCOriginalCrossValidator
from utils.data_prep import load_config
import torchvision.models as torch_models

def load_fsc22_data_original_preprocessing(data_path: str, audio_path: str, max_samples: Optional[int] = None):
    """
    Load FSC22 data with original preprocessing methodology
    Following FSC Original paper preprocessing approach
    """
    print("📂 Loading FSC22 Data with Original Preprocessing...")
    
    # Load dataset CSV
    df = pd.read_csv(data_path)
    
    if max_samples:
        df = df.head(max_samples)
        print(f"   Limited to {max_samples} samples for testing")
    
    features = []
    labels = []
    
    print(f"🔄 Processing {len(df)} audio files...")
    
    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"   Processed {idx}/{len(df)} files ({100*idx/len(df):.1f}%)")
        
        audio_file = os.path.join(audio_path, row['filename'])
        if not os.path.exists(audio_file):
            continue
        
        try:
            # FSC Original preprocessing parameters
            # Load 5 seconds at 20kHz sampling rate
            audio, sr = librosa.load(audio_file, sr=20000, duration=5.0)
            
            # Mel spectrogram with FSC Original parameters
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=128,          # FSC Original uses 128 mel bins
                n_fft=2048,          # FFT window size
                hop_length=512,      # Hop length
                fmax=sr/2           # Max frequency
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0,1] range (FSC Original normalization)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            # Ensure consistent temporal dimension
            target_frames = 196  # FSC Original target frame count
            if mel_spec_norm.shape[1] < target_frames:
                # Zero-pad if too short
                pad_width = target_frames - mel_spec_norm.shape[1]
                mel_spec_norm = np.pad(mel_spec_norm, ((0,0), (0, pad_width)), mode='constant', constant_values=0)
            else:
                # Truncate if too long
                mel_spec_norm = mel_spec_norm[:, :target_frames]
            
            # Create 3-channel image for CNN compatibility (RGB-like)
            # FSC Original approach for CNNs
            mel_spec_3ch = np.stack([mel_spec_norm, mel_spec_norm, mel_spec_norm], axis=0)
            
            features.append(mel_spec_3ch)
            labels.append(row['target'] - 1)  # Convert to 0-based indexing
            
        except Exception as e:
            continue
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"✅ Successfully processed {len(features)} files")
    print(f"📊 Feature shape: {features.shape}")
    print(f"🎯 Label range: {labels.min()} to {labels.max()}")
    print(f"🎵 Number of classes: {len(np.unique(labels))}")
    
    return features, labels


def create_model_with_fsc_original_enhancements(architecture: str, input_shape: tuple, num_classes: int = 26):
    """
    Create model with FSC Original enhancements (BatchNorm already included in AlexNet)
    """
    print(f"🏗️ Creating {architecture} model with FSC Original enhancements...")
    
    if architecture == 'alexnet':
        # Your AlexNet already has BatchNorm - perfect for FSC Original methodology
        model = AlexNet(input_size=input_shape[0], num_classes=num_classes)
        
    elif architecture == 'resnet18':
        model = torch_models.resnet18(pretrained=False)
        # Add BatchNorm (ResNet already has it)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif architecture == 'resnet50':
        model = torch_models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif architecture == 'densenet121':
        model = torch_models.densenet121(pretrained=False)
        # Add dropout for regularization
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier.in_features, num_classes)
        )
        
    elif architecture == 'kan':
        model = create_kan_model(num_classes=num_classes, model_type='high_performance')
        
    elif architecture == 'wavkan':
        model = create_wavkan_model(num_classes=num_classes, model_type='high_performance')
        
    elif architecture == 'ickan':
        model = create_ickan_model(num_classes=num_classes, model_type='high_performance')
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def run_fsc_original_training():
    """Main training function using FSC Original methodologies"""
    
    parser = argparse.ArgumentParser(description='FSC Meta Training with Original Methodologies')
    parser.add_argument('--architecture', type=str, default='alexnet',
                       choices=['alexnet', 'resnet18', 'resnet50', 'densenet121', 'kan', 'wavkan', 'ickan'],
                       help='Model architecture to train')
    parser.add_argument('--cross-validation', action='store_true', default=True,
                       help='Use 5-fold cross-validation (FSC Original approach)')
    parser.add_argument('--data-path', type=str,
                       default='data/fsc22/FSC22.csv',
                       help='Path to FSC22 CSV file')
    parser.add_argument('--audio-path', type=str,
                       default='data/fsc22/wav44',
                       help='Path to audio files directory')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of samples for testing (optional)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    print("=" * 90)
    print("🎵 FSC META TRAINING - ENHANCED WITH FSC ORIGINAL METHODOLOGIES")
    print("📚 Adopting proven techniques from FSC Original research paper")
    print("🔬 Features: BatchNorm, Exponential LR Decay, 5-Fold CV, Early Stopping")
    print("=" * 90)
    
    # Load data with original preprocessing
    features, labels = load_fsc22_data_original_preprocessing(
        args.data_path, 
        args.audio_path, 
        args.max_samples
    )
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Model creation function for cross-validation
    def model_creator():
        return create_model_with_fsc_original_enhancements(
            args.architecture, 
            features.shape[1:], 
            num_classes=len(np.unique(labels))
        )
    
    if args.cross_validation:
        print(f"\n🔄 Running 5-Fold Cross-Validation with {args.architecture}")
        print("📊 Using FSC Original methodology...")
        
        # Run cross-validation with FSC Original techniques
        cv_trainer = FSCOriginalCrossValidator(
            model_creator_func=model_creator,
            device=device,
            random_state=args.random_state
        )
        
        results = cv_trainer.run_kfold_training(features, labels, n_splits=5)
        
        # Save results
        results_file = f'fsc_original_results_{args.architecture}.txt'
        with open(results_file, 'w') as f:
            f.write(f"FSC Original Methodology Results - {args.architecture}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%\n")
            f.write(f"Best Fold: {results['best_fold_acc']:.2f}%\n")
            f.write(f"Individual Folds: {results['individual_accuracies']}\n")
            f.write("\nFSC Original Hyperparameters:\n")
            f.write("- Base LR: 0.01 -> Final LR: 0.0005 (Exponential Decay)\n")
            f.write("- Optimizer: Adam\n")
            f.write("- Batch Size: 64\n")
            f.write("- Max Epochs: 50\n")
            f.write("- Early Stopping: Patience=10\n")
            f.write("- BatchNormalization: Yes\n")
            f.write("- Cross Validation: 5-Fold Stratified\n")
        
        print(f"\n📄 Results saved to: {results_file}")
        
    else:
        # Single training run
        print(f"\n🎯 Training {args.architecture} with FSC Original methodology...")
        
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=args.random_state
        )
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        # Create and train model
        model = model_creator()
        trainer = FSCOriginalTrainer(model, device, fold_num=1)
        results = trainer.train_fold(train_loader, val_loader)
        
        print(f"\n🎉 Training Complete!")
        print(f"🏆 Best Validation Accuracy: {results['best_val_acc']:.2f}%")


if __name__ == '__main__':
    run_fsc_original_training()
