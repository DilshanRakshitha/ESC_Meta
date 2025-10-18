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

from features.fsc_original_features import FSCOriginalDataLoader
from models.training.trainer import FSCOriginalTrainer, FSCOriginalCrossValidator
from models.model_factory import SimpleModelFactory

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



class FSCMetaMain:
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        
        self.setup_components()
        

    def load_config(self) -> Dict[str, Any]:
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Config not found: {self.config_path}, using defaults")
            return self.get_default_config()
        
    
    def get_default_config(self) -> Dict[str, Any]:
        
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
            
        self.data_loader = FSCOriginalDataLoader(self.config)
        
        self.model_factory = SimpleModelFactory()
    

    def load_data(self, strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        
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
            print(f"Data loading failed")
            return self.load_data_fallback()
    

    def load_data_fallback(self) -> Tuple[np.ndarray, np.ndarray]:
        
        print("Generating synthetic data for testing...")
        n_samples = 1000
        # Generate 2D spectrogram-like data for CNN models: (samples, channels, height, width)
        # Using typical mel-spectrogram dimensions
        features = np.random.randn(n_samples, 1, 128, 87)  # 1 channel, 128 mel bins, 87 time frames
        labels = np.random.randint(0, 10, n_samples)
        print(f"   Generated synthetic spectrogram data: {features.shape}")
        return features, labels
    
    def create_model(self, model_name: str, input_shape: Tuple, num_classes: int):
        
        try:
            return self.model_factory.create_model(model_name, input_shape, num_classes)
        except Exception as e:
            print(f"Model creation failed: {e}")
            
    
    
    def run_experiment(self, model_name: str = None, strategy_preset: str = None):
        
        if strategy_preset and strategy_preset in self.config['strategy_presets']:
            preset = self.config['strategy_presets'][strategy_preset]
            model_name = preset['model']
            data_strategy = preset['data']
            print(f"Using preset: {strategy_preset}")
        else:
            model_name = model_name or list(self.config['models'].keys())[0]
            data_strategy = 'auto'
            print(f"Using model: {model_name}")
        
        features, labels = self.load_data(data_strategy)
        print(f"Dataset: {len(features)} samples, {len(np.unique(labels))} classes")
        print(f"Feature shape: {features.shape}")
        
        try:
            
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device('cpu')
            def model_creator():
                return self.create_model(model_name, features.shape[1:], len(np.unique(labels)))
            
            cv_trainer = FSCOriginalCrossValidator(
                model_creator_func=model_creator,
                device=device,
                random_state=42
            )
            
            results = cv_trainer.run_kfold_training(features, labels, n_splits=5)
        
        except Exception as e:
            print(f"Training failed: {e}")
        
        # Results
        accuracy = results['mean_accuracy']
        
        print(f"\n" + "=" * 90)
        print(f"EXPERIMENT COMPLETED!")
        print(f"Results:")
        print(f"   Model: {model_name}")
        print(f"   Mean Accuracy: {accuracy:.2f}% ± {results['std_accuracy']:.2f}%")
        print(f"   Best Fold: {results['best_fold_acc']:.2f}%")
        print(f"   Individual Folds: {[f'{acc:.1f}%' for acc in results['individual_accuracies']]}")
        
        print("=" * 90)
        return results


def main():
    
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
    
    try:
        pipeline = FSCMetaMain(args.config)
        
    except Exception as e:
        print(f"Pipeline initialization issues: {e}")
    
    try:
        results = pipeline.run_experiment(
            model_name=args.model,
            strategy_preset=args.preset
        )
        print('model =', args.model, 'strategy_preset =', args.preset)
        print("Experiment completed successfully!")
        return results
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
