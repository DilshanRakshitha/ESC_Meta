"""
FSC Meta - Unified Pipeline with FSC Original Strategies
Modular implementation of all FSC Original methodologies for identical accuracy
Uses configuration files to specify exact strategies and hyperparameters
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import librosa
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure TensorFlow for optimal performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

# Configure GPU if available
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set GPU as preferred device
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            logger.info(f"GPU configuration successful. Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Enable mixed precision for better performance
            tf.config.optimizer.set_jit(True)  # Enable XLA
            
            return True
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")
            return False
    else:
        logger.info("No GPU found. Using CPU.")
        return False

# Configure GPU at startup
gpu_available = configure_gpu()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fsc_main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FSCOriginalDataLoader:
    """
    FSC Original Data Loader - Loads real FSC22 pickle data exactly as FSC Original does
    """
    
    def __init__(self, data_dir: str = "./data/fsc22"):
        self.data_dir = Path(data_dir)
        self.pickle_dir = self.data_dir / "Pickle_Files"
        logger.info(f"FSC Data Loader initialized with data_dir: {data_dir}")
    
    def load_fsc_data(self) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Load FSC data exactly as FSC Original notebooks do
        
        Returns:
            spect_folds: List of fold data for K-fold CV
            X: Combined features array
            y: Combined labels array
        """
        logger.info("Loading FSC22 data from pickle files...")
        
        if not self.pickle_dir.exists():
            logger.error(f"Pickle directory not found: {self.pickle_dir}")
            raise FileNotFoundError(f"Pickle directory not found: {self.pickle_dir}")
        
        # Try to load fold pickle files (FSC Original method)
        spect_folds = []
        train_spects = []
        
        # Look for fold files with correct naming pattern
        fold_files = []
        
        # Check for aug_ts_ps_mel_features files in subdirectory
        aug_dir = self.pickle_dir / "aug_ts_ps_mel_features_5_20"
        if aug_dir.exists():
            for i in range(1, 6):  # FSC Original uses 5 folds
                fold_file = aug_dir / f'aug_ts_ps_mel_features_5_20_fold{i}'
                if fold_file.exists():
                    fold_files.append(fold_file)
        
        # Fallback: standard fold naming
        if not fold_files:
            for i in range(1, 6):
                fold_file = self.pickle_dir / f'fold_{i}.pkl'
                if fold_file.exists():
                    fold_files.append(fold_file)
        
        if fold_files:
            logger.info(f"Found {len(fold_files)} fold pickle files")
            
            # Load fold data (exact FSC Original method)
            for fold_file in fold_files:
                logger.info(f"Loading {fold_file}")
                with open(fold_file, 'rb') as f:
                    fold_data = pickle.load(f)
                train_spects.extend(fold_data)
                spect_folds.append(fold_data)
        else:
            # Try alternative pickle file structures
            pickle_files = list(self.pickle_dir.glob('*.pkl'))
            if pickle_files:
                logger.info(f"Found {len(pickle_files)} pickle files")
                
                # Load first available pickle file
                with open(pickle_files[0], 'rb') as f:
                    data = pickle.load(f)
                
                # Handle different data structures
                if isinstance(data, list) and len(data) > 0:
                    # List of [feature, class] pairs
                    train_spects = data
                    # Create artificial folds for K-fold CV
                    spect_folds = self._create_artificial_folds(train_spects)
                elif isinstance(data, dict):
                    # Dictionary with features and labels
                    if 'features' in data and 'labels' in data:
                        features = data['features']
                        labels = data['labels']
                        train_spects = [[f, l] for f, l in zip(features, labels)]
                        spect_folds = self._create_artificial_folds(train_spects)
                    else:
                        raise ValueError("Unknown pickle data structure")
                else:
                    raise ValueError("Unknown pickle data structure")
            else:
                logger.error("No pickle files found in the directory")
                raise FileNotFoundError("No pickle files found")
        
        # Convert to DataFrame (exact FSC Original method)
        logger.info(f"Converting {len(train_spects)} samples to DataFrame...")
        train_features_df = pd.DataFrame(train_spects, columns=['feature', 'class'])
        
        X_train = np.array(train_features_df['feature'].tolist())
        y_train = np.array(train_features_df['class'].tolist())
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  X shape: {X_train.shape}")
        logger.info(f"  y shape: {y_train.shape}")
        logger.info(f"  Unique classes: {len(np.unique(y_train))}")
        logger.info(f"  Class distribution: {np.bincount(y_train)}")
        logger.info(f"  Number of folds: {len(spect_folds)}")
        
        return spect_folds, X_train, y_train
    
    def _create_artificial_folds(self, train_spects: List, n_folds: int = 5) -> List:
        """Create artificial folds for K-fold CV if no fold structure exists"""
        logger.info(f"Creating {n_folds} artificial folds for K-fold CV...")
        
        # Convert to arrays for stratified split
        features = [item[0] for item in train_spects]
        labels = [item[1] for item in train_spects]
        
        X = np.array(features)
        y = np.array(labels)
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        spect_folds = []
        
        for train_idx, val_idx in skf.split(X, y):
            fold_data = []
            for idx in val_idx:
                fold_data.append([X[idx], y[idx]])
            spect_folds.append(fold_data)
        
        return spect_folds


class FSCOriginalPipeline:
    """
    Complete FSC Original Pipeline - Exact Implementation
    Uses real FSC22 data and FSC Original methodology for best results
    """
    
    def __init__(self, config_path: str = None):
        """Initialize FSC Original Pipeline"""
        self.config = self._load_config(config_path)
        self.data_loader = FSCOriginalDataLoader(self.config['data']['data_dir'])
        self.results = {}
        
        # Setup directories
        self._setup_directories()
        
        logger.info("FSC Original Pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with FSC Original defaults"""
        default_config = {
            'data': {
                'data_dir': './data/fsc22',
                'num_classes': 26  # FSC22 has 26 classes
            },
            'training': {
                'k_folds': 5,
                'epochs': 50,  # FSC Original uses 50-75 epochs
                'batch_size': 32,  # FSC Original optimal batch size
                'learning_rate': 0.001,  # Conservative LR for stable training
                'optimizer': 'adam',  # FSC Original uses Adam
                'early_stopping_patience': 15,
                'reduce_lr_patience': 5
            },
            'model': {
                'input_shape': (128, 196, 3),  # Actual FSC22 mel-spectrogram shape
                'use_imagenet_weights': False  # Key: FSC Original uses no pretrained weights
            },
            'output': {
                'models_dir': './saved_models',
                'results_dir': './results',
                'plots_dir': './plots/fsc22'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Deep merge
            def deep_merge(default, loaded):
                for key, value in loaded.items():
                    if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                        deep_merge(default[key], value)
                    else:
                        default[key] = value
            
            deep_merge(default_config, loaded_config)
        
        return default_config
    
    def _setup_directories(self):
        """Setup output directories"""
        for dir_path in self.config['output'].values():
            os.makedirs(dir_path, exist_ok=True)
    
    def _get_training_strategy(self):
        """Get the best training strategy based on available hardware"""
        if gpu_available:
            # Use GPU strategy for single GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if len(gpus) > 1:
                # Multi-GPU strategy
                strategy = tf.distribute.MirroredStrategy()
                logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
            else:
                # Single GPU strategy
                strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
                logger.info("Using OneDeviceStrategy with GPU")
        else:
            # CPU strategy
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
            logger.info("Using OneDeviceStrategy with CPU")
        
        return strategy
    
    def create_fsc_original_densenet(self, num_classes: int) -> tf.keras.Model:
        """
        Create FSC Original DenseNet121 model (exact implementation)
        Key: No ImageNet weights - this is critical for FSC Original performance
        """
        logger.info(f"Creating FSC Original DenseNet121 for {num_classes} classes...")
        
        # Use GPU-optimized strategy if available
        strategy = self._get_training_strategy()
        
        with strategy.scope():
            # Exact FSC Original architecture
            base_model = tf.keras.applications.DenseNet121(
                input_shape=self.config['model']['input_shape'],
                weights=None,  # No ImageNet weights - critical!
                include_top=False
            )
            
            # FSC Original top layers
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
            outputs = tf.keras.layers.Dense(
                num_classes, 
                activation='softmax', 
                name='predictions'
            )(x)
            
            model = tf.keras.models.Model(
                base_model.input, 
                outputs, 
                name='DenseNet121_FSC_Original'
            )
        
        logger.info(f"Model created with {model.count_params():,} parameters")
        if gpu_available:
            logger.info("Model created on GPU for accelerated training")
        
        return model
    
    def create_fsc_original_alexnet(self, num_classes: int) -> tf.keras.Model:
        """Create FSC Original AlexNet model"""
        logger.info(f"Creating FSC Original AlexNet for {num_classes} classes...")
        
        # Use GPU-optimized strategy if available
        strategy = self._get_training_strategy()
        
        with strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu', 
                                     input_shape=self.config['model']['input_shape']),
                tf.keras.layers.MaxPooling2D((3, 3), strides=2),
                tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D((3, 3), strides=2),
                tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D((3, 3), strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ], name='AlexNet_FSC_Original')
        
        logger.info(f"AlexNet created with {model.count_params():,} parameters")
        if gpu_available:
            logger.info("AlexNet created on GPU for accelerated training")
        
        return model
    
    def get_fsc_original_callbacks(self, model_save_path: str) -> List[tf.keras.callbacks.Callback]:
        """Get FSC Original training callbacks"""
        callbacks = [
            # Early Stopping (FSC Original settings)
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce LR on Plateau (FSC Original settings)
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['training']['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model Checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_fsc_original_model(self, model_name: str, spect_folds: List, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train model using exact FSC Original methodology with GPU optimization
        """
        logger.info(f"Training {model_name} with FSC Original methodology...")
        
        num_classes = len(np.unique(y))
        acc_per_fold = []
        loss_per_fold = []
        
        # Get training strategy for GPU optimization
        strategy = self._get_training_strategy()
        
        # K-fold Cross Validation (exact FSC Original method)
        for fold_no in range(1, self.config['training']['k_folds'] + 1):
            logger.info(f"Training fold {fold_no}/{self.config['training']['k_folds']}...")
            
            # Get fold data (exact FSC Original method)
            X_train_cv, X_valid_cv, y_train_cv, y_valid_cv = self._get_train_valid_data(spect_folds, fold_no)
            
            logger.info(f"Fold {fold_no} - Train: {X_train_cv.shape}, Valid: {X_valid_cv.shape}")
            
            with strategy.scope():
                # Create model
                if model_name == 'DenseNet121':
                    model = self.create_fsc_original_densenet(num_classes)
                elif model_name == 'AlexNet':
                    model = self.create_fsc_original_alexnet(num_classes)
                else:
                    raise ValueError(f"Model {model_name} not implemented")
                
                # Create optimizer (FSC Original settings)
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.config['training']['learning_rate']
                )
                
                # Compile model
                model.compile(
                    optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            # Setup model saving
            model_save_path = os.path.join(
                self.config['output']['models_dir'],
                f'{model_name}_fold_{fold_no}.h5'
            )
            
            # Get callbacks
            callbacks = self.get_fsc_original_callbacks(model_save_path)
            
            # Train model (exact FSC Original settings with GPU acceleration)
            logger.info(f"Training on {'GPU' if gpu_available else 'CPU'}...")
            history = model.fit(
                X_train_cv, y_train_cv,
                batch_size=self.config['training']['batch_size'],
                epochs=self.config['training']['epochs'],
                validation_data=(X_valid_cv, y_valid_cv),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            scores = model.evaluate(X_valid_cv, y_valid_cv, verbose=0)
            fold_accuracy = scores[1] * 100
            fold_loss = scores[0]
            
            logger.info(f'Fold {fold_no} - Loss: {fold_loss:.4f}, Accuracy: {fold_accuracy:.2f}%')
            
            acc_per_fold.append(fold_accuracy)
            loss_per_fold.append(fold_loss)
            
            # Clean up memory
            del model, X_train_cv, X_valid_cv, y_train_cv, y_valid_cv
            gc.collect()
            
            # Clear GPU memory if using GPU
            if gpu_available:
                tf.keras.backend.clear_session()
        
        # Calculate average performance
        avg_accuracy = np.mean(acc_per_fold)
        std_accuracy = np.std(acc_per_fold)
        avg_loss = np.mean(loss_per_fold)
        
        results = {
            'model_name': model_name,
            'fold_accuracies': acc_per_fold,
            'fold_losses': loss_per_fold,
            'average_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'average_loss': avg_loss,
            'hardware_used': 'GPU' if gpu_available else 'CPU'
        }
        
        logger.info(f"{model_name} training completed!")
        logger.info(f"Average Accuracy: {avg_accuracy:.2f}% (±{std_accuracy:.2f}%)")
        logger.info(f"Training performed on: {'GPU' if gpu_available else 'CPU'}")
        
        return results
    
    def _get_train_valid_data(self, spects_folds: List, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get training and validation data for a specific fold (exact FSC method)"""
        train_spects = []
        valid_spects = None
        
        for i in range(5):
            if i + 1 != fold:
                train_spects.extend(spects_folds[i])
            else:
                valid_spects = spects_folds[i]
        
        # Convert to DataFrames (exact FSC method)
        train_df = pd.DataFrame(train_spects, columns=['feature', 'class'])
        valid_df = pd.DataFrame(valid_spects, columns=['feature', 'class'])
        
        # Extract arrays
        X_train_cv = np.array(train_df['feature'].tolist())
        y_train_cv = np.array(train_df['class'].tolist())
        X_valid_cv = np.array(valid_df['feature'].tolist())
        y_valid_cv = np.array(valid_df['class'].tolist())
        
        return X_train_cv, X_valid_cv, y_train_cv, y_valid_cv
    
    def run_fsc_original_pipeline(self) -> Dict:
        """
        Run complete FSC Original pipeline with best models
        """
        logger.info("=== STARTING FSC ORIGINAL PIPELINE ===")
        
        try:
            # Step 1: Load real FSC22 data
            logger.info("Step 1: Loading FSC22 data...")
            spect_folds, X, y = self.data_loader.load_fsc_data()
            
            # Update input shape based on actual data
            actual_input_shape = X.shape[1:]  # Remove batch dimension
            logger.info(f"Updating input shape to match data: {actual_input_shape}")
            self.config['model']['input_shape'] = actual_input_shape
            
            # Step 2: Train best FSC Original models
            logger.info("Step 2: Training FSC Original models...")
            
            # Train DenseNet121 (FSC Original best performer)
            densenet_results = self.train_fsc_original_model('DenseNet121', spect_folds, X, y)
            self.results['DenseNet121'] = densenet_results
            
            # Train AlexNet (FSC Original comparison model)
            alexnet_results = self.train_fsc_original_model('AlexNet', spect_folds, X, y)
            self.results['AlexNet'] = alexnet_results
            
            # Step 3: Save results and create plots
            logger.info("Step 3: Saving results and creating plots...")
            self._save_results_and_plots()
            
            # Step 4: Print final summary
            self._print_final_summary()
            
            logger.info("=== FSC ORIGINAL PIPELINE COMPLETED ===")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _save_results_and_plots(self):
        """Save results and create comparison plots"""
        # Save results to pickle
        results_path = os.path.join(self.config['output']['results_dir'], 'fsc_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Create comparison plots
        self._create_comparison_plots()
    
    def _create_comparison_plots(self):
        """Create comparison plots for all models"""
        if not self.results:
            return
        
        plots_dir = self.config['output']['plots_dir']
        
        # Model comparison plot
        plt.figure(figsize=(12, 8))
        
        model_names = []
        avg_accuracies = []
        std_accuracies = []
        
        for model_name, results in self.results.items():
            model_names.append(model_name)
            avg_accuracies.append(results['average_accuracy'])
            std_accuracies.append(results['std_accuracy'])
        
        bars = plt.bar(model_names, avg_accuracies, yerr=std_accuracies, 
                      capsize=5, color=['skyblue', 'lightcoral'], alpha=0.7)
        
        plt.title('FSC Original Models - Performance Comparison', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc, std in zip(bars, avg_accuracies, std_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                    f'{acc:.2f}%\n±{std:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fsc_model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual fold results
        plt.figure(figsize=(15, 6))
        
        for i, (model_name, results) in enumerate(self.results.items()):
            plt.subplot(1, len(self.results), i+1)
            fold_numbers = list(range(1, len(results['fold_accuracies']) + 1))
            bars = plt.bar(fold_numbers, results['fold_accuracies'], 
                          color='skyblue' if model_name == 'DenseNet121' else 'lightcoral', alpha=0.7)
            
            plt.axhline(y=results['average_accuracy'], color='red', linestyle='--', 
                       label=f'Avg: {results["average_accuracy"]:.2f}%')
            
            plt.title(f'{model_name}')
            plt.xlabel('Fold')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, acc in zip(bars, results['fold_accuracies']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fsc_fold_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to: {plots_dir}")
    
    def _print_final_summary(self):
        """Print final results summary"""
        logger.info("\n" + "="*80)
        logger.info("FSC ORIGINAL PIPELINE - FINAL RESULTS")
        logger.info("="*80)
        
        for model_name, results in self.results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  Average Accuracy: {results['average_accuracy']:.4f}% (±{results['std_accuracy']:.4f}%)")
            logger.info(f"  Best Fold: {max(results['fold_accuracies']):.4f}%")
            logger.info(f"  Fold Accuracies: {[f'{acc:.2f}%' for acc in results['fold_accuracies']]}")
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['average_accuracy'])
        logger.info(f"\nBEST MODEL: {best_model[0]} with {best_model[1]['average_accuracy']:.4f}% accuracy")
        
        logger.info("\nConfiguration Used:")
        logger.info(f"  Data: Real FSC22 pickle files")
        logger.info(f"  Classes: {self.config['data']['num_classes']}")
        logger.info(f"  K-Folds: {self.config['training']['k_folds']}")
        logger.info(f"  Epochs: {self.config['training']['epochs']}")
        logger.info(f"  Batch Size: {self.config['training']['batch_size']}")
        logger.info(f"  Learning Rate: {self.config['training']['learning_rate']}")
        logger.info(f"  ImageNet Weights: {self.config['model']['use_imagenet_weights']}")
        logger.info(f"  Hardware: {'GPU' if gpu_available else 'CPU'}")
        
        # GPU information
        if gpu_available:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            logger.info(f"  GPU Details: {len(gpus)} GPU(s) - {[gpu.name for gpu in gpus]}")
        
        logger.info("="*80)


def main():
    """Main function to run FSC Original pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FSC Original Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--models', nargs='+', default=['DenseNet121', 'AlexNet'], 
                       help='Models to train')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FSCOriginalPipeline(config_path=args.config)
    
    # Run pipeline
    results = pipeline.run_fsc_original_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()
