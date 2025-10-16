"""
Objective Function for Hyperparameter Optimization
Defines the objective function that Optuna will optimize
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import optuna
import logging

class ObjectiveFunction:
    """Objective function for hyperparameter optimization"""
    
    def __init__(self, model_factory, data, config):
        """
        Initialize objective function
        
        Args:
            model_factory: Function to create model with given hyperparameters
            data: Tuple of (features, labels)
            config: OptimizationConfig instance
        """
        self.model_factory = model_factory
        self.features, self.labels = data
        self.config = config
        
        # Configuration
        self.cv_config = self.config.get_cv_config()
        self.training_config = self.config.get_training_config()
        
        # Setup
        self.device = self._get_device()
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(np.unique(self.encoded_labels))
        
        print(f"🎯 Objective function initialized")
        print(f"   Data: {len(self.features)} samples, {self.num_classes} classes")
        print(f"   Device: {self.device}")
        print(f"   CV Folds: {self.cv_config['n_splits']}")
    
    def _get_device(self):
        """Get device for training"""
        device_config = self.training_config.get('device', 'auto')
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cpu':
            return torch.device('cpu')
        else:
            return torch.device(device_config)
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Mean cross-validation accuracy
        """
        try:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial)
            
            # Perform cross-validation
            cv_scores = self._cross_validate(params, trial)
            
            # Return mean accuracy
            mean_accuracy = np.mean(cv_scores)
            
            # Log trial results
            trial.set_user_attr('cv_scores', cv_scores)
            trial.set_user_attr('std_accuracy', np.std(cv_scores))
            
            return mean_accuracy
            
        except Exception as e:
            logging.error(f"Trial failed: {e}")
            return 0.0  # Return low score for failed trials
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for the trial"""
        params = {}
        
        # Get hyperparameter ranges for the model
        model_name = self.model_factory.__name__.split('_')[-1] if hasattr(self.model_factory, '__name__') else 'unknown'
        hp_ranges = self.config.get_hyperparameters_for_model(model_name)
        
        for param_name, param_config in hp_ranges.items():
            if param_config['type'] == 'uniform':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'loguniform':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], log=True
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
        
        return params
    
    def _cross_validate(self, params: Dict[str, Any], trial: optuna.Trial) -> List[float]:
        """Perform cross-validation with given hyperparameters"""
        cv_scores = []
        
        # Create cross-validation folds
        skf = StratifiedKFold(
            n_splits=self.cv_config['n_splits'],
            shuffle=self.cv_config['shuffle'],
            random_state=self.cv_config['random_state']
        )
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.features, self.encoded_labels)):
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Split data
            X_train = self.features[train_idx]
            X_val = self.features[val_idx]
            y_train = self.encoded_labels[train_idx]
            y_val = self.encoded_labels[val_idx]
            
            # Train and evaluate fold
            accuracy = self._train_fold(X_train, X_val, y_train, y_val, params, trial, fold)
            cv_scores.append(accuracy)
            
            # Report intermediate value for pruning
            trial.report(accuracy, fold)
        
        return cv_scores
    
    def _train_fold(self, X_train, X_val, y_train, y_val, params, trial, fold) -> float:
        """Train a single fold"""
        # Create model
        model = self.model_factory(
            input_shape=self.features.shape[1:],
            num_classes=self.num_classes,
            **{k: v for k, v in params.items() if k not in ['learning_rate', 'batch_size', 'optimizer', 'weight_decay']}
        )
        model = model.to(self.device)
        
        # Setup optimizer
        optimizer = self._create_optimizer(model, params)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Training parameters
        batch_size = params.get('batch_size', 64)
        max_epochs = self.training_config['max_epochs']
        patience = self.training_config['patience']
        
        best_val_acc = 0.0
        patience_counter = 0
        
        # Training loop
        for epoch in range(max_epochs):
            model.train()
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, predicted = torch.max(val_outputs.data, 1)
                val_acc = (predicted == y_val_tensor).float().mean().item() * 100
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_acc
    
    def _create_optimizer(self, model, params):
        """Create optimizer based on parameters"""
        optimizer_name = params.get('optimizer', 'adam')
        learning_rate = params.get('learning_rate', 0.001)
        weight_decay = params.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
