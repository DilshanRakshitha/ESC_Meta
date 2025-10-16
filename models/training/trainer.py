"""
Training Utilities and Trainer Classes
Enhanced with FSC Original research methodologies for superior performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioDataset(Dataset):
    """
    Dataset class for audio features
    """
    
    def __init__(self, features: List, labels: List, transform: Optional[Callable] = None):
        """
        Initialize dataset
        
        Args:
            features: List of feature arrays
            labels: List of labels
            transform: Optional transform function
        """
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Convert to tensor if not already
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        
        # Fix tensor dimensions for conv2d: (H, W, C) -> (C, H, W)
        if len(feature.shape) == 3:
            feature = feature.permute(2, 0, 1)  # Move channels to first dimension
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label


class ModelTrainer:
    """
    Comprehensive model trainer with various optimization techniques
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 model_name: str = 'model'):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            device: Device to use for training
            model_name: Name for saving model
        """
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_history = []
        self.val_history = []
    
    def prepare_data_loaders(self, 
                           train_data: List[List],
                           val_data: Optional[List[List]] = None,
                           batch_size: int = 32,
                           num_workers: int = 4) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data loaders from feature data
        
        Args:
            train_data: Training data as [features, labels] pairs
            val_data: Validation data as [features, labels] pairs
            batch_size: Batch size
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Extract features and labels
        train_features = [item[0] for item in train_data]
        train_labels = [item[1] for item in train_data]
        
        train_dataset = AudioDataset(train_features, train_labels)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        val_loader = None
        if val_data:
            val_features = [item[0] for item in val_data]
            val_labels = [item[1] for item in val_data]
            val_dataset = AudioDataset(val_features, val_labels)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
        
        return train_loader, val_loader
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module,
                   scheduler: Optional[Any] = None) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        if scheduler:
            scheduler.step()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate_epoch(self, 
                      val_loader: DataLoader,
                      criterion: nn.Module) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def train(self,
              train_data: List[List],
              val_data: Optional[List[List]] = None,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              optimizer_type: str = 'adam',
              scheduler_type: Optional[str] = 'cosine',
              early_stopping_patience: int = 10,
              save_best: bool = True) -> Dict[str, List]:
        """
        Complete training loop
        
        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer_type: Type of optimizer
            scheduler_type: Type of scheduler
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            
        Returns:
            Training history
        """
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(
            train_data, val_data, batch_size
        )
        
        # Setup optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, 
                                momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup scheduler
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            self.train_history.append(train_metrics)
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validate
            if val_loader:
                val_metrics = self.validate_epoch(val_loader, criterion)
                self.val_history.append(val_metrics)
                
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%")
                
                # Early stopping and model saving
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    patience_counter = 0
                    
                    if save_best:
                        self.save_model(f"{self.model_name}_best.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
                
                # Scheduler step for ReduceLROnPlateau
                if scheduler_type == 'reduce':
                    scheduler.step(val_metrics['loss'])
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def evaluate(self, test_data: List[List], batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data
            batch_size: Batch size
            
        Returns:
            Evaluation metrics
        """
        test_features = [item[0] for item in test_data]
        test_labels = [item[1] for item in test_data]
        
        test_dataset = AudioDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        report = classification_report(all_targets, all_predictions, output_dict=True)
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def save_model(self, filename: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load model state"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        print(f"Model loaded from {filename}")


class CrossValidationTrainer:
    """
    Cross-validation trainer for robust evaluation
    """
    
    def __init__(self, model_class, model_kwargs: Dict[str, Any]):
        """
        Initialize CV trainer
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.fold_results = []
    
    def train_fold(self, 
                   train_data: List[List],
                   val_data: List[List],
                   fold_idx: int,
                   **training_kwargs) -> Dict[str, Any]:
        """
        Train a single fold
        
        Args:
            train_data: Training data for this fold
            val_data: Validation data for this fold
            fold_idx: Fold index
            **training_kwargs: Additional training arguments
            
        Returns:
            Fold results
        """
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}")
        print(f"{'='*60}")
        
        # Create new model for this fold
        model = self.model_class(**self.model_kwargs)
        trainer = ModelTrainer(model, model_name=f"fold_{fold_idx}")
        
        # Train
        history = trainer.train(train_data, val_data, **training_kwargs)
        
        # Evaluate on validation set
        val_results = trainer.evaluate(val_data)
        
        fold_result = {
            'fold': fold_idx,
            'history': history,
            'val_results': val_results,
            'model_state': model.state_dict()
        }
        
        self.fold_results.append(fold_result)
        
        return fold_result
    
    def cross_validate(self,
                      folds_data: List[List[List]],
                      **training_kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            folds_data: List of folds, each containing [features, labels] pairs
            **training_kwargs: Training arguments
            
        Returns:
            Cross-validation results
        """
        n_folds = len(folds_data)
        
        for fold_idx in range(n_folds):
            # Prepare data for this fold
            val_data = folds_data[fold_idx]
            train_data = []
            
            # Combine other folds for training
            for i, fold in enumerate(folds_data):
                if i != fold_idx:
                    train_data.extend(fold)
            
            # Train this fold
            self.train_fold(train_data, val_data, fold_idx, **training_kwargs)
        
        # Aggregate results
        val_accuracies = [result['val_results']['accuracy'] for result in self.fold_results]
        
        cv_results = {
            'fold_results': self.fold_results,
            'mean_accuracy': np.mean(val_accuracies),
            'std_accuracy': np.std(val_accuracies),
            'individual_accuracies': val_accuracies
        }
        
        print(f"\nCross-Validation Results:")
        print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"Individual Fold Accuracies: {val_accuracies}")
        
        return cv_results


class FSCOriginalTrainer:
    """
    FSC Original Research Trainer
    Implements exact methodologies from successful FSC Original paper:
    - BatchNormalization in CNN layers
    - Exponential Learning Rate Decay (0.01 -> 0.0005)
    - Adam Optimizer
    - 5-Fold Cross Validation
    - Early Stopping with patience=10
    - Batch size=64, Epochs=50
    """
    
    def __init__(self, model, device, fold_num=1):
        self.model = model.to(device)
        self.device = device
        self.fold_num = fold_num
        
        # FSC Original proven hyperparameters (fixed learning rate)
        self.base_lr = 0.001  # Reduced from 0.01 to 0.001 for better convergence
        self.final_lr = 0.0001  # Reduced from 0.0005 to 0.0001
        self.batch_size = 64
        self.epochs = 50
        self.patience = 10
        
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"🎯 FSC Original Trainer initialized for fold {fold_num}")
        print(f"   Base LR: {self.base_lr} -> Final LR: {self.final_lr}")
        print(f"   Batch Size: {self.batch_size}, Epochs: {self.epochs}")
        
    def add_batchnorm_to_model(self):
        """Add BatchNorm layers to CNN models if not present (FSC Original uses BatchNorm)"""
        # This would be called during model creation in architectures
        pass
    
    def setup_optimizer(self):
        """Setup optimizer with exponential decay as in FSC Original"""
        # Calculate decay factor for exponential decay: final_lr = base_lr * (gamma^epochs)
        decay_factor = (self.final_lr / self.base_lr) ** (1.0 / self.epochs)
        
        # Adam optimizer as used in FSC Original
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr)
        
        # Exponential decay scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay_factor)
        
        print(f"   Optimizer: Adam (lr={self.base_lr})")
        print(f"   Scheduler: ExponentialLR (gamma={decay_factor:.6f})")
        
    def train_fold(self, train_loader, val_loader):
        """Train one fold using FSC Original methodology"""
        print(f'\n🔄 Training Fold {self.fold_num}...')
        
        # Setup optimizer with FSC Original parameters
        self.setup_optimizer()
        
        # Early stopping tracking
        best_val_acc = 0.0
        patience_counter = 0
        
        # History tracking
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # === TRAINING PHASE ===
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Step scheduler after each epoch (FSC Original approach)
            self.scheduler.step()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # === VALIDATION PHASE ===
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100. * correct / total
            
            # Store metrics
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Early stopping check (FSC Original uses patience=10)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'best_model_fold_{self.fold_num}.pth')
                print(f"   🎉 New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'   Epoch {epoch+1:2d}/{self.epochs}: Train={train_acc:.2f}% Val={val_acc:.2f}% LR={current_lr:.6f}')
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f'   ⏹️ Early stopping at epoch {epoch+1} (patience={self.patience})')
                break
        
        print(f'✅ Fold {self.fold_num} completed: Best Val Acc = {best_val_acc:.2f}%')
        
        return {
            'best_val_acc': best_val_acc,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_epoch': epoch + 1
        }


class FSCOriginalCrossValidator:
    """
    5-Fold Cross Validation using FSC Original methodology
    """
    
    def __init__(self, model_creator_func, device, random_state=42):
        """
        Args:
            model_creator_func: Function that creates a fresh model instance
            device: Device to use
            random_state: Random state for reproducibility
        """
        self.model_creator_func = model_creator_func
        self.device = device
        self.random_state = random_state
        
    def run_kfold_training(self, features, labels, n_splits=5):
        """
        Run 5-fold cross validation using FSC Original methodology
        
        Args:
            features: Input features (numpy array)
            labels: Target labels (numpy array)
            n_splits: Number of folds (FSC Original uses 5)
            
        Returns:
            Dictionary with cross-validation results
        """
        print("=" * 80)
        print("🎯 FSC ORIGINAL 5-FOLD CROSS VALIDATION")
        print(f"📊 Dataset: {len(features)} samples, {len(np.unique(labels))} classes")
        print(f"🔄 Folds: {n_splits}")
        print("=" * 80)
        
        # Use StratifiedKFold to maintain class distribution (better than regular KFold)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        fold_results = []
        
        for fold_num, (train_idx, val_idx) in enumerate(skf.split(features, labels), 1):
            print(f"\n📂 Processing Fold {fold_num}/{n_splits}")
            
            # Create data splits
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            print(f"   Train: {len(X_train)} samples, Val: {len(X_val)} samples")
            
            # Create datasets and loaders with FSC Original batch size
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), 
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), 
                torch.LongTensor(y_val)
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=64,  # FSC Original batch size
                shuffle=True, 
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=64, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            # Create fresh model for this fold
            model = self.model_creator_func()
            
            # Train using FSC Original methodology
            trainer = FSCOriginalTrainer(model, self.device, fold_num)
            fold_result = trainer.train_fold(train_loader, val_loader)
            
            fold_results.append({
                'fold': fold_num,
                **fold_result
            })
        
        # Aggregate results
        val_accuracies = [result['best_val_acc'] for result in fold_results]
        mean_acc = np.mean(val_accuracies)
        std_acc = np.std(val_accuracies)
        
        print("\n" + "=" * 80)
        print("🎉 FSC ORIGINAL CROSS-VALIDATION COMPLETED!")
        print(f"📊 Results Summary:")
        for i, acc in enumerate(val_accuracies, 1):
            print(f"   Fold {i}: {acc:.2f}%")
        print(f"🏆 Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"📈 Best Fold: {max(val_accuracies):.2f}%")
        print(f"📉 Worst Fold: {min(val_accuracies):.2f}%")
        print("=" * 80)
        
        return {
            'fold_results': fold_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'individual_accuracies': val_accuracies,
            'best_fold_acc': max(val_accuracies),
            'worst_fold_acc': min(val_accuracies)
        }
