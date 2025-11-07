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
    
    def __init__(self, features: List, labels: List, transform: Optional[Callable] = None):
        """
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


class SimpleModelTrainer:
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cpu',  # Default to CPU for stability
                 model_name: str = 'model'):
        """
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
        
        Args:
            train_data: Training data as [features, labels] pairs
            val_data: Validation data as [features, labels] pairs
            batch_size: Batch size
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        
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
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
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
              batch_size: int = 64,
              learning_rate: float = 0.001,
              optimizer_type: str = 'adam',
              scheduler_type: Optional[str] = 'cosine',
              early_stopping_patience: int = 10,
              save_best: bool = True) -> Dict[str, List]:
        """
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
        
        train_loader, val_loader = self.prepare_data_loaders(
            train_data, val_data, batch_size
        )
        
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, 
                                momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        
        criterion = nn.CrossEntropyLoss()
        
        
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
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        print(f"Model loaded from {filename}")


class CrossValidationTrainer:

    
    def __init__(self, model_class, model_kwargs: Dict[str, Any]):
        """
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
        
        # Create new model for the fold
        model = self.model_class(**self.model_kwargs)
        trainer = SimpleModelTrainer(model, model_name=f"fold_{fold_idx}")
        
        # Train
        history = trainer.train(train_data, val_data, **training_kwargs)
        
        # Evaluate
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
        Args:
            folds_data: List of folds, each containing [features, labels] pairs
            **training_kwargs: Training arguments
            
        Returns:
            Cross-validation results
        """
        n_folds = len(folds_data)
        
        for fold_idx in range(n_folds):
            
            val_data = folds_data[fold_idx]
            train_data = []
            
            for i, fold in enumerate(folds_data):
                if i != fold_idx:
                    train_data.extend(fold)
            
            self.train_fold(train_data, val_data, fold_idx, **training_kwargs)
        
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


class FSCTrainer:
    """
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
        
        self.base_lr = 0.001
        self.final_lr = 0.0001
        self.batch_size = 64
        self.epochs = 50
        self.patience = 10
        
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Trainer initialized for fold {fold_num}")
        print(f"   Base LR: {self.base_lr} -> Final LR: {self.final_lr}")
        print(f"   Batch Size: {self.batch_size}, Epochs: {self.epochs}")
    
    def setup_optimizer(self):
        
        # Calculate decay factor for exponential decay: final_lr = base_lr * (gamma^epochs)
        decay_factor = (self.final_lr / self.base_lr) ** (1.0 / self.epochs)
        
        # Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr)
        
        # Exponential decay scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay_factor)
        
        print(f"Optimizer: Adam (lr={self.base_lr})")
        print(f"Scheduler: ExponentialLR (gamma={decay_factor:.6f})")
        
    def train_fold(self, train_loader, val_loader):
        
        print(f'Training Fold {self.fold_num}...')
        
        # Setup optimizer
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
            
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            self.scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total
            
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
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Early stopping 
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save model
                torch.save(self.model.state_dict(), f'best_model_fold_{self.fold_num}.pth')
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1:2d}/{self.epochs}: Train={train_acc:.2f}% Val={val_acc:.2f}%')
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1} (patience={self.patience})')
                break
        
        print(f'Fold {self.fold_num} completed: Best Val Acc = {best_val_acc:.2f}%')
        
        return {
            'best_val_acc': best_val_acc,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_epoch': epoch + 1
        }


class FSCCrossValidator:
    
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
        
    def run_kfold_training(self, features, labels, n_splits=5): # full data is given to the function, less work in the data preparation
        """
        Args:
            features: Input features
            labels: Target labels
            n_splits: Number of folds
            
        Returns:
            Dictionary with cross-validation results
        """
        print("=" * 80)
        print("5-FOLD CROSS VALIDATION")
        print(f"Dataset: {len(features)} samples, {len(np.unique(labels))} classes")
        print(f"Folds: {n_splits}")
        print("=" * 80)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        fold_results = []
        
        for fold_num, (train_idx, val_idx) in enumerate(skf.split(features, labels), 1):
            print(f"\nProcessing Fold {fold_num}/{n_splits}")
            
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            print(f"   Train: {len(X_train)} samples, Val: {len(X_val)} samples")
            
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
                batch_size=64,
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
            
            model = self.model_creator_func()
            
            trainer = FSCTrainer(model, self.device, fold_num)
            fold_result = trainer.train_fold(train_loader, val_loader)
            
            fold_results.append({
                'fold': fold_num,
                **fold_result
            })
        
        val_accuracies = [result['best_val_acc'] for result in fold_results]
        mean_acc = np.mean(val_accuracies)
        std_acc = np.std(val_accuracies)
        
        print("\n" + "=" * 80)
        print(f"Results Summary:")
        for i, acc in enumerate(val_accuracies, 1):
            print(f"   Fold {i}: {acc:.2f}%")
        print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print("=" * 80)
        
    def run_kfold_training_with_folds(self, folds_data, n_splits=5):
        """
        Run k-fold cross-validation with pre-seperated data
        
        Args:
            folds_data: Dict containing fold data {fold_num: [(feature, label), ...]}
            n_splits: Number of folds (should match number of folds in data)
        """
        print(f"\n{'='*80}")
        
        fold_results = []
        available_folds = list(folds_data.keys())
        
        for fold_num in range(1, n_splits + 1):
            print(f"\nProcessing Fold {fold_num}/{n_splits}")
            
            # Use current fold as validation, others as training
            val_fold = fold_num
            train_folds = [f for f in available_folds if f != val_fold]
            
            # Prepare training data from multiple folds
            train_features = []
            train_labels = []
            
            for train_fold in train_folds:
                if train_fold in folds_data:
                    fold_data = folds_data[train_fold]
                    for item in fold_data:
                        if len(item) == 2:
                            train_features.append(item[0])
                            train_labels.append(item[1])
            
            # Prepare validation data from single fold
            val_features = []
            val_labels = []
            
            if val_fold in folds_data:
                fold_data = folds_data[val_fold]
                for item in fold_data:
                    if len(item) == 2:
                        val_features.append(item[0])
                        val_labels.append(item[1])
            
            # Convert to numpy arrays
            train_features = np.array(train_features)
            train_labels = np.array(train_labels)
            val_features = np.array(val_features)
            val_labels = np.array(val_labels)
            
            # Check if we need to transpose for CNN models
            # Create a sample model to check the type
            sample_model = self.model_creator_func()
            model_name = type(sample_model).__name__.lower()
            is_kan_model = 'kan' in model_name
            del sample_model  # Clean up
            
            if len(train_features.shape) == 4 and train_features.shape[-1] == 3 and not is_kan_model:
                # CNN models expect (batch, channels, height, width) format
                train_features = np.transpose(train_features, (0, 3, 1, 2))  # (B,H,W,C) -> (B,C,H,W)
                val_features = np.transpose(val_features, (0, 3, 1, 2))
            
            print(f"   Train: {len(train_features)} samples, Val: {len(val_features)} samples")
            print(f"   Training folds: {train_folds}, Validation fold: {val_fold}")
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(val_features, dtype=torch.float32),
                torch.tensor(val_labels, dtype=torch.long)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=64,
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
            
            # Create fresh model
            model = self.model_creator_func()
            
            # Train the fold
            trainer = FSCTrainer(model, self.device, fold_num)
            fold_result = trainer.train_fold(train_loader, val_loader)
            
            fold_results.append({
                'fold': fold_num,
                'train_folds': train_folds,
                'val_fold': val_fold,
                **fold_result
            })
        
        # Calculate final results
        val_accuracies = [result['best_val_acc'] for result in fold_results]
        mean_acc = np.mean(val_accuracies)
        std_acc = np.std(val_accuracies)
        
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION COMPLETED - NO DATA LEAKAGE!")
        print(f"Results Summary:")
        for i, (acc, result) in enumerate(zip(val_accuracies, fold_results), 1):
            print(f"   Fold {i}: {acc:.2f}% (Val fold: {result['val_fold']})")
        print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print("=" * 80)
        
        return {
            'fold_results': fold_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'individual_accuracies': val_accuracies,
            'best_fold_acc': max(val_accuracies),
            'worst_fold_acc': min(val_accuracies)
        }
