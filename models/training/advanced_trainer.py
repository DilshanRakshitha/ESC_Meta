"""
Advanced Training Module for Audio Classification Models
Integrated with configuration system for modular training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, List
from config.config import TrainingConfig

class AdvancedTrainer:
    """Advanced trainer class with comprehensive training functionality"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_accuracy = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        print(f"Trainer initialized on device: {self.device}")
    
    def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Setup optimizer based on config"""
        if self.config.optimizer == 'adam':
            return optim.Adam(model.parameters(), 
                            lr=self.config.learning_rate,
                            weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(model.parameters(),
                             lr=self.config.learning_rate,
                             weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            return optim.SGD(model.parameters(),
                           lr=self.config.learning_rate,
                           momentum=0.9,
                           weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def setup_scheduler(self, optimizer: torch.optim.Optimizer, 
                       steps_per_epoch: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if self.config.scheduler == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3
            )
        elif self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        else:
            return None
    
    def setup_criterion(self, num_classes: int) -> nn.Module:
        """Setup loss criterion"""
        if self.config.loss_function == 'crossentropy':
            if self.config.label_smoothing > 0:
                return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            else:
                return nn.CrossEntropyLoss()
        elif self.config.loss_function == 'focal':
            return FocalLoss(alpha=1, gamma=2, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
    
    def train_epoch(self, model: nn.Module, 
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping if enabled
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
            
            optimizer.step()
            
            if scheduler is not None and self.config.scheduler == 'onecycle':
                scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        if scheduler is not None and self.config.scheduler != 'onecycle':
            scheduler.step()
        
        return total_loss / len(train_loader), 100.0 * correct / total
    
    def validate(self, model: nn.Module, 
                val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Validate the model"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        return total_loss / len(val_loader), accuracy * 100, np.array(all_preds), np.array(all_targets)
    
    def train(self, model: nn.Module,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_classes: int) -> Dict[str, Any]:
        """Complete training loop"""
        
        model.to(self.device)
        
        # Setup training components
        optimizer = self.setup_optimizer(model)
        scheduler = self.setup_scheduler(optimizer, len(train_loader))
        criterion = self.setup_criterion(num_classes)
        
        print(f"Training Configuration:")
        print(f"- Device: {self.device}")
        print(f"- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"- Optimizer: {self.config.optimizer}")
        print(f"- Learning rate: {self.config.learning_rate}")
        print(f"- Batch size: {self.config.batch_size}")
        print(f"- Epochs: {self.config.epochs}")
        print(f"- Scheduler: {self.config.scheduler}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler
            )
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(
                model, val_loader, criterion
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                if self.config.save_best_model:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_accuracy': self.best_accuracy,
                        'config': self.config
                    }, 'best_model.pth')
            
            # Print progress
            if epoch % self.config.print_interval == 0:
                print(f'Epoch [{epoch+1}/{self.config.epochs}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'Best Acc: {self.best_accuracy:.2f}%')
                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f'Learning Rate: {current_lr:.6f}')
                print('-' * 50)
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_report = classification_report(
            val_targets, val_preds, 
            target_names=[f'Class_{i}' for i in range(num_classes)],
            output_dict=True
        )
        
        results = {
            'best_accuracy': self.best_accuracy,
            'final_accuracy': val_acc,
            'training_time': training_time,
            'history': self.history,
            'classification_report': final_report,
            'confusion_matrix': confusion_matrix(val_targets, val_preds).tolist()
        }
        
        print(f"\n🎉 Training completed!")
        print(f"⏱️  Total time: {training_time:.2f} seconds")
        print(f"🎯 Best validation accuracy: {self.best_accuracy:.2f}%")
        print(f"📊 Final validation accuracy: {val_acc:.2f}%")
        
        return results

class KFoldTrainer:
    """K-Fold Cross-Validation Trainer"""
    
    def __init__(self, config: TrainingConfig, k_folds: int = 5):
        self.config = config
        self.k_folds = k_folds
        self.fold_results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_fold(self, model_class, model_kwargs: Dict[str, Any],
                  train_loader: DataLoader, val_loader: DataLoader,
                  fold: int, num_classes: int) -> Dict[str, Any]:
        """Train a single fold"""
        
        print(f"\n🔄 Training Fold {fold + 1}/{self.k_folds}")
        print("=" * 60)
        
        # Create fresh model instance
        model = model_class(**model_kwargs)
        
        # Use AdvancedTrainer for individual fold
        fold_trainer = AdvancedTrainer(self.config)
        fold_results = fold_trainer.train(model, train_loader, val_loader, num_classes)
        
        fold_results['fold'] = fold
        self.fold_results.append(fold_results)
        
        return fold_results
    
    def get_cv_summary(self) -> Dict[str, Any]:
        """Get cross-validation summary"""
        accuracies = [result['best_accuracy'] for result in self.fold_results]
        
        summary = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'fold_accuracies': accuracies,
            'fold_results': self.fold_results
        }
        
        print(f"\n📋 Cross-Validation Summary:")
        print(f"📊 Mean Accuracy: {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}%")
        print(f"📈 Range: [{summary['min_accuracy']:.2f}%, {summary['max_accuracy']:.2f}%]")
        print(f"🎯 Individual fold accuracies:")
        for i, acc in enumerate(accuracies):
            print(f"   Fold {i+1}: {acc:.2f}%")
        
        return summary

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, num_classes: int = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def create_trainer(config: TrainingConfig, k_folds: Optional[int] = None):
    """Factory function to create trainer based on config"""
    
    if k_folds is not None and k_folds > 1:
        return KFoldTrainer(config, k_folds)
    else:
        return AdvancedTrainer(config)
