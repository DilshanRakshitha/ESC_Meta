#!/usr/bin/env python3
"""
ESC Meta - KAN Training Demo
Optimized training configuration for KAN models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_improved_kan_training(model_name='kan'):
    """Run improved KAN training with optimized configuration"""
    print(f"🚀 IMPROVED KAN TRAINING FOR {model_name.upper()}")
    print("=" * 50)
    
    # Import based on model type
    if model_name.lower() == 'kan':
        from models.architectures.kan_models import create_high_performance_kan
        model = create_high_performance_kan((128, 196, 3), 8)
    elif model_name.lower() == 'ickan':
        from models.architectures.ickan_models import create_high_performance_ickan
        model = create_high_performance_ickan((128, 196, 3), 8)
    elif model_name.lower() == 'wavkan':
        from models.architectures.wavkan_models import create_high_performance_wavkan
        model = create_high_performance_wavkan((128, 196, 3), 8)
    else:
        raise ValueError(f"Unknown KAN model: {model_name}")
    
    # KAN-optimized configuration
    batch_size = 16
    learning_rate = 0.0001
    epochs = 15
    num_classes = 8
    
    print(f"📊 Configuration:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Classes: {num_classes}")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create structured data (multiple batches for better training)
    all_features = []
    all_labels = []
    
    for batch_idx in range(5):  # Create 5 different batches
        features = torch.randn(batch_size, 3, 128, 196)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        # Create pattern-based labels (more structured)
        for i in range(batch_size):
            # Multiple patterns for better learning
            pattern1 = features[i, 0].mean().item()
            pattern2 = features[i, 1].std().item()
            pattern3 = features[i, 2].max().item()
            
            combined_pattern = (abs(pattern1) + abs(pattern2) + abs(pattern3)) / 3
            labels[i] = int(combined_pattern * num_classes) % num_classes
        
        all_features.append(features)
        all_labels.append(labels)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.7)
    
    # Training loop
    model.train()
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Train on all batches
        for batch_idx, (features, labels) in enumerate(zip(all_features, all_labels)):
            optimizer.zero_grad()
            
            # KAN expects (batch, height, width, channels)
            model_input = features.permute(0, 2, 3, 1)
            
            outputs = model(model_input)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(all_features)
        accuracy = total_correct / total_samples
        best_accuracy = max(best_accuracy, accuracy)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f} ({accuracy*100:.1f}%)")
        
        scheduler.step(avg_loss)
        
        # Early stopping if we reach good accuracy
        if accuracy > 0.8:
            print("🎯 Excellent accuracy reached! Early stopping.")
            break
    
    print()
    print(f"✅ Training completed for {model_name}")
    print(f"🎯 Best accuracy achieved: {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        eval_correct = 0
        eval_total = 0
        
        for features, labels in zip(all_features, all_labels):
            model_input = features.permute(0, 2, 3, 1)
            outputs = model(model_input)
            _, predicted = torch.max(outputs.data, 1)
            eval_correct += (predicted == labels).sum().item()
            eval_total += labels.size(0)
        
        eval_accuracy = eval_correct / eval_total
        print(f"🔍 Final evaluation accuracy: {eval_accuracy:.4f} ({eval_accuracy*100:.1f}%)")
    
    return best_accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Improved KAN Training')
    parser.add_argument('--model', default='kan', choices=['kan', 'ickan', 'wavkan'], 
                        help='KAN model type to train')
    
    args = parser.parse_args()
    
    try:
        accuracy = run_improved_kan_training(args.model)
        print(f"\n🎉 SUCCESS: {args.model.upper()} achieved {accuracy*100:.1f}% accuracy!")
    except Exception as e:
        print(f"❌ Error: {e}")
