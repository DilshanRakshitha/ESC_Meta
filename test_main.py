#!/usr/bin/env python3
"""Test main file to verify functionality"""

import torch
import torch.nn as nn

print("🎵 FSC META - TEST VERSION")

def create_simple_cnn(input_channels=3, num_classes=26):
    """Create a simple CNN that always works"""
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 16, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )

def test_simple():
    """Test simple functionality"""
    print("Testing simple CNN creation...")
    model = create_simple_cnn()
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Simple CNN created: {params:,} parameters")
    return model

if __name__ == "__main__":
    print("Running test...")
    test_simple()
    print("Test complete!")
