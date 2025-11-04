#!/usr/bin/env python3
"""Test the new src/main.py"""

import sys
import os
sys.path.insert(0, '/home/dilshan/Documents/ESC/temp1/ESC_Meta')

# Import and test the PyTorch main
import importlib.util
spec = importlib.util.spec_from_file_location("pytorch_main", "/home/dilshan/Documents/ESC/temp1/ESC_Meta/src/main.py")
pytorch_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pytorch_main)

print("✅ PyTorch main.py loaded successfully")

# Test the factory
factory = pytorch_main.PyTorchModelFactory()
available = factory.get_available_models()
print(f"📋 Available models: {available}")

# Test model creation
if 'alexnet' in available:
    model = factory.create_model('alexnet', 26)
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ AlexNet created: {params:,} parameters")

print("🎯 New PyTorch main.py is working correctly!")
