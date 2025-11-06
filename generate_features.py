#!/usr/bin/env python3
"""
Corrected FSC22 Feature Generation Tool
Based on the working logic from Forest-Sound-Analysis-on-Edge-main
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import pickle
import random
import audiomentations as aa
import argparse
import yaml
from pathlib import Path

# Fixed parameters matching the working implementation  
SAMPLE_RATE = 20000
INPUT_LENGTH = SAMPLE_RATE * 5  # 5 seconds at 20kHz to match Forest-Sound-Analysis (128, 196, 3)

def random_crop(sound, size):
    """Original working function"""
    org_size = len(sound)
    if org_size <= size:
        return sound
    start = random.randint(0, org_size - size)
    return sound[start: start + size]

def padding(sound, size):
    """Original working function - exactly as in working repo"""
    diff = size - len(sound)
    return np.pad(sound, (diff//2, diff-(diff//2)), 'constant')

# Initialize augmentation tool
gaussian_noise = aa.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)

def augmentor(audio, augmentation):
    """Original working augmentation function"""
    if augmentation == 1:
        return audio
    elif augmentation == 2:
        spedup_sound = librosa.effects.time_stretch(y=audio, rate=1.5)
        return padding(spedup_sound, INPUT_LENGTH)
    elif augmentation == 3:
        slowed_sound = librosa.effects.time_stretch(y=audio, rate=0.667)
        return random_crop(slowed_sound, INPUT_LENGTH)
    elif augmentation == 4:
        return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=2)
    elif augmentation == 5:
        return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=-2)
    elif augmentation == 6:
        reversed_audio = audio[::-1]
        return gaussian_noise(reversed_audio, sample_rate=SAMPLE_RATE)

def mel_features_extractor(raw_audio):
    """Original working MEL extractor"""
    feature_1 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512))
    feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512))
    feature_3 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=SAMPLE_RATE, n_mels=128, n_fft=512, hop_length=512))
    
    three_channel = np.stack((feature_1, feature_2, feature_3), axis=2)
    return three_channel

def mfcc_features_extractor(raw_audio):
    """Original working MFCC extractor"""
    feature_1 = librosa.feature.mfcc(y=raw_audio, n_mfcc=128, n_fft=2048, hop_length=512)
    feature_2 = librosa.feature.mfcc(y=raw_audio, n_mfcc=128, n_fft=1024, hop_length=512)
    feature_3 = librosa.feature.mfcc(y=raw_audio, n_mfcc=128, n_fft=512, hop_length=512)
    
    three_channel = np.stack((feature_1, feature_2, feature_3), axis=2)
    return three_channel

def mixed_features_extractor(raw_audio):
    """Original working mixed extractor"""
    feature_1 = librosa.feature.mfcc(y=raw_audio, n_mfcc=128, n_fft=1024, hop_length=512)
    feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512))
    feature_3 = librosa.feature.chroma_stft(y=raw_audio, sr=SAMPLE_RATE, n_chroma=128, n_fft=1024, hop_length=512)
    
    three_channel = np.stack((feature_1, feature_2, feature_3), axis=2)
    return three_channel

def load_audio_files(csv_path, audio_base_path):
    """Load audio files from CSV - MEMORY OPTIMIZED"""
    df = pd.read_csv(csv_path)
    audios = []
    
    print(f"📂 Loading audio files from {csv_path}...")
    
    # Process in smaller batches to avoid memory issues
    batch_size = 50
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        print(f"   Processing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})...")
        
        for i, row in batch_df.iterrows():
            audio_path = os.path.join(audio_base_path, row['filename'])
            label = row['target'] - 1  # Convert from 1-26 to 0-25 range (same as Forest-Sound-Analysis)
            
            if os.path.exists(audio_path):
                try:
                    # Load shorter duration to reduce memory usage
                    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=5.0)
                    
                    # Ensure exact length
                    if len(audio) < INPUT_LENGTH:
                        audio = padding(audio, INPUT_LENGTH)
                    elif len(audio) > INPUT_LENGTH:
                        audio = random_crop(audio, INPUT_LENGTH)
                    
                    audios.append([audio, label])
                    
                except Exception as e:
                    print(f"   Warning: Failed to load {audio_path}: {e}")
                    continue
            else:
                print(f"   Warning: File not found: {audio_path}")
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
    
    print(f"✅ Loaded {len(audios)} audio files successfully")
    return audios

def generate_features(audios, feature_type='MEL', augment_level=3):
    """Generate features with augmentation - MEMORY OPTIMIZED"""
    
    print(f"🔧 Extracting {feature_type} features with augmentation level {augment_level}...")
    
    # Choose feature extractor
    if feature_type == 'MEL':
        extractor = mel_features_extractor
    elif feature_type == 'MFCC':
        extractor = mfcc_features_extractor
    elif feature_type == 'MIX':
        extractor = mixed_features_extractor
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    spects = []
    
    # Process in smaller batches to avoid memory issues
    batch_size = 25
    total_batches = len(audios) // batch_size + (1 if len(audios) % batch_size > 0 else 0)
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(audios))
        batch_audios = audios[start_idx:end_idx]
        
        print(f"   Processing feature batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})...")
        
        for i, element in enumerate(batch_audios):
            audio, label = element[0], element[1]
            
            # Original sample
            try:
                original_feature = extractor(audio)
                spects.append([original_feature, label, True])
            except Exception as e:
                print(f"   Warning: Failed to extract original features for sample {start_idx + i}: {e}")
                continue
            
            # Augmented samples - exactly as working implementation
            try:
                if augment_level == 1:  # Time stretch only
                    spects.append([extractor(augmentor(audio, 2)), label, False])  # speed up
                    spects.append([extractor(augmentor(audio, 3)), label, False])  # slow down
                    
                elif augment_level == 2:  # Pitch shift only
                    spects.append([extractor(augmentor(audio, 4)), label, False])  # pitch up
                    spects.append([extractor(augmentor(audio, 5)), label, False])  # pitch down
                    
                elif augment_level == 3:  # Both time stretch and pitch shift
                    spects.append([extractor(augmentor(audio, 2)), label, False])  # speed up
                    spects.append([extractor(augmentor(audio, 3)), label, False])  # slow down
                    spects.append([extractor(augmentor(audio, 4)), label, False])  # pitch up
                    spects.append([extractor(augmentor(audio, 5)), label, False])  # pitch down
                    
                elif augment_level == 4:  # All augmentations
                    spects.append([extractor(augmentor(audio, 2)), label, False])  # speed up
                    spects.append([extractor(augmentor(audio, 3)), label, False])  # slow down
                    spects.append([extractor(augmentor(audio, 4)), label, False])  # pitch up
                    spects.append([extractor(augmentor(audio, 5)), label, False])  # pitch down
                    spects.append([extractor(augmentor(audio, 6)), label, False])  # reverse + noise
                    
            except Exception as e:
                print(f"   Warning: Failed augmentation for sample {start_idx + i}: {e}")
                continue
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
    
    print(f"✅ Generated {len(spects)} feature samples total")
    return spects

def save_features(spects, output_dir, feature_type, augmentation_level):
    """Save features in the required 5-fold structure with proper cross-validation splits"""
    import pickle
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    import gc
    
    print(f"Processing {len(spects)} feature samples...")
    
    # Create DataFrame exactly like Forest-Sound-Analysis
    features_df = pd.DataFrame(spects, columns=['feature', 'class', 'status'])
    
    # Clean up original spects to save memory
    del spects
    gc.collect()
    
    # Extract features and labels exactly like Forest-Sound-Analysis
    X = np.array(features_df['feature'].tolist())
    y = np.array(features_df['class'].tolist())
    
    print(f'X shape: {np.shape(X)}')
    print(f'y shape: {np.shape(y)}')
    print(f'Label range: {min(y)} to {max(y)} (0-25 format)')
    
    # Create main output directory
    # output_dir is already the Pickle_Files directory, so use it directly
    pickle_dir = output_dir
    os.makedirs(pickle_dir, exist_ok=True)
    
    # Create feature type directory (matching expected format)
    feature_dir_name = f"aug_ts_ps_{feature_type.lower()}_features_5_20"
    feature_dir = os.path.join(pickle_dir, feature_dir_name)
    os.makedirs(feature_dir, exist_ok=True)
    
    # Create stratified 5-fold splits exactly like Forest-Sound-Analysis
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    spect_folds = []
    
    print(f"Creating stratified 5-fold splits...")
    
    # Iterate over the folds exactly like Forest-Sound-Analysis
    for fold, (train_index, test_index) in enumerate(stratified_kfold.split(X, y)):
        # Split the data into training and testing sets using list comprehensions
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        
        print(f"Fold {fold + 1}:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Testing bin count: {np.bincount(y_test)}")
        
        # Create test data in expected format exactly like Forest-Sound-Analysis
        test_comp = [list(e) for e in zip(X_test, y_test)]
        
        spect_folds.append(test_comp)
    
    # Verify fold structure exactly like Forest-Sound-Analysis
    print(f'len spect_folds: {len(spect_folds)}')  # num folds
    print(f'len spect_folds[0]: {len(spect_folds[0])}')  # samples in a fold
    print(f'len spect_folds[0][0]: {len(spect_folds[0][0])}')  # elements in a sample - should be 2
    print(f'shape spect_folds[0][0][0]: {np.shape(spect_folds[0][0][0])}')  # spectrogram shape
    
    # Save each fold exactly like Forest-Sound-Analysis
    for fold in range(5):
        fold_filename = f"{feature_dir_name}_fold{fold + 1}"
        fold_path = os.path.join(feature_dir, fold_filename)
        
        with open(fold_path, 'wb') as f:
            pickle.dump(spect_folds[fold], f)
    
    print(f"Features saved to {feature_dir}")
    print(f"Feature shape: {X[0].shape}")
    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {len(set(y))}")
    print("✅ Each fold contains different test samples for proper cross-validation!")
    
    return feature_dir

def main():
    parser = argparse.ArgumentParser(description='Corrected FSC22 Feature Generation')
    
    parser.add_argument('--csv-path', '-c', type=str, 
                       default='/home/dilshan/Documents/ESC/temp1/ESC_Meta/data/fsc22/FSC22.csv',
                       help='Path to FSC22.csv file')
    
    parser.add_argument('--audio-path', '-a', type=str,
                       default='/home/dilshan/Documents/ESC/temp1/ESC_Meta/data/fsc22/wav44',
                       help='Path to audio files directory')
    
    parser.add_argument('--output', '-o', type=str,
                       default='/home/dilshan/Documents/ESC/temp1/ESC_Meta/data/fsc22/Pickle_Files',
                       help='Output directory for features')
    
    parser.add_argument('--feature-type', '-f', type=str, default='MEL',
                       choices=['MEL', 'MFCC', 'MIX'],
                       help='Feature type (MEL, MFCC, MIX)')
    
    parser.add_argument('--augmentation', '-aug', type=int, default=3,
                       choices=[0, 1, 2, 3, 4],
                       help='Augmentation level: 0=none, 1=ts, 2=ps, 3=ts+ps, 4=all')
    
    # Preset for exact requirement
    parser.add_argument('--preset', '-p', type=str,
                       choices=['aug_ts_ps_mel_features_5_20'],
                       help='Use preset configuration')
    
    args = parser.parse_args()
    
    # Handle preset
    if args.preset == 'aug_ts_ps_mel_features_5_20':
        args.feature_type = 'MEL'
        args.augmentation = 3  # time stretch + pitch shift
        print(f"🎯 Using preset: {args.preset}")
    
    print(f"🎵 FSC22 Feature Generation")
    print(f"   Feature type: {args.feature_type}")
    print(f"   Augmentation level: {args.augmentation}")
    print(f"   CSV: {args.csv_path}")
    print(f"   Audio: {args.audio_path}")
    print(f"   Output: {args.output}")
    
    # Validate paths
    if not os.path.exists(args.csv_path):
        print(f"❌ CSV file not found: {args.csv_path}")
        return
    
    if not os.path.exists(args.audio_path):
        print(f"❌ Audio directory not found: {args.audio_path}")
        return
    
    try:
        # Load audio files
        audios = load_audio_files(args.csv_path, args.audio_path)
        
        if not audios:
            print("❌ No audio files loaded!")
            return
        
        # Generate features
        spects = generate_features(audios, args.feature_type, args.augmentation)
        
        if not spects:
            print("❌ No features generated!")
            return
        
        # Save features
        output_path = save_features(spects, args.output, args.feature_type, args.augmentation)
        
        print(f"\n✅ Feature generation completed successfully!")
        print(f"📁 Output: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during feature generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()