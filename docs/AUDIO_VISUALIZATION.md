# Audio Signal Visualization

This feature allows you to visualize audio signals at different processing stages in the feature generation pipeline.

## Overview

The audio visualization feature generates waveform plots showing how audio signals are transformed through the preprocessing and augmentation pipeline. This helps in understanding and debugging the audio processing workflow.

## Generated Visualizations

The system can generate visualizations at three key stages:

1. **Before Preprocessing (Raw Audio)**: The original audio signal as loaded from the file
2. **After Preprocessing**: Audio after normalization (padding/cropping to fixed length)
3. **After Augmentation**: Audio after applying augmentation techniques (pitch shift, time stretch, noise, etc.)

## Usage

### Using the Main Feature Generation Script

Enable visualization by adding the `--visualize` flag when running `generate_features.py`:

```bash
python generate_features.py \
  --csv-path data/FSC22.csv \
  --audio-path data/wav44 \
  --feature-type MEL \
  --augmentation 3 \
  --visualize \
  --viz-output audio_visualizations \
  --viz-samples 5
```

#### Command-line Arguments

- `--visualize` or `-v`: Enable audio waveform visualization
- `--viz-output`: Output directory for visualization images (default: `audio_visualizations`)
- `--viz-samples`: Number of audio samples to visualize (default: 3)

### Using the FSCFeatureGenerator Class

```python
from feature_generator.fsc_feature_generator import FSCFeatureGenerator

# Initialize with visualization enabled
generator = FSCFeatureGenerator(
    sr=20000,
    duration=5,
    enable_visualization=True,
    visualization_output_dir='my_visualizations'
)

# Load audio files
audios = [
    (audio_signal_1, label_1),
    (audio_signal_2, label_2),
    # ... more audio samples
]

# Extract features with visualization
features = generator.extract_features_with_augmentation(
    audios,
    feature_type='MEL',
    augment_level=3,
    max_visualizations=5  # Visualize first 5 samples
)
```

### Using the AudioVisualizer Directly

```python
import numpy as np
from utils.audio_visualizer import AudioVisualizer

# Initialize visualizer
visualizer = AudioVisualizer(
    sample_rate=20000,
    output_dir='my_visualizations'
)

# Visualize a single waveform
path = visualizer.plot_waveform(
    audio=audio_signal,
    title='Audio Signal',
    filename='my_audio',
    stage='Before Preprocessing',
    label=5
)

# Visualize all processing stages at once
paths = visualizer.visualize_processing_stages(
    raw_audio=raw_audio,
    preprocessed_audio=preprocessed_audio,
    augmented_audio=augmented_audio,
    filename='sample_001',
    label=5,
    augmentation_type='pitch_up'
)
```

## Output Files

Visualization images are saved as PNG files with descriptive filenames:

- **Format**: `sample_{index}_class_{label}_{stage}_{augmentation}.png`
- **Examples**:
  - `sample_0_class_5_1_before_preprocessing.png`
  - `sample_0_class_5_2_after_preprocessing.png`
  - `sample_0_class_5_3_after_augmentation_pitch_up.png`

## Visualization Stages Explained

### 1. Before Preprocessing (Raw Audio)
- Shows the original audio signal as loaded from the file
- Duration may vary (typically ≤5 seconds)
- No padding or cropping applied

### 2. After Preprocessing
- Audio normalized to exactly 5 seconds (100,000 samples at 20kHz)
- Shorter audio is padded with zeros
- Longer audio is randomly cropped

### 3. After Augmentation
- Shows the effect of various augmentation techniques:
  - **speed_up**: Audio compressed in time (1.5x rate) with padding
  - **slow_down**: Audio stretched in time (0.667x rate) with cropping
  - **pitch_up**: Pitch shifted up by 2 semitones
  - **pitch_down**: Pitch shifted down by 2 semitones
  - **noise**: Gaussian noise added
  - **reverse_noise**: Audio reversed and noise added

## Augmentation Levels

Different augmentation levels generate different visualizations:

- **Level 0**: No augmentation (only original)
- **Level 1**: Time stretch only (speed_up, slow_down)
- **Level 2**: Pitch shift only (pitch_up, pitch_down)
- **Level 3**: Time stretch + pitch shift (4 augmentations)
- **Level 4**: All augmentations including noise (6+ augmentations)

## Performance Considerations

- Visualization adds minimal overhead to the feature generation process
- Only the specified number of samples (default: 3) are visualized
- Visualization can be disabled entirely by omitting the `--visualize` flag
- Image files are typically 40-50 KB per waveform

## Examples

### Example 1: Visualize First 3 Samples with MEL Features

```bash
python generate_features.py \
  --csv-path data/FSC22.csv \
  --audio-path data/wav44 \
  --feature-type MEL \
  --augmentation 3 \
  --visualize \
  --viz-samples 3
```

### Example 2: Visualize 10 Samples with Custom Output Directory

```bash
python generate_features.py \
  --csv-path data/FSC22.csv \
  --audio-path data/wav44 \
  --feature-type MFCC \
  --augmentation 4 \
  --visualize \
  --viz-output my_audio_plots \
  --viz-samples 10
```

### Example 3: Using Preset with Visualization

```bash
python generate_features.py \
  --preset aug_ts_ps_mel_features_5_20 \
  --visualize
```

## Troubleshooting

### No Visualizations Generated

- Ensure `--visualize` flag is present
- Check that the output directory is writable
- Verify that matplotlib is installed: `pip install matplotlib`

### Missing Augmentation Visualizations

- Augmentation visualizations are only created for the first sample (index 0)
- Increase `--viz-samples` if you want more samples visualized

### File Path Issues

- Use absolute paths or paths relative to the project root
- Ensure the output directory exists or can be created

## Technical Details

- **Image Format**: PNG with 150 DPI
- **Image Size**: 12" x 4" (width x height)
- **Plot Style**: Time-domain waveform with amplitude on Y-axis
- **Time Resolution**: Full sample rate (20,000 samples/second)
- **Dependencies**: matplotlib, numpy

## See Also

- `utils/audio_visualizer.py` - Core visualization implementation
- `feature_generator/fsc_feature_generator.py` - Feature generation with visualization
- `generate_features.py` - Main feature generation script
- `test_visualization.py` - Test script demonstrating usage
