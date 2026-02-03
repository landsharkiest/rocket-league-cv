# Technical Documentation

## System Architecture

### Overview

The Rocket League CV system is built on a modular architecture with three main components:

1. **Detection Module** (`tracker.py`)
   - Uses YOLOv8 for object detection
   - Implements ByteTrack for multi-object tracking
   - Handles speed calculations and trail visualization

2. **Utility Module** (`utils.py`)
   - Configuration management
   - Data serialization/deserialization
   - Statistical analysis functions

3. **Training Module** (`train.py`)
   - Model training pipeline
   - Validation and evaluation
   - Model export for deployment

### Data Flow

```
Input Video
    ↓
Frame Extraction (OpenCV)
    ↓
Object Detection (YOLOv8)
    ↓
Multi-Object Tracking (ByteTrack)
    ↓
Speed Calculation
    ↓
Visualization & Annotation
    ↓
Output Video + JSON Data
```

## Model Details

### YOLOv8 Architecture

YOLOv8 is a state-of-the-art object detection model that offers:
- Real-time inference speed
- High accuracy on small objects
- Multiple size variants (n, s, m, l, x)
- Easy fine-tuning on custom datasets

### Model Selection Guide

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLOv8n | Fastest | Good | Real-time processing, quick prototyping |
| YOLOv8s | Fast | Better | Balanced performance |
| YOLOv8m | Medium | High | Production use |
| YOLOv8l | Slow | Higher | High accuracy needed |
| YOLOv8x | Slowest | Highest | Research, offline processing |

### ByteTrack Algorithm

ByteTrack is a multi-object tracking algorithm that:
- Associates detections across frames
- Maintains consistent track IDs
- Handles occlusions and re-identification
- Works well with YOLOv8 detections

## Speed Calculation

### Formula

```
speed = (distance_pixels / pixels_per_meter) * fps
```

Where:
- `distance_pixels`: Euclidean distance between consecutive positions
- `pixels_per_meter`: Calibration factor (pixels per meter)
- `fps`: Frames per second of the video

### Calibration Process

1. Identify a known distance in the footage (e.g., Rocket League field width)
2. Measure the corresponding pixel distance
3. Calculate: `pixels_per_meter = pixel_distance / real_distance`
4. Update in `config.yaml`

#### Rocket League Field Dimensions

For reference, standard Rocket League field dimensions:
- Field length: 10240 units (~102.4m)
- Field width: 8192 units (~81.9m)
- Goal width: 1786 units (~17.9m)
- Goal height: 642 units (~6.4m)

Note: These are game units and need conversion based on camera perspective and video resolution.

### Speed Smoothing

Speed values are smoothed using a moving average window to reduce noise:

```python
smoothed_speed = mean(last_N_speeds)
```

Where N is the `smoothing_window` parameter in config.

## Dataset Preparation

### Annotation Format

The system uses YOLO format annotations:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1] relative to image dimensions.

### Example Annotation

For a car at position (320, 240) with size (80, 60) in a 640x480 image:

```
0 0.5 0.5 0.125 0.125
```

### Dataset Structure

```
dataset/
├── images/
│   ├── train/          # 70% of data
│   │   ├── frame_0001.jpg
│   │   └── ...
│   ├── val/            # 20% of data
│   │   └── ...
│   └── test/           # 10% of data
│       └── ...
└── labels/
    ├── train/
    │   ├── frame_0001.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

### Recommended Annotation Tools

1. **LabelImg** - Simple and straightforward
2. **CVAT** - Powerful, supports video annotation
3. **Roboflow** - Cloud-based with auto-labeling features
4. **Label Studio** - Open-source, versatile

## Training Best Practices

### Data Collection

- Collect diverse gameplay footage:
  - Different camera angles
  - Different arenas
  - Different car types and colors
  - Various lighting conditions
  - Multiple boost trail effects

### Data Augmentation

YOLOv8 automatically applies augmentation:
- Random crops
- Color adjustments
- Flips and rotations
- Mosaic augmentation

### Training Parameters

Recommended starting parameters:

```python
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50,  # Early stopping
    device=0,     # GPU 0
)
```

Adjust based on:
- **Dataset size**: Larger datasets may need more epochs
- **GPU memory**: Reduce batch size if out of memory
- **Convergence**: Increase patience if model hasn't converged

### Evaluation Metrics

- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: mAP averaged over IoU 0.5 to 0.95
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)

Target metrics:
- mAP50 > 0.8 for production use
- mAP50-95 > 0.5 for robust detection

## Performance Optimization

### Hardware Recommendations

- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: SSD for faster data loading

### Software Optimization

1. **Use smaller model variants** for real-time processing
2. **Enable GPU acceleration** (CUDA)
3. **Reduce image size** if detection quality allows
4. **Use FP16 inference** for faster processing:
   ```python
   model = YOLO('model.pt', task='detect')
   model.to('cuda')
   model.half()  # FP16 mode
   ```

### Batch Processing

For offline processing of multiple videos:

```python
video_paths = ['video1.mp4', 'video2.mp4', ...]
for video_path in video_paths:
    detections = tracker.track_video(video_path, f'output_{Path(video_path).stem}.mp4')
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size
   - Use smaller model variant
   - Reduce image size

2. **Poor Detection Quality**
   - Increase confidence threshold
   - Train on more diverse data
   - Use larger model variant
   - Check dataset annotation quality

3. **Inconsistent Tracking**
   - Adjust IoU threshold
   - Increase track buffer
   - Check for rapid camera movements

4. **Incorrect Speed Calculations**
   - Verify calibration factor
   - Check FPS value
   - Ensure consistent frame rate

### Debug Mode

Add verbose output for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **3D Position Tracking**
   - Estimate 3D coordinates from 2D detections
   - More accurate speed calculations
   - Aerial maneuver detection

2. **Boost Detection**
   - Track boost pad pickups
   - Monitor boost meter levels
   - Boost usage statistics

3. **Event Detection**
   - Goals
   - Demos
   - Saves
   - Aerial hits

4. **Player Identification**
   - OCR for player names
   - Car color/design recognition
   - Team classification

5. **Real-time Processing**
   - Live stream support
   - WebSocket API
   - Dashboard interface

### Research Directions

- **Temporal modeling**: Use LSTM/Transformer for trajectory prediction
- **Action recognition**: Classify specific maneuvers (flip, air roll, etc.)
- **Game state estimation**: Extract score, time, boost levels
- **Multi-camera tracking**: Fusion of multiple camera views

## API Reference

### RocketLeagueTracker

#### Methods

- `__init__(config)`: Initialize tracker
- `load_model(model_path)`: Load YOLO model
- `track_frame(frame, fps)`: Process single frame
- `track_video(video_path, output_path)`: Process entire video
- `calculate_speed(track_id, position, fps)`: Calculate object speed

#### Configuration Parameters

See `config.yaml` for full parameter list.

### Utility Functions

- `load_config(path)`: Load YAML configuration
- `save_detections(detections, path)`: Save to JSON
- `calculate_average_speed(detections, track_id)`: Compute average speed
- `get_max_speed(detections, track_id)`: Get maximum speed
- `create_output_directory(path)`: Create output directory

## Contributing

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Document functions with docstrings
- Keep functions focused and modular

### Testing

Before submitting changes:

1. Test on sample videos
2. Verify speed calculations
3. Check output video quality
4. Validate JSON output format

### Pull Request Guidelines

- Clear description of changes
- Include example results
- Update documentation
- Add tests if applicable

## License

MIT License - See LICENSE file for details.

## Contact & Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/landsharkiest/rocket-league-cv/issues
- Pull Requests: https://github.com/landsharkiest/rocket-league-cv/pulls

## References

1. YOLOv8: https://github.com/ultralytics/ultralytics
2. ByteTrack: https://github.com/ifzhang/ByteTrack
3. Rocket League: https://www.rocketleague.com/
4. Object Detection Guide: https://docs.ultralytics.com/
