# Rocket League CV

A computer vision project for tracking Rocket League cars and ball in gameplay footage. This project uses YOLOv8 for object detection and ByteTrack for multi-object tracking to monitor car speeds, ball speeds, and other gameplay metrics.

## Features

- 🚗 **Car Detection & Tracking**: Detect and track multiple cars throughout gameplay
- ⚽ **Ball Tracking**: Track the ball's position and movement
- 📊 **Speed Calculation**: Calculate and display real-time speeds for tracked objects
- 🎨 **Visual Trails**: Draw movement trails to visualize object paths
- 💾 **Data Export**: Save tracking data to JSON for further analysis
- 🎥 **Video Output**: Generate annotated videos with tracking overlays

## Architecture

The project is built using:
- **YOLOv8**: State-of-the-art object detection model from Ultralytics
- **ByteTrack**: Robust multi-object tracking algorithm
- **OpenCV**: Video processing and visualization
- **NumPy**: Numerical computations for speed calculations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/landsharkiest/rocket-league-cv.git
cd rocket-league-cv
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Track objects in a Rocket League video:

```bash
python main.py path/to/your/video.mp4
```

This will:
- Process the video and detect/track objects
- Save an annotated video to `output/tracked_<video_name>.mp4`
- Display tracking statistics (average speeds, max speeds, etc.)

### Advanced Options

```bash
# Use a custom configuration file
python main.py video.mp4 --config custom_config.yaml

# Use a custom trained model
python main.py video.mp4 --model path/to/custom_model.pt

# Specify output path
python main.py video.mp4 --output results/my_tracked_video.mp4

# Don't save output video (faster processing)
python main.py video.mp4 --no-save

# Save detection data to JSON
python main.py video.mp4 --save-detections
```

### Configuration

Customize tracking behavior by editing `config.yaml`:

```yaml
# Model settings
model:
  type: "yolov8"
  size: "n"  # n, s, m, l, or x (larger = more accurate but slower)
  confidence_threshold: 0.5
  iou_threshold: 0.45

# Speed calculation
speed:
  fps: 30
  pixels_per_meter: 10  # Calibration factor (adjust based on your footage)
  smoothing_window: 5

# Output settings
output:
  save_video: true
  draw_trails: true
  show_speed: true
```

## Training a Custom Model

The default YOLOv8 model is pretrained on general objects and will need fine-tuning for Rocket League-specific detection. To train a custom model:

1. **Collect Training Data**: Record Rocket League gameplay and extract frames

2. **Annotate Data**: Label cars, ball, and boost pads using tools like:
   - [LabelImg](https://github.com/heartexlabs/labelImg)
   - [CVAT](https://github.com/opencv/cvat)
   - [Roboflow](https://roboflow.com/)

3. **Organize Dataset**:
```
dataset/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/
```

4. **Train Model**:
```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Train on your custom dataset
model.train(
    data='dataset.yaml',  # Path to dataset config
    epochs=100,
    imgsz=640,
    batch=16
)
```

5. **Use Trained Model**:
```bash
python main.py video.mp4 --model runs/detect/train/weights/best.pt
```

## Project Structure

```
rocket-league-cv/
├── rocket_league_cv/          # Main package
│   ├── __init__.py
│   ├── tracker.py             # Core tracking logic
│   └── utils.py               # Utility functions
├── main.py                     # Main script
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Speed Calibration

Speed calculations depend on the `pixels_per_meter` calibration factor in `config.yaml`. To calibrate:

1. Find a known distance in your footage (e.g., Rocket League field is 8192 units = ~82 meters)
2. Measure the pixel distance in your video
3. Calculate: `pixels_per_meter = pixel_distance / real_distance_meters`
4. Update `config.yaml` with this value

## Limitations & Future Work

- **Training Required**: Default model needs fine-tuning on Rocket League footage for accurate detection
- **Calibration**: Speed calculations require proper calibration for each video resolution/perspective
- **3D Tracking**: Current implementation uses 2D tracking; 3D position tracking could improve accuracy
- **Additional Metrics**: Could track boost usage, demo events, aerial maneuvers, etc.

## Contributing

Contributions are welcome! Areas for improvement:
- Training and sharing custom Rocket League detection models
- Improving speed calibration algorithms
- Adding new tracking metrics (boost, demos, goals, etc.)
- Performance optimizations
- Better visualization options

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking algorithm
- Rocket League by Psyonix
