"""
Utility functions for the Rocket League CV project
"""

import yaml
import json
from pathlib import Path


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_detections(detections, output_path):
    """
    Save detection results to JSON file.
    
    Args:
        detections (list): List of detection dictionaries for each frame
        output_path (str): Path to save JSON file
    """
    # Convert numpy types to native Python types for JSON serialization
    serializable_detections = []
    for frame_detections in detections:
        frame_data = {
            'cars': [],
            'ball': frame_detections.get('ball'),
            'boost_pads': frame_detections.get('boost_pads', [])
        }
        
        for car in frame_detections.get('cars', []):
            car_data = {
                'track_id': int(car['track_id']),
                'bbox': [float(x) for x in car['bbox']],
                'center': [float(x) for x in car['center']],
                'confidence': float(car['confidence']),
                'speed': float(car['speed']),
                'class': int(car['class'])
            }
            frame_data['cars'].append(car_data)
            
        serializable_detections.append(frame_data)
        
    with open(output_path, 'w') as f:
        json.dump(serializable_detections, f, indent=2)
        
    print(f"Detections saved to: {output_path}")


def calculate_average_speed(detections, track_id=None):
    """
    Calculate average speed across frames.
    
    Args:
        detections (list): List of detection dictionaries
        track_id (int): Specific track ID to calculate for. If None, calculates for all.
        
    Returns:
        dict or float: Average speed(s)
    """
    if track_id is not None:
        speeds = []
        for frame_detections in detections:
            for car in frame_detections.get('cars', []):
                if car['track_id'] == track_id:
                    speeds.append(car['speed'])
        return sum(speeds) / len(speeds) if speeds else 0.0
    else:
        # Calculate for all tracks
        track_speeds = {}
        for frame_detections in detections:
            for car in frame_detections.get('cars', []):
                tid = car['track_id']
                if tid not in track_speeds:
                    track_speeds[tid] = []
                track_speeds[tid].append(car['speed'])
                
        avg_speeds = {tid: sum(speeds) / len(speeds) for tid, speeds in track_speeds.items()}
        return avg_speeds


def get_max_speed(detections, track_id=None):
    """
    Get maximum speed recorded.
    
    Args:
        detections (list): List of detection dictionaries
        track_id (int): Specific track ID. If None, returns max across all tracks.
        
    Returns:
        float: Maximum speed
    """
    max_speed = 0.0
    
    for frame_detections in detections:
        for car in frame_detections.get('cars', []):
            if track_id is None or car['track_id'] == track_id:
                max_speed = max(max_speed, car['speed'])
                
    return max_speed


def create_output_directory(output_dir='output'):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Directory path
        
    Returns:
        Path: Path object for the directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return output_path
