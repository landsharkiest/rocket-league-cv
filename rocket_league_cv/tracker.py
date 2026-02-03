"""
Tracker class for detecting and tracking Rocket League objects (cars, ball)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque


class RocketLeagueTracker:
    """
    Main tracker class for detecting and tracking Rocket League game objects.
    Uses YOLOv8 for object detection and ByteTrack for multi-object tracking.
    """
    
    def __init__(self, config):
        """
        Initialize the tracker with configuration.
        
        Args:
            config (dict): Configuration dictionary containing model and tracking settings
        """
        self.config = config
        self.model = None
        self.track_history = defaultdict(lambda: deque(maxlen=config['output']['trail_length']))
        self.speed_history = defaultdict(lambda: deque(maxlen=config['speed']['smoothing_window']))
        self.prev_positions = {}
        
    def load_model(self, model_path=None):
        """
        Load YOLO model for object detection.
        
        Args:
            model_path (str): Path to custom trained model. If None, uses pretrained YOLO model.
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Use pretrained YOLOv8 model (will need to be fine-tuned for Rocket League)
            model_size = self.config['model']['size']
            self.model = YOLO(f'yolov8{model_size}.pt')
            
        print(f"Model loaded: {self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8'}")
        
    def calculate_speed(self, track_id, current_pos, fps):
        """
        Calculate speed of an object based on position change.
        
        Args:
            track_id: Unique identifier for the tracked object
            current_pos: Current position (x, y)
            fps: Frames per second of the video
            
        Returns:
            float: Speed in meters per second (requires calibration)
        """
        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = current_pos
            return 0.0
            
        prev_pos = self.prev_positions[track_id]
        
        # Calculate distance in pixels
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters using calibration factor
        pixels_per_meter = self.config['speed']['pixels_per_meter']
        distance_meters = distance_pixels / pixels_per_meter
        
        # Calculate speed (distance per frame * fps = distance per second)
        speed = distance_meters * fps
        
        # Update position
        self.prev_positions[track_id] = current_pos
        
        # Smooth speed using moving average
        self.speed_history[track_id].append(speed)
        smoothed_speed = np.mean(list(self.speed_history[track_id]))
        
        return smoothed_speed
        
    def track_frame(self, frame, fps=30):
        """
        Process a single frame to detect and track objects.
        
        Args:
            frame: Input video frame (numpy array)
            fps: Frames per second of the video
            
        Returns:
            tuple: (annotated_frame, detections_dict)
                annotated_frame: Frame with drawn bounding boxes and tracking info
                detections_dict: Dictionary with tracking information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Run tracking on the frame
        results = self.model.track(
            frame,
            persist=True,
            conf=self.config['model']['confidence_threshold'],
            iou=self.config['model']['iou_threshold'],
            tracker=f"{self.config['tracking']['tracker_type']}.yaml"
        )
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Extract detection information
        detections = {
            'cars': [],
            'ball': None,
            'boost_pads': []
        }
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Calculate speed
                speed = self.calculate_speed(track_id, (center_x, center_y), fps)
                
                # Store track history for trails
                self.track_history[track_id].append((int(center_x), int(center_y)))
                
                detection_info = {
                    'track_id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf,
                    'speed': speed,
                    'class': cls
                }
                
                # Categorize detection (in actual use, you'd have custom class IDs)
                # For now, treating all as potential objects
                detections['cars'].append(detection_info)
                
        # Draw trails if enabled
        if self.config['output']['draw_trails']:
            for track_id, track in self.track_history.items():
                if len(track) > 1:
                    points = np.array(list(track), dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], False, (0, 255, 0), 2)
                    
        # Draw speed information if enabled
        if self.config['output']['show_speed']:
            for detection in detections['cars']:
                x1, y1, x2, y2 = detection['bbox']
                speed_text = f"{detection['speed']:.1f} m/s"
                cv2.putText(
                    annotated_frame,
                    speed_text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
        return annotated_frame, detections
        
    def track_video(self, video_path, output_path=None):
        """
        Process an entire video file.
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path to save output video. If None, doesn't save.
            
        Returns:
            list: List of detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path and self.config['output']['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        all_detections = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Track objects in frame
                annotated_frame, detections = self.track_frame(frame, fps)
                all_detections.append(detections)
                
                # Write frame to output video
                if writer is not None:
                    writer.write(annotated_frame)
                    
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
                    
        finally:
            cap.release()
            if writer is not None:
                writer.release()
                print(f"Output video saved to: {output_path}")
                
        print(f"Tracking complete. Processed {frame_count} frames.")
        return all_detections
