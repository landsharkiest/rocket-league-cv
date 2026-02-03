"""
Example script demonstrating how to use the Rocket League CV tracker programmatically
"""

from rocket_league_cv.tracker import RocketLeagueTracker
from rocket_league_cv.utils import load_config, calculate_average_speed, get_max_speed

def example_basic_tracking():
    """Basic example of tracking a video"""
    print("Example 1: Basic Video Tracking")
    print("-" * 50)
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Initialize tracker
    tracker = RocketLeagueTracker(config)
    
    # Load model (using pretrained YOLOv8 - will need fine-tuning)
    tracker.load_model()
    
    # Track video (replace with your video path)
    # detections = tracker.track_video('path/to/your/video.mp4', 'output/tracked.mp4')
    
    print("To run this example, uncomment the line above and provide a video path")
    print()


def example_custom_config():
    """Example with custom configuration"""
    print("Example 2: Custom Configuration")
    print("-" * 50)
    
    # Custom configuration dictionary
    custom_config = {
        'model': {
            'type': 'yolov8',
            'size': 's',  # Use small model for faster processing
            'confidence_threshold': 0.6,  # Higher confidence
            'iou_threshold': 0.45
        },
        'tracking': {
            'tracker_type': 'bytetrack',
            'track_buffer': 30,
            'match_threshold': 0.8
        },
        'speed': {
            'fps': 60,  # High FPS video
            'pixels_per_meter': 15,
            'smoothing_window': 10
        },
        'output': {
            'save_video': True,
            'draw_trails': True,
            'trail_length': 50,
            'show_speed': True,
            'show_confidence': True
        }
    }
    
    # Initialize with custom config
    tracker = RocketLeagueTracker(custom_config)
    tracker.load_model()
    
    print("Tracker initialized with custom configuration")
    print(f"Model size: {custom_config['model']['size']}")
    print(f"Confidence threshold: {custom_config['model']['confidence_threshold']}")
    print()


def example_frame_by_frame():
    """Example of processing video frame by frame"""
    print("Example 3: Frame-by-Frame Processing")
    print("-" * 50)
    
    import cv2
    
    config = load_config('config.yaml')
    tracker = RocketLeagueTracker(config)
    tracker.load_model()
    
    # Open video (replace with your video path)
    # cap = cv2.VideoCapture('path/to/your/video.mp4')
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     
    #     # Track objects in frame
    #     annotated_frame, detections = tracker.track_frame(frame, fps)
    #     
    #     # Process detections
    #     for car in detections['cars']:
    #         print(f"Track {car['track_id']}: Speed = {car['speed']:.2f} m/s")
    #     
    #     # Display frame
    #     cv2.imshow('Tracking', annotated_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # 
    # cap.release()
    # cv2.destroyAllWindows()
    
    print("To run this example, uncomment the code above and provide a video path")
    print()


def example_statistics():
    """Example of calculating statistics from detections"""
    print("Example 4: Calculate Statistics")
    print("-" * 50)
    
    # Assuming you have detections from a previous run
    # detections = [...]  # List of detection dictionaries
    
    # Calculate average speeds for each track
    # avg_speeds = calculate_average_speed(detections)
    # for track_id, speed in avg_speeds.items():
    #     print(f"Track {track_id} average speed: {speed:.2f} m/s")
    
    # Get max speed
    # max_speed = get_max_speed(detections)
    # print(f"Maximum speed recorded: {max_speed:.2f} m/s")
    
    # Get max speed for specific track
    # track_max = get_max_speed(detections, track_id=1)
    # print(f"Track 1 max speed: {track_max:.2f} m/s")
    
    print("This example requires detection data from a previous tracking run")
    print()


if __name__ == '__main__':
    print("="*50)
    print("Rocket League CV - Usage Examples")
    print("="*50)
    print()
    
    example_basic_tracking()
    example_custom_config()
    example_frame_by_frame()
    example_statistics()
    
    print("="*50)
    print("For more information, see README.md")
    print("To run tracking, use: python main.py <video_path>")
    print("="*50)
