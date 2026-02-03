"""
Main script to run Rocket League object tracking
"""

import argparse
from pathlib import Path
from rocket_league_cv.tracker import RocketLeagueTracker
from rocket_league_cv.utils import (
    load_config, 
    save_detections, 
    calculate_average_speed,
    get_max_speed,
    create_output_directory
)


def main():
    parser = argparse.ArgumentParser(description='Rocket League CV - Track cars and ball in gameplay footage')
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None, help='Path to custom trained model (optional)')
    parser.add_argument('--output', type=str, default=None, help='Path for output video (default: output/tracked_<input_name>.mp4)')
    parser.add_argument('--no-save', action='store_true', help='Do not save output video')
    parser.add_argument('--save-detections', action='store_true', help='Save detection data to JSON')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Disable video saving if requested
    if args.no_save:
        config['output']['save_video'] = False
        
    # Create output directory
    output_dir = create_output_directory()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        video_name = Path(args.video).stem
        output_path = output_dir / f"tracked_{video_name}.mp4"
        output_path = str(output_path)
        
    # Initialize tracker
    print("Initializing tracker...")
    tracker = RocketLeagueTracker(config)
    
    # Load model
    print("Loading model...")
    tracker.load_model(args.model)
    
    # Process video
    print(f"\nProcessing video: {args.video}")
    detections = tracker.track_video(args.video, output_path if config['output']['save_video'] else None)
    
    # Print statistics
    print("\n" + "="*50)
    print("TRACKING STATISTICS")
    print("="*50)
    
    avg_speeds = calculate_average_speed(detections)
    for track_id, avg_speed in avg_speeds.items():
        max_speed = get_max_speed(detections, track_id)
        print(f"Track {track_id}:")
        print(f"  Average Speed: {avg_speed:.2f} m/s")
        print(f"  Max Speed: {max_speed:.2f} m/s")
        
    # Overall max speed
    overall_max = get_max_speed(detections)
    print(f"\nOverall Max Speed: {overall_max:.2f} m/s")
    print("="*50)
    
    # Save detection data if requested
    if args.save_detections:
        video_name = Path(args.video).stem
        detections_path = output_dir / f"detections_{video_name}.json"
        save_detections(detections, str(detections_path))
        
    print("\nTracking complete!")


if __name__ == '__main__':
    main()
