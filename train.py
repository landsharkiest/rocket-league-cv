"""
Script for training a custom YOLOv8 model on Rocket League footage
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_yaml='dataset.yaml',
    model_size='n',
    epochs=100,
    batch_size=16,
    img_size=640,
    pretrained=True,
    device='',
    project='runs/train',
    name='rocket_league_model'
):
    """
    Train a YOLOv8 model on Rocket League dataset.
    
    Args:
        data_yaml (str): Path to dataset YAML configuration
        model_size (str): Model size - n, s, m, l, or x
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Input image size
        pretrained (bool): Use pretrained weights
        device (str): Device to use ('', '0', '0,1', 'cpu')
        project (str): Project directory
        name (str): Experiment name
    """
    
    # Load model
    if pretrained:
        model_path = f'yolov8{model_size}.pt'
        print(f"Loading pretrained model: {model_path}")
    else:
        model_path = f'yolov8{model_size}.yaml'
        print(f"Training from scratch using: {model_path}")
        
    model = YOLO(model_path)
    
    # Verify dataset exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml}\n"
            f"Please create this file or use --data to specify the correct path."
        )
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {data_yaml}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Pretrained: {pretrained}")
    print(f"Device: {device if device else 'auto'}")
    print(f"Output: {project}/{name}")
    print("="*60 + "\n")
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        verbose=True,
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,  # Cache images for faster training (requires more RAM)
        plots=True,  # Save training plots
        val=True,  # Validate during training
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    print(f"Last model saved to: {project}/{name}/weights/last.pt")
    print("\nTo use the trained model:")
    print(f"  python main.py video.mp4 --model {project}/{name}/weights/best.pt")
    print("="*60)
    
    return results


def validate_model(model_path, data_yaml='dataset.yaml', img_size=640):
    """
    Validate a trained model on the test set.
    
    Args:
        model_path (str): Path to trained model
        data_yaml (str): Path to dataset YAML configuration
        img_size (int): Input image size
    """
    print(f"Validating model: {model_path}")
    
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        split='test',  # Use test split
        plots=True
    )
    
    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results


def export_model(model_path, format='onnx'):
    """
    Export trained model to different formats for deployment.
    
    Args:
        model_path (str): Path to trained model
        format (str): Export format (onnx, torchscript, coreml, etc.)
    """
    print(f"Exporting model to {format} format...")
    
    model = YOLO(model_path)
    model.export(format=format)
    
    print(f"Model exported successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for Rocket League object detection'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', type=str, default='dataset.yaml',
                            help='Path to dataset YAML file')
    train_parser.add_argument('--model', type=str, default='n',
                            choices=['n', 's', 'm', 'l', 'x'],
                            help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    train_parser.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_parser.add_argument('--batch', type=int, default=16,
                            help='Batch size')
    train_parser.add_argument('--imgsz', type=int, default=640,
                            help='Input image size')
    train_parser.add_argument('--no-pretrained', action='store_true',
                            help='Train from scratch without pretrained weights')
    train_parser.add_argument('--device', type=str, default='',
                            help='Device to use (empty for auto, 0 for GPU 0, cpu for CPU)')
    train_parser.add_argument('--project', type=str, default='runs/train',
                            help='Project directory')
    train_parser.add_argument('--name', type=str, default='rocket_league_model',
                            help='Experiment name')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate a trained model')
    val_parser.add_argument('model', type=str, help='Path to trained model')
    val_parser.add_argument('--data', type=str, default='dataset.yaml',
                          help='Path to dataset YAML file')
    val_parser.add_argument('--imgsz', type=int, default=640,
                          help='Input image size')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export a trained model')
    export_parser.add_argument('model', type=str, help='Path to trained model')
    export_parser.add_argument('--format', type=str, default='onnx',
                             choices=['onnx', 'torchscript', 'coreml', 'tflite'],
                             help='Export format')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            pretrained=not args.no_pretrained,
            device=args.device,
            project=args.project,
            name=args.name
        )
    elif args.command == 'validate':
        validate_model(args.model, args.data, args.imgsz)
    elif args.command == 'export':
        export_model(args.model, args.format)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
