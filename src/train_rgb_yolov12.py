#!/usr/bin/env python
"""
Simple script to train YOLOv12 model on RGB snow pole data.
"""
import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv12 model for snow pole detection')
    parser.add_argument('--model', type=str, default='results/models/yolov12n.pt',
                        help='Path to pretrained YOLOv12 model')
    parser.add_argument('--data', type=str, default='data/rgb/data.yaml',
                        help='Path to data YAML file')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (if None, automatically determined based on model)')
    parser.add_argument('--save_period', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--run_name', type=str, default='run1',
                        help='Name for this run (used as subfolder name)')
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Determine model type from model path
    model_path = Path(args.model)
    model_name = model_path.stem  # Get model name without extension
    
    # Determine output directory based on model type if not specified
    if args.output_dir is None:
        if 'yolov12n' in model_name:
            args.output_dir = 'results/rgb/yolov12n'
        elif 'yolov12s' in model_name:
            args.output_dir = 'results/rgb/yolov12s'
        else:
            # Default to yolov12n folder for unknown models
            args.output_dir = 'results/rgb/yolov12n'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting YOLOv12 training with the following parameters:")
    print(f"- Model: {args.model} ({model_name})")
    print(f"- Data: {args.data}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Save checkpoint period: every {args.save_period} epochs")
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    
    # Load model
    model = YOLO(args.model)
    
    # Train model with recommended YOLOv12n parameters
    # IMPORTANT: Disable validation completely to avoid NMS errors
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=640,
        scale=0.5,           # YOLOv12n recommended value
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        project=args.output_dir,
        name=args.run_name,
        exist_ok=True,
        val=True,           # CRITICAL: Disable validation during training
        device=device,
        save_period=args.save_period,
        amp=False,           # Disable mixed precision to avoid potential issues
        plots=True,
        verbose=True,
        hsv_h=0.015,        # Minimal color augmentation for LiDAR
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,        # Minimal rotation for LiDAR
        translate=0.1,      # Translation augmentation
        fliplr=0,         # Horizontal flip
        optimizer='SGD',    # SGD optimizer for LiDAR
        lr0=0.002,           # Initial learning rate: 10^-2
        lrf=0.01,   
    )
    
    print("Training completed!")
    print(f"Final model saved to: {args.output_dir}/{args.run_name}/weights/last.pt")
    print(f"\nTo validate the model, use: python src/validate_rgb_yolov12.py --model {args.output_dir}/{args.run_name}/weights/best.pt --data {args.data}")

if __name__ == '__main__':
    main() 