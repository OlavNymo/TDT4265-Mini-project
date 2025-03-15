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
    parser.add_argument('--output_dir', type=str, default='results/yolov12_training',
                        help='Directory to save results')
    parser.add_argument('--save_period', type=int, default=5,
                        help='Save checkpoint every N epochs')
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting YOLOv12 training with the following parameters:")
    print(f"- Model: {args.model}")
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
        name='run4',
        exist_ok=True,
        val=False,           # CRITICAL: Disable validation during training
        device=device,
        save_period=args.save_period,
        amp=False,           # Disable mixed precision to avoid potential issues
        plots=True,
        verbose=True
    )
    
    print("Training completed!")
    print(f"Final model saved to: {args.output_dir}/run4/weights/last.pt")
    print("\nTo validate the model, use: python src/validate_yolov12.py --model {args.output_dir}/run4/weights/last.pt --data {args.data}")

if __name__ == '__main__':
    main() 