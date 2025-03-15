#!/usr/bin/env python
"""
Script to validate a trained YOLOv12 model on LiDAR data.
This script is designed to work around validation issues with the LiDAR dataset.
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate YOLOv12 model for LiDAR snow pole detection')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='data/lidar/data.yaml',
                        help='Path to data YAML file')
    parser.add_argument('--output_dir', type=str, default='results/lidar_yolov12_validation',
                        help='Directory to save validation results')
    parser.add_argument('--conf_thres', type=float, default=0.25,
                        help='Confidence threshold for validation')
    parser.add_argument('--iou_thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--show_preview', action='store_true',
                        help='Run inference on a few test images and save results')
    return parser.parse_args()

def main():
    """Main validation function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting YOLOv12 LiDAR validation with the following parameters:")
    print(f"- Model: {args.model}")
    print(f"- Data: {args.data}")
    print(f"- Confidence threshold: {args.conf_thres}")
    print(f"- IoU threshold: {args.iou_thres}")
    print(f"- Output directory: {args.output_dir}")
    
    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file {args.model} does not exist.")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = YOLO(args.model)
        print(f"Model loaded successfully: {model.names}")
        
        # Use a more robust validation approach
        print("\nRunning validation on LiDAR data...")
        
        # Get validation directory from data.yaml
        if 'lidar' in args.data:
            val_dir = Path('/datasets/tdt4265/ad/open/Poles/lidar/combined_color/valid')
        else:
            val_dir = Path('/datasets/tdt4265/ad/open/Poles/lidar/combined_color/valid')
        
        if not val_dir.exists():
            print(f"Error: Validation directory {val_dir} does not exist.")
            return
        
        # Run prediction on validation set
        print(f"Running prediction on validation set: {val_dir}")
        results = model.predict(
            source=str(val_dir),
            conf=args.conf_thres,
            iou=args.iou_thres,
            save=True,
            project=args.output_dir,
            name='validation',
            exist_ok=True,
            imgsz=1024,  # Use a single integer value as recommended in the warning
            rect=True    # Enable rectangular validation mode
        )
        
        # Calculate basic metrics manually
        total_images = len(results)
        images_with_detections = sum(1 for r in results if len(r.boxes) > 0)
        total_detections = sum(len(r.boxes) for r in results)
        
        print("\nBasic Validation Results:")
        print(f"Total images processed: {total_images}")
        print(f"Images with detections: {images_with_detections} ({images_with_detections/total_images*100:.2f}%)")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/total_images:.2f}")
        
        # Run inference on a few test images if requested
        if args.show_preview:
            print("\nGenerating preview images...")
            
            # Get a sample of validation images
            val_images = list(val_dir.glob('*.png'))[:5]  # First 5 validation images
            
            if val_images:
                preview_dir = output_dir / 'preview_images'
                preview_dir.mkdir(parents=True, exist_ok=True)
                
                for img in tqdm(val_images, desc="Processing preview images"):
                    result = model.predict(
                        source=img,
                        conf=args.conf_thres,
                        iou=args.iou_thres,
                        save=True,
                        project=args.output_dir,
                        name='preview_images',
                        exist_ok=True
                    )
                
                print(f"Preview images saved to {preview_dir}")
            else:
                print("No validation images found for preview.")
        
    except Exception as e:
        print(f"Error during validation: {e}")

if __name__ == '__main__':
    main() 