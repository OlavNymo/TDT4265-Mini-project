#!/usr/bin/env python
"""
Script to validate a trained YOLOv12 model.
This script is designed to work around NMS compatibility issues.
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate YOLOv12 model for snow pole detection')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='data/rgb/data.yaml',
                        help='Path to data YAML file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save validation results (if None, automatically determined based on model)')
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
    
    # Determine model type from model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file {args.model} does not exist.")
        return
    
    # Try to determine model type from the model file name
    model_name = model_path.stem  # Get model name without extension
    
    # Determine output directory based on model type if not specified
    if args.output_dir is None:
        # Extract run name from model path if possible
        parent_dir = model_path.parent.parent  # weights/best.pt -> parent.parent is the run folder
        run_name = parent_dir.name if 'weights' in str(model_path.parent) else 'validation'
        
        if 'yolov12n' in model_name:
            args.output_dir = f'results/rgb/yolov12n/{run_name}/validation'
        elif 'yolov12s' in model_name:
            args.output_dir = f'results/rgb/yolov12s/{run_name}/validation'
        else:
            # Default to yolov12n folder for unknown models
            args.output_dir = f'results/rgb/yolov12n/{run_name}/validation'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting YOLOv12 validation with the following parameters:")
    print(f"- Model: {args.model} ({model_name})")
    print(f"- Data: {args.data}")
    print(f"- Confidence threshold: {args.conf_thres}")
    print(f"- IoU threshold: {args.iou_thres}")
    print(f"- Output directory: {args.output_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = YOLO(args.model)
        print(f"Model loaded successfully: {model.names}")
        
        # Try validation with safer settings
        try:
            print("\nRunning validation...")
            # Use a simpler validation approach to avoid NMS issues
            metrics = model.val(
                data=args.data,
                conf=args.conf_thres,
                iou=args.iou_thres,
                project=args.output_dir,
                name='val_results',
                plots=True,
                save_json=False,  # Avoid potential issues with JSON saving
                verbose=True,
                device=device,
                half=False,  # Avoid half-precision to reduce errors
                batch=1      # Use smallest batch size to avoid memory issues
            )
            
            # Print validation results
            print("\nValidation Results:")
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")
            print(f"Precision: {metrics.box.precision:.4f}")
            print(f"Recall: {metrics.box.recall:.4f}")
            
        except Exception as e:
            print(f"Validation failed: {e}")
            print("\nTrying alternative validation approach...")
            
            # Try a more basic validation approach
            try:
                # Use predict instead of val to avoid NMS issues
                test_dir = Path('/datasets/tdt4265/ad/open/Poles/rgb/images/valid')
                if test_dir.exists():
                    print(f"Running prediction on validation set: {test_dir}")
                    results = model.predict(
                        source=str(test_dir),
                        conf=args.conf_thres,
                        iou=args.iou_thres,
                        save=True,
                        project=args.output_dir,
                        name='validation_alt',
                        exist_ok=True
                    )
                    print(f"Prediction completed. Results saved to {args.output_dir}/validation_alt/")
                else:
                    print(f"Validation directory not found: {test_dir}")
            except Exception as e:
                print(f"Alternative validation also failed: {e}")
        
        # Run inference on a few test images if requested
        if args.show_preview:
            print("\nRunning inference on test images...")
            
            # Load the model
            model = YOLO(args.model)
            
            # Set the test directory based on the data file
            if 'rgb' in args.data:
                test_dir = Path('/datasets/tdt4265/ad/open/Poles/rgb/images/valid')
            else:
                # Default to RGB test directory
                test_dir = Path('/datasets/tdt4265/ad/open/Poles/rgb/images/valid')
            
            if test_dir.exists():
                test_images = list(test_dir.glob('*.PNG'))[:5]  # First 5 test images
                if test_images:
                    print("\nRunning inference on sample test images...")
                    for img in test_images:
                        print(f"Processing {img}")
                        results = model.predict(
                            source=img,
                            conf=args.conf_thres,
                            iou=args.iou_thres,
                            save=True,
                            project=args.output_dir,
                            name='preview_images',
                            exist_ok=True
                        )
                    print(f"Results saved to {args.output_dir}/preview_images/")
                else:
                    print("No test images found.")
            else:
                print(f"Test directory not found: {test_dir}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == '__main__':
    main() 