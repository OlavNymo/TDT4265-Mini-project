#!/usr/bin/env python
"""
Prediction script for running inference on images using the trained YOLO model.
"""
import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.yolo_model import YOLOModel
from src.utils.utils import load_config, merge_configs, Timer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on images using trained YOLO model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/lidar_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--base_config', type=str, default='config/base_config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (if not provided, --image_dir must be)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images for inference (if not provided, --image must be)')
    parser.add_argument('--output_dir', type=str, default='results/predictions',
                        help='Directory to save output visualizations')
    parser.add_argument('--conf_thres', type=float, default=None,
                        help='Confidence threshold for detection (overrides config)')
    parser.add_argument('--iou_thres', type=float, default=None,
                        help='IoU threshold for NMS (overrides config)')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save YOLO format detections to text files')
    return parser.parse_args()


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    
    # Check that either image or image_dir is provided
    if args.image is None and args.image_dir is None:
        print("Error: Either --image or --image_dir must be provided")
        sys.exit(1)
    
    # Load configurations
    base_config = load_config(args.base_config)
    specific_config = load_config(args.config)
    config = merge_configs(base_config, specific_config)
    
    # Override config with command line arguments if provided
    if args.conf_thres is not None:
        config['model']['conf_threshold'] = args.conf_thres
    if args.iou_thres is not None:
        config['model']['iou_threshold'] = args.iou_thres
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = YOLOModel(config)
    model.load(args.model)
    print(f"Loaded model from {args.model}")
    
    # Get list of image paths to process
    if args.image is not None:
        image_paths = [Path(args.image)]
    else:
        # Get all image files in the directory
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.png'))
        
    print(f"Found {len(image_paths)} images to process")
    
    # Run prediction on each image
    conf_thres = config['model']['conf_threshold']
    iou_thres = config['model']['iou_threshold']
    
    with Timer("Inference") as timer:
        for image_path in image_paths:
            print(f"Processing: {image_path}")
            
            # Run prediction
            results = model.predict(str(image_path), conf_thres=conf_thres, iou_thres=iou_thres)
            
            # Save results
            filename = image_path.stem
            result = results[0]  # Get the first result (only one image)
            
            # Save visualization
            save_path = output_dir / f"{filename}_pred.jpg"
            result.save(save_path)
            
            # Save detections in YOLO format if requested
            if args.save_txt:
                txt_path = output_dir / f"{filename}.txt"
                
                # Extract detections
                boxes = result.boxes
                with open(txt_path, 'w') as f:
                    for box in boxes:
                        # Get box coordinates (normalized xywh format)
                        x, y, w, h = box.xywhn[0].cpu().numpy()
                        # Get class id and confidence
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        # Write YOLO format line: class_id x_center y_center width height confidence
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
    
    print(f"Processed {len(image_paths)} images in {timer.elapsed:.2f} seconds")
    print(f"Results saved to {output_dir}")
    
    # If only one image was processed, display the path to the result
    if len(image_paths) == 1:
        save_path = output_dir / f"{image_paths[0].stem}_pred.jpg"
        print(f"Output visualization: {save_path}")
    
    # Estimate energy usage
    energy_usage = timer.estimate_energy(
        gpu_usage=1.0 if torch.cuda.is_available() else 0,
        cpu_count=2  # Typically fewer cores needed for inference
    )
    
    print(f"Estimated energy usage: {energy_usage:.6f} kWh")


if __name__ == '__main__':
    main() 