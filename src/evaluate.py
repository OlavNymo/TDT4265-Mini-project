#!/usr/bin/env python
"""
Evaluation script for snow pole detection model using YOLO.
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.yolo_model import YOLOModel
from src.evaluation.metrics import plot_precision_recall_curve
from src.utils.utils import (
    load_config,
    merge_configs,
    Timer
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for snow pole detection')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/lidar_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--base_config', type=str, default='config/base_config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data YAML (overrides config)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--conf_thres', type=float, default=None,
                        help='Confidence threshold for detection (overrides config)')
    parser.add_argument('--iou_thres', type=float, default=None,
                        help='IoU threshold for NMS (overrides config)')
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Load configurations
    base_config = load_config(args.base_config)
    specific_config = load_config(args.config)
    config = merge_configs(base_config, specific_config)
    
    # Override config with command line arguments if provided
    if args.conf_thres is not None:
        config['evaluation']['conf_threshold'] = args.conf_thres
    if args.iou_thres is not None:
        config['evaluation']['iou_threshold'] = args.iou_thres
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine dataset type from config
    dataset_name = config['dataset']['name']
    print(f"Evaluating model on {dataset_name} dataset")
    
    # Set up data_yaml path
    if args.data is not None:
        data_yaml = args.data
    else:
        # Use the configured paths to create a data YAML
        data_yaml = {
            'test': config['dataset']['test_path'],
            'val': config['dataset']['val_path'],
            'nc': config['dataset']['num_classes'],
            'names': config['dataset']['classes']
        }
        
        # Save data YAML for YOLO
        yaml_path = output_dir / 'data.yaml'
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        data_yaml = str(yaml_path)
    
    # Initialize model
    model = YOLOModel(config)
    model.load(args.model)
    print(f"Loaded model from {args.model}")
    
    # Evaluate model
    with Timer("Evaluation") as timer:
        results = model.model.val(
            data=data_yaml,
            conf=config['evaluation']['conf_threshold'],
            iou=config['evaluation']['iou_threshold'],
            verbose=True,
            save_json=True,
            save_dir=str(output_dir)
        )
    
    # Extract results
    metrics = results.results_dict
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print key metrics
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    
    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@50: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")
    print(f"Evaluation completed in {timer.elapsed:.2f} seconds")
    
    # Create visualization for PR curve if available
    if hasattr(results, 'pr_curve') and results.pr_curve is not None:
        pr_curve = results.pr_curve
        precisions = pr_curve['precision']
        recalls = pr_curve['recall']
        
        # Plot P-R curve
        plot_path = output_dir / 'precision_recall_curve.png'
        plot_precision_recall_curve(precisions, recalls, str(plot_path))
        print(f"Saved P-R curve to {plot_path}")
        
    # Estimate energy usage
    energy_usage = timer.estimate_energy(
        gpu_usage=1.0 if torch.cuda.is_available() else 0,
        cpu_count=config['evaluation']['workers']
    )
    
    print(f"Estimated energy usage: {energy_usage:.6f} kWh")
    
    # Calculate Tesla Model Y Range Equivalent 
    # Assuming 6.2 kWh per mile for Tesla Model Y
    range_equivalent = energy_usage / 0.062
    print(f"Equivalent Tesla Model Y range: {range_equivalent:.6f} miles")


if __name__ == '__main__':
    main() 