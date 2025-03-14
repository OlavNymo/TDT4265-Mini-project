#!/usr/bin/env python
"""
Training script for snow pole detection using YOLO.
"""
import os
import sys
import argparse
import torch
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.yolo_model import YOLOModel, create_data_yaml
from src.utils.utils import (
    load_config, 
    merge_configs, 
    get_experiment_name, 
    create_experiment_dir, 
    save_config,
    set_seed,
    Timer
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO model for snow pole detection')
    parser.add_argument('--config', type=str, default='config/lidar_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--base_config', type=str, default='config/base_config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configurations
    base_config = load_config(args.base_config)
    specific_config = load_config(args.config)
    config = merge_configs(base_config, specific_config)
    
    # Override config with command line arguments if provided
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.seed is not None:
        config['training']['seed'] = args.seed
    
    # Set random seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Create experiment name and directories
    experiment_name = get_experiment_name(config)
    exp_dirs = create_experiment_dir(config, experiment_name)
    
    # Save merged configuration
    config_path = Path(exp_dirs['log_dir']) / 'config.yaml'
    save_config(config, config_path)
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Training on dataset: {config['dataset']['name']}")
    
    # Create data YAML for YOLO
    data_yaml_path = Path(exp_dirs['log_dir']) / 'data.yaml'
    yaml_path = create_data_yaml(config, data_yaml_path)
    
    # Initialize model
    model = YOLOModel(config)
    
    # Train model
    with Timer("Training") as timer:
        results = model.train(
            data_config=str(yaml_path),
            output_dir=str(exp_dirs['model_dir']),
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size']
        )
    
    # Log energy usage for sustainability report
    energy_usage = timer.estimate_energy(
        gpu_usage=1.0 if torch.cuda.is_available() else 0,
        cpu_count=config['training']['workers']
    )
    
    print(f"Training completed in {timer.elapsed:.2f} seconds")
    print(f"Estimated energy usage: {energy_usage:.4f} kWh")
    
    # Print Tesla Model Y Range Equivalent
    # Assuming 6.2 kWh per mile for Tesla Model Y
    range_equivalent = energy_usage / 0.062
    print(f"Equivalent Tesla Model Y range: {range_equivalent:.2f} miles")


if __name__ == '__main__':
    main() 