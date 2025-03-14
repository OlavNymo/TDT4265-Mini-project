"""
Utility functions for snow pole detection project.
"""
import os
import yaml
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config, specific_config):
    """
    Merge base configuration with dataset-specific configuration.
    
    Args:
        base_config (dict): Base configuration dictionary
        specific_config (dict): Dataset-specific configuration dictionary
        
    Returns:
        dict: Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    # Recursive merge function
    def merge_dict(d1, d2):
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                merge_dict(d1[k], v)
            else:
                d1[k] = v
    
    merge_dict(merged_config, specific_config)
    return merged_config


def get_experiment_name(config):
    """
    Generate a unique experiment name based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        str: Unique experiment name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config['dataset']['name']
    model_name = config['model']['name']
    
    return f"{dataset_name}_{model_name}_{timestamp}"


def create_experiment_dir(config, experiment_name):
    """
    Create directory structure for experiment outputs.
    
    Args:
        config (dict): Configuration dictionary
        experiment_name (str): Unique experiment name
        
    Returns:
        dict: Dictionary with paths to experiment directories
    """
    # Get base directories from config
    base_output_dir = Path(config['paths']['output_dir'])
    model_dir = base_output_dir / 'models' / experiment_name
    log_dir = base_output_dir / 'logs' / experiment_name
    viz_dir = base_output_dir / 'visualizations' / experiment_name
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Return paths
    return {
        'model_dir': model_dir,
        'log_dir': log_dir,
        'viz_dir': viz_dir
    }


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str or Path): Path to save the configuration file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def calculate_mAP(precision, recall):
    """
    Calculate mean Average Precision (mAP).
    
    Args:
        precision (np.ndarray): Precision values
        recall (np.ndarray): Recall values
        
    Returns:
        float: mAP value
    """
    # Calculate area under PR curve using trapezoidal rule
    mAP = np.trapz(precision, recall)
    return mAP


def plot_precision_recall_curve(precision, recall, save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        precision (np.ndarray): Precision values
        recall (np.ndarray): Recall values
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    # Add mAP value to plot
    mAP = calculate_mAP(precision, recall)
    plt.text(0.5, 0.5, f'mAP: {mAP:.4f}', 
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class Timer:
    """Simple timer for tracking execution time and energy consumption estimates."""
    
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.stop()
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self):
        """Stop the timer and calculate elapsed time."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        if self.name:
            print(f"{self.name} completed in {self.elapsed:.4f} seconds")
        
        return self.elapsed
    
    def estimate_energy(self, gpu_usage=1.0, cpu_count=1):
        """
        Estimate energy consumption based on execution time.
        Very rough estimate assuming:
        - ~250W for GPU at full load
        - ~10W per CPU core
        
        Args:
            gpu_usage (float): GPU utilization between 0-1
            cpu_count (int): Number of CPU cores used
            
        Returns:
            float: Estimated energy consumption in kWh
        """
        # Convert to hours
        hours = self.elapsed / 3600
        
        # Estimate power consumption (kW)
        gpu_power = 0.25 * gpu_usage if torch.cuda.is_available() else 0
        cpu_power = 0.01 * cpu_count
        total_power = gpu_power + cpu_power
        
        # Calculate energy (kWh)
        energy = total_power * hours
        
        return energy 