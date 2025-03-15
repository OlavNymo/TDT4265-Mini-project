#!/usr/bin/env python
"""
Script to check the progress of training.
"""
import os
import sys
import argparse
from pathlib import Path
import glob
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check training progress')
    parser.add_argument('--run_dir', type=str, default='results/yolov12_training/run1',
                        help='Directory of the training run')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    
    # Check if directory exists
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        return
    
    # Check weights directory
    weights_dir = run_dir / 'weights'
    if not weights_dir.exists():
        print(f"Weights directory {weights_dir} does not exist.")
        return
    
    # List all weight files
    weight_files = sorted(glob.glob(str(weights_dir / '*.pt')))
    
    if not weight_files:
        print("No weight files found. Training might still be in progress.")
    else:
        print(f"Found {len(weight_files)} weight files:")
        for i, file in enumerate(weight_files):
            file_path = Path(file)
            size_mb = os.path.getsize(file) / (1024 * 1024)
            mod_time = time.ctime(os.path.getmtime(file))
            print(f"  {i+1}. {file_path.name} - {size_mb:.2f} MB (Modified: {mod_time})")
    
    # Check for log files (TensorBoard events)
    event_files = glob.glob(str(run_dir / 'events.out.tfevents.*'))
    if event_files:
        print(f"\nFound {len(event_files)} TensorBoard event files.")
    else:
        print("\nNo TensorBoard event files found.")
    
    # Check for results.csv
    results_file = run_dir / 'results.csv'
    if results_file.exists():
        print("\nResults file exists. Last few lines:")
        with open(results_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:  # Show last 5 lines
                print(f"  {line.strip()}")
    else:
        print("\nNo results.csv file found yet.")

if __name__ == '__main__':
    main() 