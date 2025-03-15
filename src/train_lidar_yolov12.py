#!/usr/bin/env python
"""
Script to train YOLOv12 model on LiDAR snow pole data.
This script creates temporary symlinks in a working directory
to point labels to where YOLOv12 expects them to be.
"""
import os
import sys
import argparse
import yaml
import glob
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv12 model for LiDAR snow pole detection')
    parser.add_argument('--model', type=str, default='results/models/yolov12n.pt',
                        help='Path to pretrained YOLOv12 model')
    parser.add_argument('--data', type=str, default='data/lidar/data.yaml',
                        help='Path to data YAML file')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='results/lidar_yolov12_training',
                        help='Directory to save results')
    parser.add_argument('--save_period', type=int, default=5,
                        help='Save checkpoint every N epochs')
    return parser.parse_args()

def create_temp_links(data_config, temp_dir):
    """
    Create temporary symbolic links for dataset so labels are in the same
    directory as images, which is what YOLOv12 expects.
    
    Args:
        data_config: Data configuration with paths
        temp_dir: Temporary directory to create structure
        
    Returns:
        tuple: (train_dir, val_dir, test_dir) paths
    """
    # Extract paths from config
    orig_train_img_dir = data_config.get('train', '')
    orig_val_img_dir = data_config.get('val', '')
    orig_test_img_dir = data_config.get('test', '')
    
    orig_train_label_dir = data_config.get('train_labels', '')
    orig_val_label_dir = data_config.get('val_labels', '')
    orig_test_label_dir = data_config.get('test_labels', '')
    
    # Create temp directories
    temp_train_dir = os.path.join(temp_dir, 'train')
    temp_val_dir = os.path.join(temp_dir, 'valid')
    temp_test_dir = os.path.join(temp_dir, 'test')
    
    os.makedirs(temp_train_dir, exist_ok=True)
    os.makedirs(temp_val_dir, exist_ok=True)
    os.makedirs(temp_test_dir, exist_ok=True)
    
    # Create symbolic links for images
    print("Creating symbolic links for images...")
    create_symlinks_for_directory(orig_train_img_dir, temp_train_dir, "*.png")
    create_symlinks_for_directory(orig_val_img_dir, temp_val_dir, "*.png")
    create_symlinks_for_directory(orig_test_img_dir, temp_test_dir, "*.png")
    
    # Create hard copies for labels (YOLOv12 expects them in the same directory)
    print("Creating hard copies of label files...")
    if orig_train_label_dir:
        copy_labels_to_img_dir(orig_train_label_dir, temp_train_dir)
    if orig_val_label_dir:
        copy_labels_to_img_dir(orig_val_label_dir, temp_val_dir)
    if orig_test_label_dir:
        copy_labels_to_img_dir(orig_test_label_dir, temp_test_dir)
    
    return os.path.abspath(temp_train_dir), os.path.abspath(temp_val_dir), os.path.abspath(temp_test_dir)

def create_symlinks_for_directory(source_dir, target_dir, pattern):
    """Create symbolic links for files matching pattern from source to target directory."""
    if not source_dir or not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} doesn't exist. Skipping.")
        return 0
    
    files = glob.glob(os.path.join(source_dir, pattern))
    count = 0
    
    for file_path in files:
        filename = os.path.basename(file_path)
        link_path = os.path.join(target_dir, filename)
        
        try:
            # Use absolute path for symlink source
            os.symlink(os.path.abspath(file_path), link_path)
            count += 1
        except FileExistsError:
            # Skip if already exists
            pass
        except Exception as e:
            print(f"Error creating symlink for {filename}: {e}")
    
    print(f"Created {count} symlinks in {target_dir}")
    return count

def copy_labels_to_img_dir(label_dir, img_dir):
    """Copy label files from label directory to image directory."""
    if not label_dir or not os.path.exists(label_dir):
        print(f"Warning: Label directory {label_dir} doesn't exist. Skipping.")
        return 0
    
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    count = 0
    
    for label_path in label_files:
        label_filename = os.path.basename(label_path)
        img_filename = os.path.splitext(label_filename)[0] + ".png"
        
        # Only copy if corresponding image exists in target directory
        if os.path.exists(os.path.join(img_dir, img_filename)):
            target_path = os.path.join(img_dir, label_filename)
            try:
                shutil.copy2(label_path, target_path)
                count += 1
            except Exception as e:
                print(f"Error copying {label_filename}: {e}")
    
    print(f"Copied {count} label files to {img_dir}")
    return count

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Get absolute path for output directory
    output_dir = os.path.abspath(Path(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for dataset
    temp_dataset_dir = os.path.join(output_dir, "temp_dataset")
    os.makedirs(temp_dataset_dir, exist_ok=True)
    
    # Load data configuration
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Starting YOLOv12 LiDAR training with the following parameters:")
    print(f"- Model: {args.model}")
    print(f"- Data: {args.data}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Output directory: {output_dir}")
    print(f"- Save checkpoint period: every {args.save_period} epochs")
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create temp dataset structure with symlinks
    print("\nPreparing temporary dataset structure...")
    temp_train_dir, temp_val_dir, temp_test_dir = create_temp_links(data_config, temp_dataset_dir)
    
    # Create a new data.yaml for our temp dataset
    temp_data_yaml_path = os.path.join(output_dir, "temp_data.yaml")
    
    # Create a direct data.yaml with absolute paths
    temp_data_config = {
        'path': '',  # Use empty path to prevent Ultralytics from prepending anything
        'train': temp_train_dir,  # Absolute paths
        'val': temp_val_dir,      # Absolute paths
        'test': temp_test_dir,    # Absolute paths
        'nc': data_config.get('nc', 1),
        'names': data_config.get('names', ['pole'])
    }
    
    # Write the temp data config
    with open(temp_data_yaml_path, 'w') as f:
        yaml.dump(temp_data_config, f)
    
    print(f"Created temporary data configuration at {temp_data_yaml_path}")
    print(f"  Train path: {temp_train_dir}")
    print(f"  Val path: {temp_val_dir}")
    print(f"  Test path: {temp_test_dir}")
    
    # Load model
    model = YOLO(args.model)
    
    # Train model with LiDAR-specific parameters
    try:
        print("\nStarting training...")
        results = model.train(
            data=temp_data_yaml_path,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=640,
            scale=0.5,          # YOLOv12n recommended value
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.1,
            project=output_dir,
            name='run1',
            exist_ok=True,
            val=False,          # Disable validation during training
            device=device,
            save_period=args.save_period,
            amp=False,          # Disable mixed precision
            plots=True,
            verbose=True,
            hsv_h=0.015,        # Minimal color augmentation for LiDAR
            hsv_s=0.2,
            hsv_v=0.2,
            degrees=0.0,        # Minimal rotation for LiDAR
            translate=0.1,      # Translation augmentation
            fliplr=0.5,         # Horizontal flip
            optimizer='Adam'    # Adam optimizer often works better for LiDAR
        )
        
        print("Training completed!")
        print(f"Final model saved to: {output_dir}/run1/weights/last.pt")
        print(f"Best model saved to: {output_dir}/run1/weights/best.pt")
        print(f"\nTo validate the model, use: python src/validate_lidar_yolov12.py --model {output_dir}/run1/weights/best.pt --data {args.data}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 