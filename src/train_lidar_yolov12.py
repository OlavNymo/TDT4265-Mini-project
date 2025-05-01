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
import numpy as np
from sklearn.model_selection import KFold

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
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (if None, automatically determined based on model)')
    parser.add_argument('--save_period', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--run_name', type=str, default='run1',
                        help='Name for this run (used as subfolder name)')
    parser.add_argument('--cv', type=int, default=0,
                        help='Number of cross-validation folds (0 to disable)')
    parser.add_argument('--metric', type=str, default='map50-95',
                        choices=['map50', 'map50-95'],
                        help='Metric to use for selecting best model')
    parser.add_argument('--no_symlinks', action='store_true',
                        help='Do not use symlinks (copy files instead) - use this if symlinks cause issues')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum number of samples to use (0 for all) - useful to avoid disk quota issues')
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

def prepare_cv_data(data_config, fold_idx, n_folds, temp_dir, use_symlinks=True, max_samples=0):
    """
    Prepare cross-validation data split for LiDAR data.
    
    Args:
        data_config: Data configuration dictionary
        fold_idx: Current fold index
        n_folds: Total number of folds
        temp_dir: Directory to create temp dataset
        use_symlinks: Whether to use symlinks (True) or copy files (False)
        max_samples: Maximum number of samples to use (0 for all)
        
    Returns:
        Dictionary with paths to the cross-validation split data
    """
    # Extract original paths
    orig_train_img_dir = data_config.get('train', '')
    orig_val_img_dir = data_config.get('val', '')
    
    # Determine label paths
    orig_train_label_dir = data_config.get('train_labels', '')
    orig_val_label_dir = data_config.get('val_labels', '')
    
    print(f"Original train path: {orig_train_img_dir}")
    print(f"Original val path: {orig_val_img_dir}")
    
    # If label paths not explicitly defined in config, try to infer them
    if not orig_train_label_dir and orig_train_img_dir:
        # Try standard YOLO directory structure (labels dir next to images)
        parent_dir = os.path.dirname(os.path.dirname(orig_train_img_dir.rstrip('/')))
        orig_train_label_dir = os.path.join(parent_dir, 'labels', 
                                          os.path.basename(orig_train_img_dir.rstrip('/')))
        print(f"Inferred train label path: {orig_train_label_dir}")
    
    if not orig_val_label_dir and orig_val_img_dir:
        parent_dir = os.path.dirname(os.path.dirname(orig_val_img_dir.rstrip('/')))
        orig_val_label_dir = os.path.join(parent_dir, 'labels', 
                                        os.path.basename(orig_val_img_dir.rstrip('/')))
        print(f"Inferred val label path: {orig_val_label_dir}")
    
    # Get all image files (case-insensitive)
    train_img_files = []
    if orig_train_img_dir and os.path.exists(orig_train_img_dir):
        for f in os.listdir(orig_train_img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                train_img_files.append(f)
        print(f"Found {len(train_img_files)} training images")
    
    val_img_files = []
    if orig_val_img_dir and os.path.exists(orig_val_img_dir):
        for f in os.listdir(orig_val_img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                val_img_files.append(f)
        print(f"Found {len(val_img_files)} validation images")
    
    # Combine all files for CV split
    all_files = train_img_files + val_img_files
    
    # Limit samples if requested
    if max_samples > 0 and max_samples < len(all_files):
        print(f"Limiting to {max_samples} samples from original {len(all_files)} images")
        np.random.seed(42)
        all_files = np.random.choice(all_files, max_samples, replace=False).tolist()
    
    print(f"Total images for CV: {len(all_files)}")
    
    if not all_files:
        raise ValueError(f"No image files found for cross-validation. Check paths in data.yaml: {orig_train_img_dir}, {orig_val_img_dir}")
    
    # Shuffle files with fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    # Create KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Get indices for current fold
    all_indices = np.arange(len(all_files))
    fold_splits = list(kf.split(all_indices))
    train_indices, val_indices = fold_splits[fold_idx]
    
    # Create directories for this fold with proper YOLO structure
    fold_dir = os.path.join(temp_dir, f"fold_{fold_idx}")
    fold_train_img_dir = os.path.join(fold_dir, "images", "train")
    fold_val_img_dir = os.path.join(fold_dir, "images", "valid")
    fold_train_label_dir = os.path.join(fold_dir, "labels", "train")
    fold_val_label_dir = os.path.join(fold_dir, "labels", "valid")
    
    # Create all necessary directories
    os.makedirs(fold_train_img_dir, exist_ok=True)
    os.makedirs(fold_val_img_dir, exist_ok=True)
    os.makedirs(fold_train_label_dir, exist_ok=True)
    os.makedirs(fold_val_label_dir, exist_ok=True)
    
    # Create training set for this fold
    train_labels_count = 0
    for idx in train_indices:
        filename = all_files[idx]
        base_name = os.path.splitext(filename)[0]
        
        # Determine source paths
        if filename in train_img_files:
            src_img_path = os.path.join(orig_train_img_dir, filename)
            src_label_path = os.path.join(orig_train_label_dir, f"{base_name}.txt")
        else:
            src_img_path = os.path.join(orig_val_img_dir, filename)
            src_label_path = os.path.join(orig_val_label_dir, f"{base_name}.txt")
        
        # Create symlink for image (always use symlinks for images)
        dst_img_path = os.path.join(fold_train_img_dir, filename)
        if not os.path.exists(dst_img_path):
            try:
                os.symlink(src_img_path, dst_img_path)
            except Exception as e:
                print(f"Error creating symlink for {filename}: {e}")
                continue  # Skip this file if we can't create symlink
        
        # Copy label file if it exists
        dst_label_path = os.path.join(fold_train_label_dir, f"{base_name}.txt")
        if os.path.exists(src_label_path) and not os.path.exists(dst_label_path):
            if use_symlinks:
                try:
                    os.symlink(src_label_path, dst_label_path)
                except Exception as e:
                    print(f"Failed to create symlink for label, trying copy instead: {e}")
                    shutil.copy2(src_label_path, dst_label_path)
            else:
                shutil.copy2(src_label_path, dst_label_path)
            train_labels_count += 1
    
    # Create validation set for this fold
    val_labels_count = 0
    for idx in val_indices:
        filename = all_files[idx]
        base_name = os.path.splitext(filename)[0]
        
        # Determine source paths
        if filename in train_img_files:
            src_img_path = os.path.join(orig_train_img_dir, filename)
            src_label_path = os.path.join(orig_train_label_dir, f"{base_name}.txt")
        else:
            src_img_path = os.path.join(orig_val_img_dir, filename)
            src_label_path = os.path.join(orig_val_label_dir, f"{base_name}.txt")
        
        # Create symlink for image (always use symlinks for images)
        dst_img_path = os.path.join(fold_val_img_dir, filename)
        if not os.path.exists(dst_img_path):
            try:
                os.symlink(src_img_path, dst_img_path)
            except Exception as e:
                print(f"Error creating symlink for {filename}: {e}")
                continue  # Skip this file if we can't create symlink
        
        # Copy label file if it exists
        dst_label_path = os.path.join(fold_val_label_dir, f"{base_name}.txt")
        if os.path.exists(src_label_path) and not os.path.exists(dst_label_path):
            if use_symlinks:
                try:
                    os.symlink(src_label_path, dst_label_path)
                except Exception as e:
                    print(f"Failed to create symlink for label, trying copy instead: {e}")
                    shutil.copy2(src_label_path, dst_label_path)
            else:
                shutil.copy2(src_label_path, dst_label_path)
            val_labels_count += 1
    
    print(f"Created fold {fold_idx+1}/{n_folds} with:")
    print(f"  - {len(train_indices)} training images and {train_labels_count} labels")
    print(f"  - {len(val_indices)} validation images and {val_labels_count} labels")
    
    # Get absolute path for fold directory
    abs_fold_dir = os.path.abspath(fold_dir)
    
    print(f"Fold directory: {abs_fold_dir}")
    
    # Return a dict with the fold directory (to be used for creating data.yaml)
    return {
        'path': abs_fold_dir,
        'train': 'images/train',  # Relative to path
        'val': 'images/valid',    # Relative to path
    }

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
            args.output_dir = 'results/lidar/yolov12n'
        elif 'yolov12s' in model_name:
            args.output_dir = 'results/lidar/yolov12s'
        else:
            # Default to yolov12n folder for unknown models
            args.output_dir = 'results/lidar/yolov12n'
    
    # Get absolute path for output directory
    output_dir = os.path.abspath(Path(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data configuration
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Cross-validation mode
    if args.cv > 1:
        print(f"\nRunning {args.cv}-fold cross-validation")
        
        # Dictionary to store results for each fold
        cv_results = {}
        best_model_path = None
        best_metric_value = 0
        
        # Create directory for CV
        cv_base_dir = os.path.join(output_dir, "cv")
        os.makedirs(cv_base_dir, exist_ok=True)
        
        for fold in range(args.cv):
            print(f"\n{'='*20} Fold {fold+1}/{args.cv} {'='*20}")
            
            # Create temporary directory for this fold's dataset
            fold_temp_dir = os.path.join(cv_base_dir, f"fold_{fold}")
            os.makedirs(fold_temp_dir, exist_ok=True)
            
            # Prepare data for this fold
            fold_paths = prepare_cv_data(data_config, fold, args.cv, fold_temp_dir, not args.no_symlinks, args.max_samples)
            fold_run_name = f"{args.run_name}_fold{fold+1}"
            
            # Create a data.yaml for this fold
            fold_data_yaml_path = os.path.join(fold_temp_dir, "data.yaml")
            fold_data_config = {
                'path': fold_paths['path'],
                'train': fold_paths['train'],
                'val': fold_paths['val'],
                'nc': data_config.get('nc', 1),
                'names': data_config.get('names', ['pole']),
                # Add rectangular dimensions information
                'rect': True,
                'width': 1024,
                'height': 128
            }
            
            with open(fold_data_yaml_path, 'w') as f:
                yaml.dump(fold_data_config, f)
            
            # Load model for this fold
            model = YOLO(args.model)
            
            # Train model for this fold
            try:
                results = model.train(
                    data=fold_data_yaml_path,
                    epochs=args.epochs,
                    batch=args.batch_size,
                    imgsz=1024,       # Changed from single value to tuple
                    rect=True,              # Changed from True to False
                    scale=0.0,               # Prevent poles from being scaled down too much
                    mosaic=1,                # Disable mosaic augmentation
                    mixup=0,                 # Disable mixup
                    copy_paste=0,            # Disable copy-paste
                    project=output_dir,
                    name=fold_run_name,
                    exist_ok=True,
                    val=True,                # Enable validation during training
                    device=device,
                    save_period=args.save_period,
                    plots=True,
                    verbose=True,
                    degrees=4,               # Disable rotation
                    translate=0.1,           # Reduced from 0.2 to 0.1
                    fliplr=0.5,                # Disable horizontal flip
                    optimizer='SGD',
                    lr0=0.01,                # Changed from 0.002 to 0.01
                    lrf=0.01,
                    single_cls=True,
                    auto_augment=False,
                    cos_lr=True,
                    amp=True,
                )
                
                # Get metrics for this fold
                metrics = results.results_dict
                if args.metric == 'map50':
                    metric_value = metrics.get('metrics/mAP50(B)', 0)
                    metric_name = 'mAP@50'
                else:  # map50-95
                    metric_value = metrics.get('metrics/mAP50-95(B)', 0)
                    metric_name = 'mAP@50-95'
                
                cv_results[fold] = metric_value
                print(f"Fold {fold+1} {metric_name}: {metric_value:.4f}")
                
                # Keep track of best model across folds
                fold_model_path = os.path.join(output_dir, fold_run_name, 'weights', 'best.pt')
                if metric_value > best_metric_value and os.path.exists(fold_model_path):
                    best_metric_value = metric_value
                    best_model_path = fold_model_path
                    print(f"New best model found in fold {fold+1}")
                
            except Exception as e:
                print(f"Training failed for fold {fold+1} with error: {e}")
                import traceback
                traceback.print_exc()
                cv_results[fold] = 0
        
        # Calculate and display CV results
        if cv_results:
            mean_metric = np.mean(list(cv_results.values()))
            std_metric = np.std(list(cv_results.values()))
            print(f"\nCross-validation results:")
            print(f"Mean {metric_name}: {mean_metric:.4f} ± {std_metric:.4f}")
            
            # Copy best model to final location
            if best_model_path:
                final_model_path = os.path.join(output_dir, args.run_name, 'weights')
                os.makedirs(final_model_path, exist_ok=True)
                shutil.copy2(best_model_path, os.path.join(final_model_path, 'best.pt'))
                print(f"Best model (from fold with {metric_name}={best_metric_value:.4f}) copied to: {final_model_path}/best.pt")
        else:
            print("No valid cross-validation results obtained.")
    
    else:
        # Regular single training run
        print(f"Starting YOLOv12 LiDAR training with the following parameters:")
        print(f"- Model: {args.model} ({model_name})")
        print(f"- Data: {args.data}")
        print(f"- Epochs: {args.epochs}")
        print(f"- Batch size: {args.batch_size}")
        print(f"- Output directory: {output_dir}")
        print(f"- Save checkpoint period: every {args.save_period} epochs")
        print(f"- Metric for best model: {args.metric}")
        
        # Create temporary directory for dataset
        temp_dataset_dir = os.path.join(output_dir, "temp_dataset")
        os.makedirs(temp_dataset_dir, exist_ok=True)
        
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
            'names': data_config.get('names', ['pole']),
            # Add rectangular dimensions information
            'rect': True,
            'width': 1024,
            'height': 128
        }
        
        # Write the temp data config
        with open(temp_data_yaml_path, 'w') as f:
            yaml.dump(temp_data_config, f)
        
        print(f"Created temporary data configuration at {temp_data_yaml_path}")
        print(f"  Train path: {temp_train_dir}")
        print(f"  Val path: {temp_val_dir}")
        print(f"  Test path: {temp_test_dir}")
        
        # Load model
        print(f"\nLoading YOLOv12n model from: {args.model}")
        model = YOLO(args.model)
        
        # Train model with LiDAR-specific parameters
        try:
            print("\nStarting training...")
            # Use a tuple for imgsz and enable augmentations for better training
            results = model.train(
                    data=temp_data_yaml_path,
                    epochs=args.epochs,
                    batch=args.batch_size,
                    imgsz=1024,       # Changed from single value to tuple
                    rect=True,              # Changed from True to False
                    scale=0.5,               # Prevent poles from being scaled down too much
                    mosaic=0,                # Disable mosaic augmentation
                    mixup=0,                 # Disable mixup
                    copy_paste=0,            # Disable copy-paste
                    project=output_dir,
                    name=args.run_name,
                    exist_ok=True,
                    val=True,                # Enable validation during training
                    device=device,
                    save_period=args.save_period,
                    plots=True,
                    verbose=True,
                    degrees=4,               # Disable rotation
                    translate=0.1,           # Reduced from 0.2 to 0.1
                    fliplr=0.5,                # Disable horizontal flip
                    optimizer='SGD',
                    lr0=0.002,                # Changed from 0.002 to 0.01
                    lrf=0.01,
                    single_cls=True,
                    auto_augment=False,
                    cos_lr=True,
                    erasing=0,
                    amp=True,
                )
            
            print("Training completed!")
            print(f"Final model saved to: {output_dir}/{args.run_name}/weights/last.pt")
            print(f"Best model saved to: {output_dir}/{args.run_name}/weights/best.pt")
            print(f"\nTo validate the model, use: python src/validate_lidar_yolov12.py --model {output_dir}/{args.run_name}/weights/best.pt --data {args.data}")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main() 