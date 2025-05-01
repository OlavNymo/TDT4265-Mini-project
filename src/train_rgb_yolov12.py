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
import numpy as np
import shutil
import yaml
from sklearn.model_selection import KFold

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
    parser.add_argument('--use_tiling', action='store_true',
                        help='Enable image tiling for improved small object detection')
    parser.add_argument('--tile_size', type=int, default=640,
                        help='Size of image tiles when tiling is enabled')
    parser.add_argument('--tile_overlap', type=float, default=0.2,
                        help='Overlap between tiles as a fraction of tile size')
    parser.add_argument('--tile_mix_ratio', type=float, default=0.5,
                        help='Ratio of tiled to complete images (0.5 means 50/50 mix)')
    return parser.parse_args()

def prepare_cv_data(data_path, fold_idx, n_folds, output_dir, use_symlinks=True, max_samples=0):
    """
    Prepare cross-validation data split.
    
    Args:
        data_path: Path to original data.yaml
        fold_idx: Current fold index
        n_folds: Total number of folds
        output_dir: Directory to save new data configuration
        use_symlinks: Whether to use symlinks (True) or copy files (False)
        max_samples: Maximum number of samples to use (0 for all)
        
    Returns:
        Path to the new data.yaml for this fold
    """
    # Load original data configuration
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get train and validation paths
    train_path = data_config.get('train', '')
    val_path = data_config.get('val', '')
    
    print(f"Original train path: {train_path}")
    print(f"Original val path: {val_path}")
    
    # Determine label paths based on YOLO directory structure
    train_label_dir = os.path.join(os.path.dirname(os.path.dirname(train_path)), 'labels', 'train')
    val_label_dir = os.path.join(os.path.dirname(os.path.dirname(val_path)), 'labels', 'valid')
    
    print(f"Train label path: {train_label_dir}")
    print(f"Val label path: {val_label_dir}")
    
    # Combine train and validation datasets for CV split
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    
    # Get all image files from training directory (case-insensitive)
    train_files = []
    for f in os.listdir(train_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            train_files.append(f)
    print(f"Found {len(train_files)} training images")
    
    # If validation directory exists, include those files too
    val_files = []
    if os.path.exists(val_path):
        for f in os.listdir(val_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                val_files.append(f)
        all_files = train_files + val_files
        print(f"Found {len(val_files)} validation images")
    else:
        all_files = train_files
    
    # Limit samples if requested
    if max_samples > 0 and max_samples < len(all_files):
        print(f"Limiting to {max_samples} samples from original {len(all_files)} images")
        np.random.seed(42)
        all_files = np.random.choice(all_files, max_samples, replace=False).tolist()
    
    print(f"Total images for CV: {len(all_files)}")
    
    if len(all_files) == 0:
        raise ValueError(f"No images found in {train_path} or {val_path}. Check paths in data.yaml.")
    
    # Shuffle files with fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    # Create KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Get indices for current fold
    all_indices = np.arange(len(all_files))
    train_indices, val_indices = list(kf.split(all_indices))[fold_idx]
    
    # Create directories for this fold
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    fold_train_dir = os.path.join(fold_dir, "images", "train")
    fold_val_dir = os.path.join(fold_dir, "images", "val")
    fold_train_label_dir = os.path.join(fold_dir, "labels", "train")
    fold_val_label_dir = os.path.join(fold_dir, "labels", "val")
    
    # Create all necessary directories
    os.makedirs(fold_train_dir, exist_ok=True)
    os.makedirs(fold_val_dir, exist_ok=True)
    os.makedirs(fold_train_label_dir, exist_ok=True)
    os.makedirs(fold_val_label_dir, exist_ok=True)
    
    # Check for data leakage to ensure no overlap between train and val
    train_indices_set = set(train_indices)
    val_indices_set = set(val_indices)
    overlap = train_indices_set.intersection(val_indices_set)
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping indices between train and validation sets!")
    else:
        print("✓ No data leakage detected between train and validation sets")
    
    # Create symlinks or copy files for training and validation sets
    for idx in train_indices:
        file_name = all_files[idx]
        base_name = os.path.splitext(file_name)[0]
        
        # Handle image file
        if file_name in train_files:
            src_path = os.path.join(train_path, file_name)
            src_label_path = os.path.join(train_label_dir, f"{base_name}.txt")
        else:
            src_path = os.path.join(val_path, file_name)
            src_label_path = os.path.join(val_label_dir, f"{base_name}.txt")
        
        # Create symlink for image (always use symlinks for images)
        dst_path = os.path.join(fold_train_dir, file_name)
        if not os.path.exists(dst_path):
            try:
                os.symlink(src_path, dst_path)
            except Exception as e:
                print(f"Error creating symlink for {file_name}: {e}")
                continue  # Skip this file if we can't create symlink
        
        # Handle label file if it exists
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
    
    # Create symlinks or copy files for validation set
    for idx in val_indices:
        file_name = all_files[idx]
        base_name = os.path.splitext(file_name)[0]
        
        # Handle image file
        if file_name in train_files:
            src_path = os.path.join(train_path, file_name)
            src_label_path = os.path.join(train_label_dir, f"{base_name}.txt")
        else:
            src_path = os.path.join(val_path, file_name)
            src_label_path = os.path.join(val_label_dir, f"{base_name}.txt")
        
        # Create symlink for image (always use symlinks for images)
        dst_path = os.path.join(fold_val_dir, file_name)
        if not os.path.exists(dst_path):
            try:
                os.symlink(src_path, dst_path)
            except Exception as e:
                print(f"Error creating symlink for {file_name}: {e}")
                continue  # Skip this file if we can't create symlink
        
        # Handle label file if it exists
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
    
    # Count actual labels that were found and copied/linked
    train_labels_count = len(os.listdir(fold_train_label_dir))
    val_labels_count = len(os.listdir(fold_val_label_dir))
    
    print(f"Created fold {fold_idx+1}/{n_folds} with:")
    print(f"  - {len(train_indices)} training images and {train_labels_count} labels")
    print(f"  - {len(val_indices)} validation images and {val_labels_count} labels")
    
    # Get absolute paths for train and val directories
    abs_fold_dir = os.path.abspath(fold_dir)
    
    # Create new data.yaml for this fold with proper YOLO structure
    fold_data_config = data_config.copy()
    fold_data_config['path'] = abs_fold_dir
    fold_data_config['train'] = os.path.join('images', 'train')  # Relative to path
    fold_data_config['val'] = os.path.join('images', 'val')      # Relative to path
    
    fold_data_path = os.path.join(fold_dir, "data.yaml")
    with open(fold_data_path, 'w') as f:
        yaml.dump(fold_data_config, f)
    
    print(f"Prepared fold {fold_idx+1}/{n_folds} with YOLO-compatible directory structure")
    return fold_data_path

def prepare_hybrid_dataset(original_data_path, output_dir, tile_size=640, overlap=0.2, mix_ratio=0.5):
    """
    Create a hybrid dataset combining both full images and tiled images.
    
    Args:
        original_data_path: Path to original data.yaml
        output_dir: Directory to save the hybrid dataset
        tile_size: Size of the tiles (square tiles)
        overlap: Overlap between tiles as a fraction of tile size
        mix_ratio: Ratio of tiled to complete images (0.5 means 50/50 mix)
        
    Returns:
        Path to the new data.yaml for the hybrid dataset
    """
    import cv2
    
    # Load original data configuration
    with open(original_data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get train and validation paths
    train_path = data_config.get('train', '')
    val_path = data_config.get('val', '')
    
    # Compute actual pixel overlap
    overlap_px = int(tile_size * overlap)
    stride = tile_size - overlap_px
    
    print(f"Creating hybrid dataset with {mix_ratio*100:.0f}% tiled images")
    print(f"Tile settings: size {tile_size}px, overlap {overlap_px}px")
    
    # Create hybrid dataset directory structure
    hybrid_dir = os.path.join(output_dir, f"hybrid_{tile_size}")
    hybrid_train_img_dir = os.path.join(hybrid_dir, "images", "train")
    hybrid_val_img_dir = os.path.join(hybrid_dir, "images", "val")
    hybrid_train_label_dir = os.path.join(hybrid_dir, "labels", "train")
    hybrid_val_label_dir = os.path.join(hybrid_dir, "labels", "val")
    
    # Create directories
    for d in [hybrid_train_img_dir, hybrid_val_img_dir, hybrid_train_label_dir, hybrid_val_label_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Get paths to label directories
    train_label_dir = os.path.join(os.path.dirname(os.path.dirname(train_path)), 'labels', 'train')
    val_label_dir = os.path.join(os.path.dirname(os.path.dirname(val_path)), 'labels', 'valid')
    
    # First, copy all original images and labels (this ensures we keep full context)
    # Process training images
    if os.path.exists(train_path):
        copy_original_images(train_path, train_label_dir, hybrid_train_img_dir, hybrid_train_label_dir, "train")
    
    # Process validation images
    if os.path.exists(val_path):
        copy_original_images(val_path, val_label_dir, hybrid_val_img_dir, hybrid_val_label_dir, "val")
    
    # Now add tiled versions of a subset of images
    if os.path.exists(train_path):
        process_directory_for_hybrid(train_path, train_label_dir, hybrid_train_img_dir, hybrid_train_label_dir, 
                                     tile_size, stride, "train", mix_ratio)
    
    # Process validation images
    if os.path.exists(val_path):
        process_directory_for_hybrid(val_path, val_label_dir, hybrid_val_img_dir, hybrid_val_label_dir, 
                                     tile_size, stride, "val", mix_ratio)
    
    # Create new data.yaml for the hybrid dataset
    hybrid_data_config = data_config.copy()
    hybrid_data_config['path'] = os.path.abspath(hybrid_dir)
    hybrid_data_config['train'] = os.path.join('images', 'train')
    hybrid_data_config['val'] = os.path.join('images', 'val')
    hybrid_data_config['names'] = data_config.get('names', ['pole'])
    
    hybrid_data_path = os.path.join(hybrid_dir, "data.yaml")
    with open(hybrid_data_path, 'w') as f:
        yaml.dump(hybrid_data_config, f)
    
    return hybrid_data_path

def copy_original_images(img_dir, label_dir, target_img_dir, target_label_dir, split_name):
    """Copy all original images and labels to the target directory."""
    import cv2
    
    copied_count = 0
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Skip if label doesn't exist
        if not os.path.exists(label_path):
            print(f"Warning: No label found for {img_file}, skipping")
            continue
        
        # Copy image - using full copy to avoid symlink issues
        dst_img_path = os.path.join(target_img_dir, img_file)
        if not os.path.exists(dst_img_path):
            shutil.copy2(img_path, dst_img_path)
        
        # Copy label
        dst_label_path = os.path.join(target_label_dir, f"{base_name}.txt")
        if not os.path.exists(dst_label_path):
            shutil.copy2(label_path, dst_label_path)
        
        copied_count += 1
    
    print(f"Copied {copied_count} original images and labels to hybrid {split_name} set")
    return copied_count

def process_directory_for_hybrid(img_dir, label_dir, hybrid_img_dir, hybrid_label_dir, 
                                 tile_size, stride, split_name, mix_ratio):
    """
    Process images to add tiled versions to the hybrid dataset.
    Only process a subset of images based on the mix_ratio.
    """
    import cv2
    
    # Get list of all valid images
    valid_images = []
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Skip if label doesn't exist
        if not os.path.exists(label_path):
            continue
        
        # Check if image can be read
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        valid_images.append((img_file, img_path, label_path))
    
    # Determine how many images to tile based on mix_ratio
    # We already have 100% of original images, so this is additional
    # For mix_ratio=0.5, we'd process half the images for tiling
    num_to_tile = int(len(valid_images) * mix_ratio)
    
    # Randomly select images to tile
    np.random.seed(42)  # For reproducibility
    indices_to_tile = np.random.choice(len(valid_images), num_to_tile, replace=False)
    
    processed_count = 0
    tiled_count = 0
    
    for idx in indices_to_tile:
        img_file, img_path, label_path = valid_images[idx]
        base_name = os.path.splitext(img_file)[0]
        
        # Read image
        img = cv2.imread(img_path)
        
        # Read labels (YOLO format: class x_center y_center width height)
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines() if line.strip()]
        
        # Convert labels to absolute format
        h, w = img.shape[:2]
        abs_labels = []
        for label in labels:
            cls_id = int(label[0])
            x_center = float(label[1]) * w
            y_center = float(label[2]) * h
            width = float(label[3]) * w
            height = float(label[4]) * h
            abs_labels.append([cls_id, x_center, y_center, width, height])
        
        # Skip tiling for small images that fit within a single tile
        if h <= tile_size and w <= tile_size:
            continue  # Already have the original image
        
        # Create tiles
        tile_idx = 0
        for y in range(0, h - tile_size + stride, stride):
            for x in range(0, w - tile_size + stride, stride):
                # Make sure we don't exceed image dimensions
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)
                
                # Extract tile
                tile = img[y_start:y_end, x_start:x_end]
                
                # Filter and adjust bounding boxes for this tile
                tile_labels = []
                for cls_id, x_center, y_center, width, height in abs_labels:
                    # Check if bounding box center is within tile
                    if (x_start <= x_center < x_end and 
                        y_start <= y_center < y_end):
                        
                        # Adjust coordinates relative to tile
                        x_rel = (x_center - x_start) / (x_end - x_start)
                        y_rel = (y_center - y_start) / (y_end - y_start)
                        
                        # Adjust width and height relative to tile
                        w_rel = min(width / (x_end - x_start), 1.0)
                        h_rel = min(height / (y_end - y_start), 1.0)
                        
                        # Ensure relative coordinates are within [0, 1]
                        x_rel = max(0, min(1, x_rel))
                        y_rel = max(0, min(1, y_rel))
                        w_rel = max(0, min(1, w_rel))
                        h_rel = max(0, min(1, h_rel))
                        
                        tile_labels.append([cls_id, x_rel, y_rel, w_rel, h_rel])
                
                # Only save tiles that contain objects
                if tile_labels:
                    tile_filename = f"{base_name}_tile{tile_idx}.jpg"
                    tile_label_filename = f"{base_name}_tile{tile_idx}.txt"
                    
                    # Save tile image
                    cv2.imwrite(os.path.join(hybrid_img_dir, tile_filename), tile)
                    
                    # Save tile labels
                    with open(os.path.join(hybrid_label_dir, tile_label_filename), 'w') as f:
                        for label in tile_labels:
                            f.write(' '.join(map(str, label)) + '\n')
                    
                    tile_idx += 1
                    tiled_count += 1
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{num_to_tile} images for tiling, created {tiled_count} tiles in {split_name} set")
    
    print(f"Finished creating hybrid {split_name} set: added {tiled_count} tiles from {processed_count} images")
    return processed_count, tiled_count

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
            args.output_dir = 'results/rgb/yolov12n'
        elif 'yolov12s' in model_name:
            args.output_dir = 'results/rgb/yolov12s'
        else:
            # Default to yolov12n folder for unknown models
            args.output_dir = 'results/rgb/yolov12n'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Apply tiling if requested
    data_path = args.data
    if args.use_tiling:
        print(f"Creating hybrid dataset with {args.tile_mix_ratio*100:.0f}% tiled images")
        data_path = prepare_hybrid_dataset(args.data, args.output_dir, args.tile_size, 
                                          args.tile_overlap, args.tile_mix_ratio)
        print(f"Created hybrid dataset at: {data_path}")
    
    # Cross-validation
    if args.cv > 1:
        print(f"Running {args.cv}-fold cross-validation")
        
        # Dictionary to store results for each fold
        cv_results = {}
        best_model_path = None
        best_metric_value = 0
        
        for fold in range(args.cv):
            print(f"\n{'='*20} Fold {fold+1}/{args.cv} {'='*20}")
            
            # Prepare data for this fold
            fold_data_path = prepare_cv_data(data_path, fold, args.cv, output_dir, args.no_symlinks, args.max_samples)
            fold_run_name = f"{args.run_name}_fold{fold+1}"
            
            # Load model for this fold
            model = YOLO(args.model)
            
            # Train model with recommended YOLOv12n parameters
            results = model.train(
                data=fold_data_path,
                epochs=args.epochs,
                batch=args.batch_size,
                imgsz=(640, 1024),  # Height, width - maintains better aspect ratio
                rect=True,  # Enable rectangular training
                scale=0.5,           # YOLOv12n recommended value
                mosaic=0.9,          # Slightly reduced from 1.0 to avoid overfitting
                mixup=0.1,           # Add mixup for better generalization
                copy_paste=0.1,
                project=args.output_dir,
                name=fold_run_name,
                exist_ok=True,
                val=True,
                device=device,
                save_period=args.save_period,
                amp=False,
                plots=True,
                verbose=True,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0,
                translate=0.1,
                fliplr=0.5,          # Enable horizontal flipping
                optimizer='SGD',
                lr0=0.001,           # Slightly lower learning rate
                lrf=0.01,
                cos_lr=True,         # Use cosine annealing scheduler
                iou=0.7,             # IoU training threshold for DIoU loss
            )
            
            # Get final metric values
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
            fold_model_path = os.path.join(args.output_dir, fold_run_name, 'weights', 'best.pt')
            if metric_value > best_metric_value and os.path.exists(fold_model_path):
                best_metric_value = metric_value
                best_model_path = fold_model_path
                print(f"New best model found in fold {fold+1}")
        
        # Calculate and display CV results
        mean_metric = np.mean(list(cv_results.values()))
        std_metric = np.std(list(cv_results.values()))
        print(f"\nCross-validation results:")
        print(f"Mean {metric_name}: {mean_metric:.4f} ± {std_metric:.4f}")
        
        # Copy best model to final location
        if best_model_path:
            final_model_path = os.path.join(args.output_dir, args.run_name, 'weights')
            os.makedirs(final_model_path, exist_ok=True)
            shutil.copy2(best_model_path, os.path.join(final_model_path, 'best.pt'))
            print(f"Best model (from fold with {metric_name}={best_metric_value:.4f}) copied to: {final_model_path}/best.pt")
        
    else:
        # Regular single training run without CV
        print(f"Starting YOLOv12 training with the following parameters:")
        print(f"- Model: {args.model} ({model_name})")
        print(f"- Data: {data_path}")
        print(f"- Epochs: {args.epochs}")
        print(f"- Batch size: {args.batch_size}")
        print(f"- Output directory: {args.output_dir}")
        print(f"- Save checkpoint period: every {args.save_period} epochs")
        print(f"- Metric for best model: {args.metric}")
        if args.use_tiling:
            print(f"- Using hybrid dataset with tile size: {args.tile_size}px, overlap: {args.tile_overlap}, mix ratio: {args.tile_mix_ratio}")
        
        # Load model
        model = YOLO(args.model)
        
        # Train model with recommended YOLOv12n parameters
        results = model.train(
            data=data_path,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=(640, 1024),  # Height, width - maintains better aspect ratio
            rect=True,  # Enable rectangular training
            scale=0.5,           # YOLOv12n recommended value
            mosaic=0,          # Slightly reduced from 1.0 to avoid overfitting
            project=args.output_dir,
            name=args.run_name,
            exist_ok=True,
            val=True,
            device=device,
            save_period=args.save_period,
            amp=True,
            plots=True,
            verbose=True,
            translate=0,
            hsv_h=0.015,
            hsv_s=1,
            hsv_v=0.4,
            degrees=0,
            fliplr=0.5,          # Enable horizontal flipping
            optimizer='SGD',
            lr0=0.005,           # Slightly lower learning rate
            lrf=0.01,
            single_cls=True,
            erasing=0.4,
            copy_paste=0.1,
        )
    
    print("Training completed!")
    print(f"Final model saved to: {args.output_dir}/{args.run_name}/weights/last.pt")
    print(f"Best model saved to: {args.output_dir}/{args.run_name}/weights/best.pt")
    print(f"\nTo validate the model, use: python src/validate_rgb_yolov12.py --model {args.output_dir}/{args.run_name}/weights/best.pt --data {args.data}")

if __name__ == '__main__':
    main() 