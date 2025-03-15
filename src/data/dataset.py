"""
Dataset modules for snow pole detection project.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SnowPoleDataset(Dataset):
    """Base dataset class for snow pole detection."""
    
    def __init__(self, img_dir, label_dir, transform=None, img_size=640):
        """
        Initialize the snow pole dataset.
        
        Args:
            img_dir (str): Directory containing images
            label_dir (str): Directory containing labels
            transform (albumentations.Compose): Transformations to apply
            img_size (int): Image size for resizing
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        self.img_files = sorted([f for f in os.listdir(self.img_dir) 
                               if f.endswith(('.jpg', '.jpeg', '.png', '.PNG'))])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Dictionary containing image, bboxes, and labels
        """
        img_file = self.img_files[idx]
        img_path = self.img_dir / img_file
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get label file path (YOLO format)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = self.label_dir / label_file
        
        # Load labels
        bboxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])  # Assuming 0 for 'pole'
                    # YOLO format: class_id, x_center, y_center, width, height (normalized)
                    x_center, y_center, width, height = map(float, data[1:5])
                    
                    # Convert to albumentations format: xmin, ymin, xmax, ymax (normalized)
                    xmin = x_center - width/2
                    ymin = y_center - height/2
                    xmax = x_center + width/2
                    ymax = y_center + height/2
                    
                    bboxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)
                    
        # Convert to numpy arrays
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transformations
        if self.transform and len(bboxes) > 0:
            transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        elif self.transform:
            transformed = self.transform(image=img, bboxes=[], labels=[])
            img = transformed['image']
            bboxes = []
            labels = []
            
        if len(bboxes) == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int64)
        else:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            
        # Convert to tensor format expected by YOLO
        targets = torch.zeros((len(bboxes), 6))
        if len(bboxes) > 0:
            # Format: batch_idx, class, x_center, y_center, width, height
            targets[:, 1] = torch.tensor(labels)
            targets[:, 2:] = torch.tensor(self._convert_to_yolo_format(bboxes))
        
        return {
            'image': img,
            'targets': targets,
            'image_path': str(img_path)
        }
    
    def _convert_to_yolo_format(self, bboxes):
        """
        Convert bboxes from [xmin, ymin, xmax, ymax] to YOLO [x_center, y_center, width, height].
        
        Args:
            bboxes (np.ndarray): Bounding boxes in [xmin, ymin, xmax, ymax] format
            
        Returns:
            np.ndarray: Bounding boxes in YOLO [x_center, y_center, width, height] format
        """
        yolo_bboxes = np.zeros_like(bboxes)
        yolo_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x_center
        yolo_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y_center
        yolo_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]        # width
        yolo_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]        # height
        
        return yolo_bboxes


class LiDARDataset(SnowPoleDataset):
    """Dataset for LiDAR snow pole detection."""
    
    def __init__(self, img_dir, label_dir, transform=None, img_size=640, img_width=1024, img_height=128):
        """
        Initialize the LiDAR dataset.
        
        Args:
            img_dir (str): Directory containing LiDAR images
            label_dir (str): Directory containing labels
            transform (albumentations.Compose): Transformations to apply
            img_size (int): Legacy image size parameter (for square images)
            img_width (int): Image width for resizing (default: 1024 as per paper)
            img_height (int): Image height for resizing (default: 128 as per paper)
        """
        # Call parent constructor but we'll override the transform later if needed
        super().__init__(img_dir, label_dir, transform, img_size)
        self.img_width = img_width
        self.img_height = img_height


class RGBDataset(SnowPoleDataset):
    """Dataset for RGB snow pole detection."""
    
    def __init__(self, img_dir, label_dir, transform=None, img_size=640):
        """
        Initialize the RGB dataset.
        
        Args:
            img_dir (str): Directory containing RGB images
            label_dir (str): Directory containing labels
            transform (albumentations.Compose): Transformations to apply
            img_size (int): Image size for resizing
        """
        super().__init__(img_dir, label_dir, transform, img_size)


def get_transformations(config, is_train=True):
    """
    Get data transformations based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        is_train (bool): Whether in training mode (apply augmentations)
        
    Returns:
        albumentations.Compose: Composition of transformations
    """
    # Check if we're using rectangular format for LiDAR
    is_lidar = config['dataset']['name'] == 'lidar'
    
    # Use rectangular dimensions for LiDAR if specified
    if is_lidar and 'input_width' in config['model'] and 'input_height' in config['model']:
        img_width = config['model']['input_width']
        img_height = config['model']['input_height']
        is_rectangular = True
    else:
        img_size = config['model']['input_size']
        img_width = img_size
        img_height = img_size
        is_rectangular = False
    
    # Note: When using rect=True with YOLOv12, the model will handle aspect ratio internally
    # We still need to resize to the correct dimensions in our dataset for consistency
    
    if is_train:
        # Training transformations with augmentations
        if is_rectangular:
            # For rectangular images (LiDAR)
            transforms = A.Compose([
                A.Resize(height=img_height, width=img_width),  # Resize to rectangular format
                A.HorizontalFlip(p=config['augmentation'].get('fliplr', 0.5)),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=config['augmentation'].get('hsv_h', 0.015) * 180,
                    sat_shift_limit=config['augmentation'].get('hsv_s', 0.7) * 255,
                    val_shift_limit=config['augmentation'].get('hsv_v', 0.4) * 255,
                    p=0.5
                ),
                A.ToGray(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            # For square images (RGB)
            transforms = A.Compose([
                A.RandomResizedCrop(height=img_height, width=img_width, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=config['augmentation'].get('fliplr', 0.5)),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=config['augmentation'].get('hsv_h', 0.015) * 180,
                    sat_shift_limit=config['augmentation'].get('hsv_s', 0.7) * 255,
                    val_shift_limit=config['augmentation'].get('hsv_v', 0.4) * 255,
                    p=0.5
                ),
                A.ToGray(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        # Validation/test transformations (no augmentations)
        transforms = A.Compose([
            A.Resize(height=img_height, width=img_width),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    return transforms


def get_dataset(config, split='train'):
    """
    Get the appropriate dataset based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        split (str): Dataset split ('train', 'val', or 'test')
        
    Returns:
        torch.utils.data.Dataset: Dataset for the specified split
    """
    dataset_config = config['dataset']
    is_train = split == 'train'
    
    # Get appropriate paths
    if split == 'train':
        img_dir = dataset_config['train_path']
    elif split == 'val':
        img_dir = dataset_config['val_path']
    elif split == 'test':
        img_dir = dataset_config['test_path']
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    label_dir = f"{dataset_config['label_path']}/{split}"
    
    # Get transformations
    transforms = get_transformations(config, is_train=is_train)
    
    # Create dataset based on type
    if dataset_config['name'] == 'lidar':
        # For LiDAR, check if we have rectangular dimensions
        if 'input_width' in config['model'] and 'input_height' in config['model']:
            return LiDARDataset(
                img_dir, 
                label_dir, 
                transforms, 
                config['model']['input_size'],
                config['model']['input_width'],
                config['model']['input_height']
            )
        else:
            return LiDARDataset(img_dir, label_dir, transforms, config['model']['input_size'])
    elif dataset_config['name'] == 'rgb':
        return RGBDataset(img_dir, label_dir, transforms, config['model']['input_size'])
    else:
        raise ValueError(f"Invalid dataset type: {dataset_config['name']}. Must be 'lidar' or 'rgb'")


def create_data_loaders(config):
    """
    Create data loaders for all splits.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary containing data loaders for train, val, and test splits
    """
    # Get datasets
    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'val')
    test_dataset = get_dataset(config, 'test')
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['workers'],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['training']['workers'],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['training']['workers'],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def collate_fn(batch):
    """
    Custom collate function for YOLO detection batches.
    
    Args:
        batch (list): List of samples from the dataset
        
    Returns:
        tuple: Tuple of (images, targets)
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Combine targets with batch index
    targets = []
    for batch_idx, item in enumerate(batch):
        if len(item['targets']) > 0:
            item_targets = item['targets'].clone()
            item_targets[:, 0] = batch_idx  # Set batch index
            targets.append(item_targets)
    
    # Combine all targets into a single tensor if there are any
    if len(targets) > 0:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros((0, 6))
        
    paths = [item['image_path'] for item in batch]
    
    return images, targets, paths 