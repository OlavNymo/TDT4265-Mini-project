"""
YOLO model implementation for snow pole detection.
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import os


class YOLOModel:
    """Wrapper class for YOLO model used for snow pole detection."""
    
    def __init__(self, config):
        """
        Initialize YOLO model with configuration.
        
        Args:
            config (dict): Model configuration
        """
        self.config = config
        self.model_name = config['model']['name']
        self.model_path = Path(config['paths']['model_dir'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Handle YOLOv12 models
        if 'yolov12n' in self.model_name:
            pretrained_path = self.model_path / f"{self.model_name}.pt"
            
            if config['model']['pretrained']:
                if not pretrained_path.exists():
                    print(f"Pretrained model {pretrained_path} not found.")
                    print(f"Please download the YOLOv12 model from:")
                    print(f"https://github.com/sunsmarterjie/yolov12/releases/download/turbo/{self.model_name}.pt")
                    print(f"And place it in {pretrained_path}")
                    raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
                
                self.model = YOLO(str(pretrained_path))
                print(f"Loaded pretrained {self.model_name} model from {pretrained_path}")
            else:
                # If not pretrained, start from scratch with the specified architecture
                model_yaml = f"{self.model_name}.yaml"  # YOLOv12 model YAML
                self.model = YOLO(model_yaml)
                print(f"Initialized {self.model_name} model from yaml: {model_yaml}")
        else:
            # Default behavior for other YOLO models
            if config['model']['pretrained']:
                self.model = YOLO(self.model_name)
            else:
                # If not pretrained, start from scratch with the specified architecture
                self.model = YOLO(self.model_name)
            
        print(f"Model running on {self.device}")
        
    def train(self, data_config, output_dir, epochs=None, batch_size=None):
        """
        Train the YOLO model.
        
        Args:
            data_config (str or dict): Path to data YAML or data configuration
            output_dir (str): Directory to save results
            epochs (int, optional): Number of epochs to train
            batch_size (int, optional): Batch size for training
            
        Returns:
            dict: Training results
        """
        # Get training parameters from config if not specified
        if epochs is None:
            epochs = self.config['training']['epochs']
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
            
        # Set up training parameters
        train_params = {
            'data': data_config,
            'epochs': epochs,
            'batch': batch_size,
            'save': True,
            'save_period': self.config['training'].get('save_checkpoint_interval', -1),
            'patience': self.config['training'].get('early_stopping_patience', 50),
            'device': self.device,
            'workers': self.config['training'].get('workers', 4),
            'project': output_dir
        }
        
        # Handle image size - check if we're using rectangular format for LiDAR
        if 'input_width' in self.config['model'] and 'input_height' in self.config['model']:
            # Use the larger dimension as the imgsz and enable rect=True for rectangular training
            train_params['imgsz'] = max(self.config['model']['input_width'], self.config['model']['input_height'])
            train_params['rect'] = True  # Enable rectangular training
            print(f"Using rectangular training with imgsz={train_params['imgsz']} and rect=True")
        else:
            # Use square dimensions
            train_params['imgsz'] = self.config['model']['input_size']
            print(f"Using square image size: {train_params['imgsz']}")
        
        # Add YOLOv12-specific training parameters if applicable
        if 'yolov12n' in self.model_name:
            # Add recommended YOLOv12 parameters based on model size
            if 'yolov12n' in self.model_name:
                train_params.update({
                    'scale': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.0,
                    'copy_paste': 0.1,
                })
            elif 'yolov12s' in self.model_name:
                train_params.update({
                    'scale': 0.9,
                    'mosaic': 1.0,
                    'mixup': 0.05,
                    'copy_paste': 0.15,
                })
            elif 'yolov12m' in self.model_name:
                train_params.update({
                    'scale': 0.9,
                    'mosaic': 1.0,
                    'mixup': 0.15,
                    'copy_paste': 0.4,
                })
            elif 'yolov12l' in self.model_name:
                train_params.update({
                    'scale': 0.9,
                    'mosaic': 1.0,
                    'mixup': 0.15,
                    'copy_paste': 0.5,
                })
            elif 'yolov12x' in self.model_name:
                train_params.update({
                    'scale': 0.9,
                    'mosaic': 1.0,
                    'mixup': 0.2,
                    'copy_paste': 0.6,
                })
        
        # Train model
        results = self.model.train(**train_params)
        return results
    
    def evaluate(self, data_loader, conf_thres=None, iou_thres=None):
        """
        Evaluate the YOLO model.
        
        Args:
            data_loader: DataLoader for evaluation
            conf_thres (float, optional): Confidence threshold
            iou_thres (float, optional): IoU threshold
            
        Returns:
            dict: Evaluation results
        """
        # Get evaluation parameters from config if not specified
        if conf_thres is None:
            conf_thres = self.config['evaluation'].get('conf_threshold', 0.001)
        if iou_thres is None:
            iou_thres = self.config['evaluation'].get('iou_threshold', 0.5)
            
        # Set up evaluation parameters
        eval_params = {
            'conf': conf_thres,
            'iou': iou_thres,
            'device': self.device
        }
        
        # Evaluate model on validation data
        results = self.model.val(**eval_params)
        return results
    
    def predict(self, image, conf_thres=None, iou_thres=None):
        """
        Run inference on an image.
        
        Args:
            image: Image to run inference on
            conf_thres (float, optional): Confidence threshold
            iou_thres (float, optional): IoU threshold
            
        Returns:
            list: Prediction results
        """
        # Get prediction parameters from config if not specified
        if conf_thres is None:
            conf_thres = self.config['model'].get('conf_threshold', 0.25)
        if iou_thres is None:
            iou_thres = self.config['model'].get('iou_threshold', 0.45)
            
        # Set up prediction parameters
        pred_params = {
            'conf': conf_thres,
            'iou': iou_thres,
            'device': self.device
        }
        
        # Handle image size - check if we're using rectangular format for LiDAR
        if 'input_width' in self.config['model'] and 'input_height' in self.config['model']:
            # Use the larger dimension as the imgsz and enable rect=True for rectangular inference
            pred_params['imgsz'] = max(self.config['model']['input_width'], self.config['model']['input_height'])
            pred_params['rect'] = True  # Enable rectangular inference
        else:
            # Use square dimensions
            pred_params['imgsz'] = self.config['model']['input_size']
        
        # Run prediction
        results = self.model.predict(image, **pred_params)
        return results
    
    def save(self, path):
        """
        Save model to file.
        
        Args:
            path (str): Path to save model
        """
        self.model.save(path)
        
    def load(self, path):
        """
        Load model from file.
        
        Args:
            path (str): Path to load model from
        """
        self.model = YOLO(path)
        

def create_data_yaml(config, output_path):
    """
    Create YAML file for YOLO training configuration.
    
    Args:
        config (dict): Configuration dictionary
        output_path (str): Path to save YAML file
        
    Returns:
        str: Path to created YAML file
    """
    dataset_config = config['dataset']
    
    # Create data config
    data_yaml = {
        'train': dataset_config['train_path'],
        'val': dataset_config['val_path'],
        'test': dataset_config['test_path'],
        'nc': dataset_config['num_classes'],
        'names': dataset_config['classes']
    }
    
    # Save to file
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
        
    return output_path 