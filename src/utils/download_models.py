#!/usr/bin/env python
"""
Script to download pretrained YOLOv12 models.
"""
import os
import sys
import argparse
import urllib.request
import shutil
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.utils import load_config, merge_configs


def download_file(url, destination):
    """
    Download a file from a URL to a destination path.
    
    Args:
        url (str): URL to download from
        destination (str or Path): Destination path
    """
    destination = Path(destination)
    
    # Create directory if it doesn't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the file
    print(f"Downloading {url} to {destination}...")
    
    try:
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            file_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            
            if file_size > 0:
                # Show progress
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    # Display progress
                    percent = int(downloaded * 100 / file_size)
                    if percent % 10 == 0:
                        print(f"Downloaded: {percent}% ({downloaded/1024/1024:.1f} MB)")
            else:
                # No content length header
                shutil.copyfileobj(response, out_file)
        
        print(f"Downloaded {destination}")
        return True
    except urllib.error.HTTPError as e:
        print(f"Error downloading file: {e}")
        print(f"HTTP Error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def download_yolov12_model(model_name="yolov12n", destination_dir=None):
    """
    Download a YOLOv12 model from the GitHub repository.
    
    Args:
        model_name (str): Model name (yolov12n, yolov12s, yolov12m, yolov12l, yolov12x)
        destination_dir (str or Path, optional): Destination directory
    
    Returns:
        Path: Path to downloaded model
    """
    # Load base config to get model directory
    if destination_dir is None:
        base_config = load_config("config/base_config.yaml")
        destination_dir = Path(base_config['paths']['model_dir'])
    else:
        destination_dir = Path(destination_dir)
    
    # Create directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model URL
    base_url = "https://github.com/sunsmarterjie/yolov12/releases/download/turbo"
    model_url = f"{base_url}/{model_name}.pt"
    
    # Define destination path
    destination_path = destination_dir / f"{model_name}.pt"
    
    # Download model
    download_file(model_url, destination_path)
    
    return destination_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download pretrained YOLOv12 models')
    parser.add_argument('--model', type=str, default='yolov12n',
                       help='Model name (yolov12n, yolov12s, yolov12m, yolov12l, yolov12x)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for downloaded models')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Download model
    download_yolov12_model(args.model, args.output) 