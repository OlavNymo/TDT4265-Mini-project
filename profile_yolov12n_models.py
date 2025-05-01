import torch
from ultralytics import YOLO

# Paths to best model weights
lidar_best = 'results/lidar/yolov12n/lidar_0mosaic_0cp_0flr_75e/weights/best.pt'
rgb_best = 'results/rgb/yolov12n/rgb_75e_hue_0.005/weights/best.pt'

# Input size used for profiling (adjust if needed)
imgsz = 1024  # as used in your training args

def print_model_profile(model_path, name):
    print(f'\n===== {name} =====')
    model = YOLO(model_path)
    model.info(verbose=True)

if __name__ == '__main__':
    print_model_profile(lidar_best, 'LiDAR YOLOv12n')
    print_model_profile(rgb_best, 'RGB YOLOv12n') 