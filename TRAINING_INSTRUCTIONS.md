# Training and Validation Instructions

This document provides instructions for training and validating YOLOv12 models on RGB or LiDAR data for pole detection.

## Prerequisites

Make sure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

## Download Pretrained YOLOv12 Model

Before training, you need to download the pretrained YOLOv12n model:

```bash
python src/utils/download_models.py --model yolov12n
```

This will download the YOLOv12n model to the `results/models` directory as specified in the `base_config.yaml`.

## Training on RGB Data

To train the model on RGB data:

```bash
python src/train.py --config config/rgb_config.yaml --base_config config/base_config.yaml
```

Options:
- `--epochs <number>` - Override the number of epochs in the config
- `--batch_size <number>` - Override the batch size in the config
- `--seed <number>` - Override the random seed in the config

## Training on LiDAR Data

To train the model on LiDAR data:

```bash
python src/train.py --config config/lidar_config.yaml --base_config config/base_config.yaml
```

## Validation

To validate a trained model:

```bash
python src/evaluate.py --config config/rgb_config.yaml --model_path <path_to_model>
```

Where `<path_to_model>` is the path to the trained model.

## Architecture Overview

The training process:

1. Loads base and specific configurations
2. Creates a dataset-specific experiment directory
3. Initializes a YOLOv12n model with pretrained weights
4. Applies YOLOv12-specific training parameters
5. Trains the model on the specified dataset
6. Evaluates the model performance

## YOLOv12 Parameters

The YOLOv12n model uses these recommended parameters:
- `scale`: 0.5
- `mosaic`: 1.0
- `mixup`: 0.0
- `copy_paste`: 0.1

For other YOLOv12 model variants (s, m, l, x), different parameters are used as specified in the YOLOv12 documentation.

## Troubleshooting

If you encounter the error "Pretrained model not found", make sure you've downloaded the model using the download script provided above. 