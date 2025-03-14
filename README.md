# Snow Pole Detection for Autonomous Driving

This project implements an object detection system for snow poles to improve autonomous driving capabilities in winter conditions. Using YOLO models trained on both LiDAR and natural image datasets, the system enables reliable road edge detection in snowy environments.

## Project Overview

In winter conditions, autonomous vehicles struggle with standard road detection methods. Snow poles erected along road edges provide reliable reference points. This project focuses on real-time detection of these poles using:

1. **LiDAR dataset**: Images combining Near-IR, Signal, and Reflectivity channels
2. **Natural image dataset**: RGB images of snow poles in various conditions

## Dataset Structure

- **LiDAR data**: `data/lidar/`

  - `combined_color/`: LiDAR modality images (RGB-like)
  - `labels/`: Annotations in YOLO format

- **RGB data**: `data/rgb/`
  - `images/`: Natural RGB images
  - `labels/`: Annotations in YOLO format

## Setup Instructions

### Environment Setup

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd snow-pole-detection
   ```

2. Create and activate a virtual environment:

   ```bash
   # Using conda
   conda create -n snow-pole python=3.9
   conda activate snow-pole

   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

The datasets should be organized as follows:

```
data/
├── lidar/
│   ├── combined_color/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── valid/
│       └── test/
├── rgb/
│   ├── images/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── valid/
│       └── test/
└── processed/
    └── ... (generated during preprocessing)
```

## Project Structure

```
./
├── config/                # Configuration files
├── data/                  # Datasets
├── notebooks/             # Jupyter notebooks for EDA
├── papers/                # Relevant research papers
├── results/               # Results and outputs
│   ├── logs/              # Training logs
│   ├── models/            # Saved model checkpoints
│   └── visualizations/    # Output visualizations
└── src/                   # Source code
    ├── data/              # Data processing utilities
    ├── evaluation/        # Evaluation metrics and tools
    ├── models/            # Model definitions
    ├── training/          # Training loops and routines
    └── utils/             # Utility functions
```

## Usage

### Exploratory Data Analysis

Run the notebooks in the `notebooks/` directory to explore the datasets:

```bash
jupyter notebook notebooks/
```

### Training

To train a model on the LiDAR dataset:

```bash
python src/train.py --config config/lidar_train_config.yaml
```

To train a model on the RGB dataset:

```bash
python src/train.py --config config/rgb_train_config.yaml
```

### Evaluation

Evaluate a trained model on the test set:

```bash
python src/evaluate.py --model results/models/your_model.pt --dataset lidar
```

### Inference

Run inference on a single image:

```bash
python src/predict.py --model results/models/your_model.pt --image path/to/image.jpg
```

## Performance Metrics

The project evaluates models using:

- Precision
- Recall
- mAP@50
- mAP@0.5:0.95

## License

This project is for academic purposes only. Redistribution of the datasets is prohibited.

## Acknowledgments

The datasets are provided by the NAPLab at NTNU as part of the TDT4265 course.
