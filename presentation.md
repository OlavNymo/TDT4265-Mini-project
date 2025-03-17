# Object Detection Presentation Outline: Snow Pole Detection for Autonomous Driving

## Slide 1: Title Slide

Project Title: Snow Pole Detection for Autonomous Vehicles Using YOLOv12
Course: TDT4265 Mini-Project
Your Name

## Slide 2: Background & Motivation

Autonomous driving challenges in harsh winter conditions
Importance of snow poles for road detection when road markings are covered
Personal interest in self-driving technologies
Particular relevance for Norway and Nordic countries with extensive winter seasons
Safety implications for autonomous vehicles in winter environments

## Slide 3: Problem Statement

Detection of snow poles in real-time from both:
LiDAR data (Near-IR, Signal, Reflectivity channels combined as RGB)
Natural RGB images
Goal: Develop efficient, real-time capable model for edge devices

## Slide 4: Approach & Strategy

Literature review of relevant papers:
"A Pole Detection and Geospatial Localization Framework using LiDAR-GNSS Data Fusion"
"SnowPole Detection: A comprehensive dataset for detection and localization using LiDAR imaging in Nordic winter conditions"
Dataset exploration and understanding
Research on state-of-the-art object detection models
Focus on YOLOv12 as recent advancement in object detection

## Slide 5: Data Analysis (1/2)

Dataset overview:
Two datasets: LiDAR images and natural RGB images
Single class detection (snow poles)
YOLO format annotations
Sample images from both datasets with visualized annotations
Data distribution analysis

## Slide 6: Data Analysis (2/2)

Key findings from exploratory data analysis
Challenges identified in the datasets
Distribution of pole appearances, sizes, and contexts
Lighting and weather condition variations
Include visualizations from your EDA notebook

## Slide 7: Methods & Models (1/3)

Object detection approach selection
YOLOv12 architecture overview
Advantages of YOLOv12 over previous iterations
Size variants comparison (YOLOv12n, YOLOv12s, etc.)

## Slide 8: Methods & Models (2/3)

Implementation details:
Model selection rationale (emphasize small model due to edge deployment requirement)
Training configuration
Data augmentation strategies
Training hardware used and compute time

## Slide 9: Methods & Models (3/3)

Optimization strategies:
Hyperparameter tuning
Learning rate scheduling
Optimal parameters from YOLOv12 paper
Any custom modifications made

## Slide 10: Results (1/2)

Performance metrics on test set:
Precision
Recall
mAP@50
mAP@0.5:0.95
Comparison between LiDAR and RGB-based models
Comparison with baseline performance from research papers

## Slide 11: Results (2/2)

Visual results:
Example detections on challenging cases
Failure cases analysis
Inference time measurements
Model size and resource utilization

## Slide 12: Discussion

Analysis of what worked well and why
Comparison with findings from reference papers
Limitations of the approach
Potential improvements for future work
Generalizability to other domains or conditions

## Slide 13: Key Learning Points

Effectiveness of pre-trained state-of-the-art models
Performance exceeding results from research papers
Importance of domain knowledge for parameter tuning
Insights about balancing model size with performance for edge deployment
Specific learnings about object detection in challenging winter conditions

## Slide 14: Sustainability Analysis

Compute resources used during training
Total GPU hours
Estimated energy consumption
Energy equivalent in practical terms:
How far a Tesla Model Y could travel using the same energy
Considerations for more energy-efficient training approaches

## Slide 15: References

Academic papers:
"A Pole Detection and Geospatial Localization Framework using LiDAR-GNSS Data Fusion"
"SnowPole Detection: A comprehensive dataset for detection and localization using LiDAR imaging in Nordic winter conditions"
YOLOv12 paper arXiv:2502.12524
Technical resources used
Dataset source acknowledgment

## Slide 16: Questions

Thank you slide
Contact information
