"""
Evaluation metrics for snow pole detection.
"""
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    
    Args:
        box1 (np.ndarray): First box in format [x1, y1, x2, y2]
        box2 (np.ndarray): Second box in format [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou


def calculate_ap(recalls, precisions, method='interp'):
    """
    Calculate Average Precision (AP).
    
    Args:
        recalls (np.ndarray): Recall values
        precisions (np.ndarray): Precision values
        method (str): Method to calculate AP ('interp' or 'area')
        
    Returns:
        float: AP value
    """
    if method == 'area':
        # Calculate AP as area under PR curve using trapezoidal rule
        ap = np.trapz(precisions, recalls)
    else:  # 'interp' method (COCO-style)
        # Make sure recalls array starts with 0 and ends with 1
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        # Ensure precision array has matching size and starts high and ends low
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        
        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
            
        # Look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Sum (\Delta recall) * precision
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
    return ap


def calculate_precision_recall(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
    """
    Calculate precision and recall values for different confidence thresholds.
    
    Args:
        pred_boxes (list): List of predicted bounding boxes
        pred_scores (list): List of confidence scores for predictions
        true_boxes (list): List of ground truth bounding boxes
        iou_threshold (float): IoU threshold for a correct detection
        
    Returns:
        tuple: (precision values, recall values, average precision)
    """
    # Sort predictions by confidence score (high to low)
    sorted_indices = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    
    n_gt = len(true_boxes)
    n_pred = len(pred_boxes)
    
    # Array to keep track of whether each ground truth box has been detected
    gt_detected = [False] * n_gt
    
    # Arrays to store true positives and false positives for each prediction
    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)
    
    # Process each prediction
    for pred_idx, pred_box in enumerate(pred_boxes):
        # Find best matching ground truth box
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(true_boxes):
            # Skip if this ground truth has already been detected
            if gt_detected[gt_idx]:
                continue
                
            # Calculate IoU between prediction and this ground truth
            iou = calculate_iou(pred_box, gt_box)
            
            # Update best match if this IoU is better
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        # Check if the best match exceeds the IoU threshold
        if best_iou >= iou_threshold and best_gt_idx != -1:
            # True positive
            tp[pred_idx] = 1
            gt_detected[best_gt_idx] = True
        else:
            # False positive
            fp[pred_idx] = 1
    
    # Calculate cumulative false positives and true positives
    fp_cumsum = np.cumsum(fp)
    tp_cumsum = np.cumsum(tp)
    
    # Calculate precision and recall values
    recalls = tp_cumsum / n_gt if n_gt > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Calculate average precision
    ap = calculate_ap(recalls, precisions)
    
    return precisions, recalls, ap


def calculate_map(pred_boxes_list, pred_scores_list, true_boxes_list, iou_thresholds=None):
    """
    Calculate mAP at different IoU thresholds.
    
    Args:
        pred_boxes_list (list): List of predicted bounding boxes for each image
        pred_scores_list (list): List of confidence scores for each image
        true_boxes_list (list): List of ground truth bounding boxes for each image
        iou_thresholds (list): List of IoU thresholds for evaluation
        
    Returns:
        dict: mAP values at different thresholds and overall
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    
    # Calculate AP at each IoU threshold
    ap_values = []
    for iou_threshold in iou_thresholds:
        ap_sum = 0.0
        
        # Process each image
        for pred_boxes, pred_scores, true_boxes in zip(pred_boxes_list, pred_scores_list, true_boxes_list):
            # Skip if there are no predictions or no ground truths
            if len(pred_boxes) == 0 or len(true_boxes) == 0:
                continue
                
            # Calculate precision, recall, and AP for this image
            _, _, ap = calculate_precision_recall(pred_boxes, pred_scores, true_boxes, iou_threshold)
            ap_sum += ap
            
        # Calculate mean AP over all images
        n_valid_images = sum(1 for pb, tb in zip(pred_boxes_list, true_boxes_list) if len(pb) > 0 and len(tb) > 0)
        if n_valid_images > 0:
            ap_values.append(ap_sum / n_valid_images)
        else:
            ap_values.append(0.0)
            
    # Create results dictionary
    results = {
        'mAP@50': ap_values[0] if len(iou_thresholds) >= 1 and iou_thresholds[0] == 0.5 else None,
    }
    
    # If multiple IoU thresholds, calculate mAP@0.5:0.95 (COCO-style)
    if len(iou_thresholds) > 1 and np.allclose(iou_thresholds, np.linspace(0.5, 0.95, 10)):
        results['mAP@0.5:0.95'] = np.mean(ap_values)
    
    return results


def compute_coco_metrics(predictions, targets, iou_thres=0.5):
    """
    Compute COCO-style metrics for object detection.
    
    Args:
        predictions (list): List of prediction dictionaries
        targets (list): List of target dictionaries
        iou_thres (float): IoU threshold for a correct detection
        
    Returns:
        dict: Dictionary with precision, recall, and mAP metrics
    """
    # Extract predictions and targets by image
    pred_by_img = defaultdict(list)
    target_by_img = defaultdict(list)
    
    for pred in predictions:
        img_id = pred['image_id']
        pred_by_img[img_id].append({
            'bbox': pred['bbox'],
            'score': pred['score'],
            'category_id': pred['category_id']
        })
    
    for target in targets:
        img_id = target['image_id']
        target_by_img[img_id].append({
            'bbox': target['bbox'],
            'category_id': target['category_id']
        })
    
    # Calculate metrics
    pred_boxes_list = []
    pred_scores_list = []
    pred_classes_list = []
    true_boxes_list = []
    true_classes_list = []
    
    for img_id in set(list(pred_by_img.keys()) + list(target_by_img.keys())):
        # Get predictions and targets for this image
        preds = pred_by_img[img_id]
        targs = target_by_img[img_id]
        
        # Extract boxes, scores, and classes
        pred_boxes = [p['bbox'] for p in preds]
        pred_scores = [p['score'] for p in preds]
        pred_classes = [p['category_id'] for p in preds]
        
        true_boxes = [t['bbox'] for t in targs]
        true_classes = [t['category_id'] for t in targs]
        
        # Add to lists
        pred_boxes_list.append(pred_boxes)
        pred_scores_list.append(pred_scores)
        pred_classes_list.append(pred_classes)
        true_boxes_list.append(true_boxes)
        true_classes_list.append(true_classes)
    
    # Calculate per-class metrics
    class_ids = set()
    for classes in true_classes_list + pred_classes_list:
        class_ids.update(classes)
    
    metrics = {}
    for class_id in class_ids:
        # Filter predictions and targets by class
        class_pred_boxes = []
        class_pred_scores = []
        class_true_boxes = []
        
        for idx in range(len(pred_boxes_list)):
            pb = [pred_boxes_list[idx][i] for i, c in enumerate(pred_classes_list[idx]) if c == class_id]
            ps = [pred_scores_list[idx][i] for i, c in enumerate(pred_classes_list[idx]) if c == class_id]
            tb = [true_boxes_list[idx][i] for i, c in enumerate(true_classes_list[idx]) if c == class_id]
            
            class_pred_boxes.append(pb)
            class_pred_scores.append(ps)
            class_true_boxes.append(tb)
        
        # Calculate class metrics
        class_metrics = calculate_map(
            class_pred_boxes, 
            class_pred_scores, 
            class_true_boxes, 
            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
        
        metrics[f'class_{class_id}'] = class_metrics
    
    # Calculate overall metrics
    overall_metrics = calculate_map(
        pred_boxes_list, 
        pred_scores_list, 
        true_boxes_list, 
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    
    metrics['overall'] = overall_metrics
    
    return metrics


def plot_precision_recall_curve(precisions, recalls, save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        precisions (np.ndarray): Precision values
        recalls (np.ndarray): Recall values
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    # Add AP value to plot
    ap = calculate_ap(recalls, precisions)
    plt.text(0.5, 0.5, f'AP: {ap:.4f}', 
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 