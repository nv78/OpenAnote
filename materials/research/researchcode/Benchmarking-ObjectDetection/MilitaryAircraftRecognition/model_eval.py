import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class ObjectDetectionEvaluator:
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.0):
        """
        Initialize evaluator with IoU and confidence thresholds.
        
        Args:
            iou_threshold (float): IoU threshold for considering a detection as correct
            confidence_threshold (float): Confidence threshold for filtering predictions
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            box1, box2: dict with keys 'x_min', 'y_min', 'x_max', 'y_max'
        
        Returns:
            float: IoU value
        """
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = box1['x_min'], box1['y_min'], box1['x_max'], box1['y_max']
        x2_min, y2_min, x2_max, y2_max = box2['x_min'], box2['y_min'], box2['x_max'], box2['y_max']
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(self, gt_data, pred_data):
        """
        Match ground truth and prediction data for the same images.
        
        Args:
            gt_data (list): Ground truth data
            pred_data (list): Prediction data
        
        Returns:
            dict: Matched data organized by image
        """
        matched_data = {}
        
        # For dataset format, we'll use index as image identifier
        # In practice, you might want to use image filename or hash
        for i, (gt_item, pred_item) in enumerate(zip(gt_data, pred_data)):
            image_id = f"image_{i}"
            
            # For datasets, we can also use image properties as identifier
            # This is more robust if images have consistent properties
            try:
                if hasattr(gt_item.get('image'), 'size'):
                    img_size = gt_item['image'].size
                    image_id = f"image_{i}_{img_size[0]}x{img_size[1]}"
            except:
                image_id = f"image_{i}"
            
            if image_id not in matched_data:
                matched_data[image_id] = {'gt': [], 'pred': []}
            
            matched_data[image_id]['gt'].append({
                'label': gt_item['label'],
                'x_min': gt_item['x_min'],
                'y_min': gt_item['y_min'],
                'x_max': gt_item['x_max'],
                'y_max': gt_item['y_max']
            })
            
            matched_data[image_id]['pred'].append({
                'label': pred_item['label'],
                'x_min': pred_item['x_min'],
                'y_min': pred_item['y_min'],
                'x_max': pred_item['x_max'],
                'y_max': pred_item['y_max'],
                'confidence': pred_item.get('confidence', 1.0)  # Default confidence if not provided
            })
        
        return matched_data
    
    def evaluate_detection(self, gt_boxes, pred_boxes):
        """
        Evaluate detections for a single image.
        
        Args:
            gt_boxes (list): Ground truth boxes
            pred_boxes (list): Predicted boxes
        
        Returns:
            dict: Evaluation results
        """
        # Filter predictions by confidence
        pred_boxes = [box for box in pred_boxes if box.get('confidence', 1.0) >= self.confidence_threshold]
        
        # Sort predictions by confidence (highest first)
        pred_boxes = sorted(pred_boxes, key=lambda x: x.get('confidence', 1.0), reverse=True)
        
        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)
        
        if num_gt == 0 and num_pred == 0:
            return {'tp': 0, 'fp': 0, 'fn': 0, 'ious': [], 'matched_pairs': []}
        
        if num_gt == 0:
            return {'tp': 0, 'fp': num_pred, 'fn': 0, 'ious': [], 'matched_pairs': []}
        
        if num_pred == 0:
            return {'tp': 0, 'fp': 0, 'fn': num_gt, 'ious': [], 'matched_pairs': []}
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((num_pred, num_gt))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if pred_box['label'] == gt_box['label']:  # Only match same class
                    iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Match predictions to ground truth using greedy matching
        matched_gt = set()
        matched_pred = set()
        matched_pairs = []
        tp = 0
        
        for i in range(num_pred):
            best_iou = 0
            best_gt_idx = -1
            
            for j in range(num_gt):
                if j not in matched_gt and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j
            
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
                matched_pairs.append((i, best_gt_idx, best_iou))
        
        fp = num_pred - tp
        fn = num_gt - tp
        
        all_ious = iou_matrix[iou_matrix > 0].flatten()
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'ious': all_ious.tolist(),
            'matched_pairs': matched_pairs
        }
    
    def calculate_metrics(self, gt_data, pred_data):
        """
        Calculate all evaluation metrics.
        
        Args:
            gt_data (list): Ground truth data
            pred_data (list): Prediction data
        
        Returns:
            dict: All calculated metrics
        """
        matched_data = self.match_detections(gt_data, pred_data)
        
        # Initialize counters
        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_ious = []
        
        # Class-wise metrics
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # For confusion matrix
        y_true = []
        y_pred = []
        
        # Process each image
        for image_id, data in matched_data.items():
            gt_boxes = data['gt']
            pred_boxes = data['pred']
            
            result = self.evaluate_detection(gt_boxes, pred_boxes)
            
            total_tp += result['tp']
            total_fp += result['fp']
            total_fn += result['fn']
            all_ious.extend(result['ious'])
            
            # Update class-wise metrics
            gt_labels = [box['label'] for box in gt_boxes]
            pred_labels = [box['label'] for box in pred_boxes]
            
            # For matched pairs, both are correct
            for pred_idx, gt_idx, iou in result['matched_pairs']:
                class_metrics[gt_boxes[gt_idx]['label']]['tp'] += 1
                y_true.append(gt_boxes[gt_idx]['label'])
                y_pred.append(pred_boxes[pred_idx]['label'])
            
            # Unmatched predictions are false positives
            matched_pred_indices = {pair[0] for pair in result['matched_pairs']}
            for i, pred_box in enumerate(pred_boxes):
                if i not in matched_pred_indices:
                    class_metrics[pred_box['label']]['fp'] += 1
                    y_pred.append(pred_box['label'])
                    # For confusion matrix, we need a corresponding true label
                    # We'll use 'background' or the most frequent class as approximation
                    y_true.append('background')
            
            # Unmatched ground truth are false negatives
            matched_gt_indices = {pair[1] for pair in result['matched_pairs']}
            for i, gt_box in enumerate(gt_boxes):
                if i not in matched_gt_indices:
                    class_metrics[gt_box['label']]['fn'] += 1
                    y_true.append(gt_box['label'])
                    y_pred.append('background')
        
        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
        # Calculate class-wise metrics
        class_precisions = {}
        class_recalls = {}
        class_f1s = {}
        
        for class_name, metrics in class_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            class_precisions[class_name] = class_precision
            class_recalls[class_name] = class_recall
            class_f1s[class_name] = class_f1
        
        # Calculate mAP (simplified version - single IoU threshold)
        map_score = np.mean(list(class_precisions.values())) if class_precisions else 0
        
        # Mean IoU
        mean_iou = np.mean(all_ious) if all_ious else 0
        
        return {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'map': map_score,
                'mean_iou': mean_iou,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn
            },
            'class_wise': {
                'precision': class_precisions,
                'recall': class_recalls,
                'f1': class_f1s
            },
            'confusion_matrix_data': {
                'y_true': y_true,
                'y_pred': y_pred
            }
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix."""
        if not y_true or not y_pred:
            print("No data available for confusion matrix")
            return
        
        # Get unique labels
        labels = sorted(list(set(y_true + y_pred)))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def print_results(self, results):
        """Print formatted results."""
        print("="*50)
        print("OBJECT DETECTION EVALUATION RESULTS")
        print("="*50)
        
        overall = results['overall']
        print(f"\nOVERALL METRICS:")
        print(f"Accuracy:     {overall['accuracy']:.4f}")
        print(f"Precision:    {overall['precision']:.4f}")
        print(f"Recall:       {overall['recall']:.4f}")
        print(f"F1-Score:     {overall['f1']:.4f}")
        print(f"mAP:          {overall['map']:.4f}")
        print(f"Mean IoU:     {overall['mean_iou']:.4f}")
        print(f"\nCOUNTS:")
        print(f"True Positives:  {overall['total_tp']}")
        print(f"False Positives: {overall['total_fp']}")
        print(f"False Negatives: {overall['total_fn']}")
        
        class_wise = results['class_wise']
        if class_wise['precision']:
            print(f"\nCLASS-WISE METRICS:")
            print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 50)
            for class_name in class_wise['precision']:
                precision = class_wise['precision'][class_name]
                recall = class_wise['recall'][class_name]
                f1 = class_wise['f1'][class_name]
                print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")


def load_dataset_split(dataset, split_name):
    """
    Load data from Hugging Face Dataset split.
    
    Args:
        dataset: DatasetDict or Dataset object
        split_name: Name of the split ('train', 'validation', 'test')
    
    Returns:
        list: List of dictionaries with bounding box and label info
    """
    if hasattr(dataset, 'keys') and split_name in dataset:
        # DatasetDict case
        split_data = dataset[split_name]
    else:
        # Single Dataset case
        split_data = dataset
    
    # Convert to list of dictionaries
    data = []
    for i in range(len(split_data)):
        item = split_data[i]
        data.append({
            'image': item['image'],  # Keep PIL image for reference
            'label': item['label'],
            'x_min': item['x_min'],
            'y_min': item['y_min'],
            'x_max': item['x_max'],
            'y_max': item['y_max']
        })
    
    return data

def load_json_data(file_path):
    """
    Load data from JSON file (kept for backward compatibility).
    Note: PIL Image objects cannot be directly serialized to JSON.
    This function assumes the JSON contains the bounding box and label info only.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def evaluate_dataset(dataset_dict, gt_split='test', pred_split='test', model_predictions=None):
    """
    Evaluate object detection performance on a dataset split.
    
    Args:
        dataset_dict: DatasetDict object with splits
        gt_split: Split to use for ground truth (default: 'test')
        pred_split: Split to use for predictions (default: 'test') 
        model_predictions: Optional list of predictions. If None, uses pred_split as predictions
    
    Returns:
        dict: Evaluation results
    """
    # Load ground truth data
    gt_data = load_dataset_split(dataset_dict, gt_split)
    
    # Load prediction data
    if model_predictions is not None:
        pred_data = model_predictions
    else:
        pred_data = load_dataset_split(dataset_dict, pred_split)
    
    # Ensure same length
    min_length = min(len(gt_data), len(pred_data))
    gt_data = gt_data[:min_length]
    pred_data = pred_data[:min_length]
    
    print(f"Evaluating {min_length} samples from {gt_split} split")
    
    # Initialize evaluator
    evaluator = ObjectDetectionEvaluator(iou_threshold=0.5, confidence_threshold=0.0)
    
    # Calculate metrics
    results = evaluator.calculate_metrics(gt_data, pred_data)
    
    return results, evaluator


def main():
   
    print("Object Detection Evaluation Script")
    print("=" * 60)
    
    
    print("Running demo with sample data...")
    

    #REPLACE FOR YOUR USE
    sample_gt_data = load_dataset('your_dataset')
    
    sample_pred_data = load_dataset('your_dataset')
    
    # Initialize evaluator
    evaluator = ObjectDetectionEvaluator(iou_threshold=0.5, confidence_threshold=0.0)
    
    # Calculate metrics
    results = evaluator.calculate_metrics(sample_gt_data, sample_pred_data)
    
    # Print results
    evaluator.print_results(results)
    
    # Plot confusion matrix
    cm_data = results['confusion_matrix_data']
    evaluator.plot_confusion_matrix(cm_data['y_true'], cm_data['y_pred'])
    
    print("\n" + "="*60)
    print("To use with your DatasetDict:")
    print("1. Load your dataset: dataset = load_dataset('your_dataset')")
    print("2. Call: results, evaluator = evaluate_dataset(dataset)")
    print("3. Print results: evaluator.print_results(results)")
    print("="*60)


if __name__ == "__main__":
    main()