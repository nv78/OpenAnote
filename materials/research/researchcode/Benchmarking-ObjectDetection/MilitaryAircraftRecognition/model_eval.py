import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ObjectDetectionEvaluator:
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.0):
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def prepare_data(self, raw_data, is_prediction=False):
        parsed = defaultdict(list)
        for entry in raw_data:
            img_id = entry['id']
            labels = entry['label']
            boxes = entry['boxes']
            scores = entry.get('scores', [1.0] * len(boxes))

            for i in range(len(boxes)):
                item = {
                    'label': labels[i],
                    'box': boxes[i],
                }
                if is_prediction:
                    item['confidence'] = scores[i]
                parsed[img_id].append(item)
        return parsed

    def evaluate_image(self, gt_items, pred_items):
        pred_items = [p for p in pred_items if p['confidence'] >= self.confidence_threshold]
        pred_items.sort(key=lambda x: x['confidence'], reverse=True)

        iou_matrix = np.zeros((len(pred_items), len(gt_items)))
        for i, pred in enumerate(pred_items):
            for j, gt in enumerate(gt_items):
                if pred['label'] == gt['label']:
                    iou_matrix[i, j] = self.calculate_iou(pred['box'], gt['box'])

        matched_gt = set()
        matched_pred = set()
        matched_pairs = []

        for i in range(len(pred_items)):
            best_iou = 0
            best_gt_idx = -1
            for j in range(len(gt_items)):
                if j not in matched_gt and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
                matched_pairs.append((i, best_gt_idx, best_iou))

        tp = len(matched_pairs)
        fp = len(pred_items) - tp
        fn = len(gt_items) - tp
        all_ious = [pair[2] for pair in matched_pairs]

        return tp, fp, fn, matched_pairs, all_ious

    def evaluate(self, gt_raw, pred_raw):
        gt_data = self.prepare_data(gt_raw, is_prediction=False)
        pred_data = self.prepare_data(pred_raw, is_prediction=True)

        total_tp = total_fp = total_fn = 0
        all_ious = []
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        y_true, y_pred = [], []

        for img_id in gt_data:
            gt_items = gt_data[img_id]
            pred_items = pred_data.get(img_id, [])

            tp, fp, fn, matches, ious = self.evaluate_image(gt_items, pred_items)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_ious.extend(ious)

            matched_gt_ids = {m[1] for m in matches}
            matched_pred_ids = {m[0] for m in matches}

            for i, j, _ in matches:
                label = gt_items[j]['label']
                class_metrics[label]['tp'] += 1
                y_true.append(label)
                y_pred.append(label)

            for i, pred in enumerate(pred_items):
                if i not in matched_pred_ids:
                    label = pred['label']
                    class_metrics[label]['fp'] += 1
                    y_true.append('background')
                    y_pred.append(label)

            for j, gt in enumerate(gt_items):
                if j not in matched_gt_ids:
                    label = gt['label']
                    class_metrics[label]['fn'] += 1
                    y_true.append(label)
                    y_pred.append('background')

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean(all_ious) if all_ious else 0
        map_score = np.mean([
            m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
            for m in class_metrics.values()
        ])

        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_iou': mean_iou,
                'mAP': map_score,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
            },
            'class_metrics': class_metrics,
            'confusion_data': (y_true, y_pred)
        }

    def plot_confusion(self, y_true, y_pred):
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def print_report(self, results):
        print("\n=== Overall Metrics ===")
        for k, v in results['overall'].items():
            print(f"{k.capitalize()}: {v:.4f}")

        print("\n=== Per-Class Metrics ===")
        print(f"{'Class':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
        for cls, m in results['class_metrics'].items():
            print(f"{cls:<10} {m['tp']:<5} {m['fp']:<5} {m['fn']:<5}")


def main():
    gt_path = "/Users/mohamedzakariakheder/Documents/code/Anote/cv-research/ground_truths (3).json"
    pred_path = "/Users/mohamedzakariakheder/Documents/code/Anote/cv-research/predictions (1).json"

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    evaluator = ObjectDetectionEvaluator(iou_threshold=0.5, confidence_threshold=0.0)
    results = evaluator.evaluate(gt_data, pred_data)
    evaluator.print_report(results)
    evaluator.plot_confusion(*results['confusion_data'])


if __name__ == "__main__":
    main()
