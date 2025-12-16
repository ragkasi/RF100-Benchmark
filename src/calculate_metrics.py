import os
import glob
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import box_iou

# --- CONFIGURATION ---

# Where the Ground Truth is (ESS)
GT_ROOT = Path("/fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark/data/rf100_yolov5")

# Where your Predictions are (SCRATCH)
PRED_ROOT = Path("/fs/scratch/PAS3162/ragkasi/rf100_results/groundingdino")

# Output CSV
OUTPUT_CSV = Path("/fs/scratch/PAS3162/ragkasi/groundingdino_rf100_metrics.csv")


def parse_yolo_file(file_path):
    """Parses a YOLO format txt file into a tensor of boxes and classes."""
    if not os.path.exists(file_path):
        return torch.tensor([]), torch.tensor([])
    
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # class, cx, cy, w, h
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                # Convert to xyxy for IoU
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                data.append([x1, y1, x2, y2, cls])
    
    if not data:
        return torch.tensor([]), torch.tensor([])
        
    data = torch.tensor(data)
    boxes = data[:, :4]
    classes = data[:, 4].int()
    return boxes, classes


def calculate_dataset_metrics(dataset_name):
    # Setup paths
    gt_dir = GT_ROOT / dataset_name / "test" / "labels"
    if not gt_dir.exists():
        gt_dir = GT_ROOT / dataset_name / "valid" / "labels"

        
    pred_dir = PRED_ROOT / dataset_name / "labels"

    
    if not gt_dir.exists() or not pred_dir.exists():
        return None



    # Get all images in GT
    gt_files = list(gt_dir.glob("*.txt"))

    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for gt_file in gt_files:
        stem = gt_file.stem
        pred_file = pred_dir / f"{stem}.txt"
        
        gt_boxes, gt_classes = parse_yolo_file(gt_file)
        pred_boxes, pred_classes = parse_yolo_file(pred_file)

        
        # 1. Handle empty cases
        if len(gt_boxes) == 0:
            fp_total += len(pred_boxes)
            continue
        if len(pred_boxes) == 0:
            fn_total += len(gt_boxes)
            continue
            
        # 2. Calculate IoU Matrix
        iou_matrix = box_iou(gt_boxes, pred_boxes)
        
        # 3. Match Boxes (IoU > 0.5 and Same Class)
        matched_gt = set()
        matched_pred = set()
        

        # Greedy matching
        for i in range(len(gt_boxes)):
            best_iou = 0
            best_j = -1
            for j in range(len(pred_boxes)):
                if j in matched_pred:
                    continue
                # Check Class Match
                if gt_classes[i] != pred_classes[j]:
                    continue
                # Check IoU Match
                if iou_matrix[i, j] > 0.5 and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j

            
            if best_j != -1:
                matched_gt.add(i)
                matched_pred.add(best_j)
                tp_total += 1
            else:
                fn_total += 1

        
        fp_total += len(pred_boxes) - len(matched_pred)

    # 4. Compute Metrics for this Dataset
    epsilon = 1e-6
    precision = tp_total / (tp_total + fp_total + epsilon)
    recall = tp_total / (tp_total + fn_total + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    
    return {
        "dataset": dataset_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total
    }


def main():
    print("Calculating Metrics for RF100...")
    datasets = sorted([d.name for d in GT_ROOT.iterdir() if d.is_dir()])
    results = []
    
    for ds in tqdm(datasets):
        try:
            metrics = calculate_dataset_metrics(ds)
            if metrics:
                results.append(metrics)
        except Exception as e:
            print(f"Error on {ds}: {e}")

            
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n=== SUMMARY ===")
    print(f"Mean F1 Score: {df['f1'].mean():.4f}")
    print(f"Mean Precision: {df['precision'].mean():.4f}")
    print(f"Mean Recall: {df['recall'].mean():.4f}")
    print(f"Detailed results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
