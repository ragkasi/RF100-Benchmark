import cv2
import os
import random
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---

RF100_ROOT = Path("/fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark/data/rf100_yolov5")
PRED_ROOT = Path("/fs/scratch/PAS3162/ragkasi/rf100_results/groundingdino")
OUTPUT_DIR = Path("/fs/scratch/PAS3162/ragkasi/vis_results")

# How many images to save per dataset
NUM_SAMPLES = 3


def load_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def draw_boxes(img, label_file, class_names, color, thickness=2):
    if not label_file.exists():
        return img
    
    h, w, _ = img.shape
    with open(label_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        
        # Denormalize
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # Draw Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw Label
        if cls_id < len(class_names):
            label = class_names[cls_id]
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = sorted([d for d in RF100_ROOT.iterdir() if d.is_dir()])
    
    print(f"Generating visualizations in {OUTPUT_DIR}...")
    
    for ds_dir in tqdm(datasets):
        ds_name = ds_dir.name
        yaml_path = ds_dir / "data.yaml"
        if not yaml_path.exists(): continue
        
        try:
            class_names = load_classes(yaml_path)

        except:
            continue

        # Find images (Test set)
        img_dir = ds_dir / "test" / "images"
        if not img_dir.exists(): img_dir = ds_dir / "valid" / "images"

        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        random.shuffle(images)

        

        # Process a few samples
        for i, img_path in enumerate(images[:NUM_SAMPLES]):
            # Load Image
            img = cv2.imread(str(img_path))
            if img is None: continue

            
            # Paths to labels
            gt_path = ds_dir / "test" / "labels" / f"{img_path.stem}.txt"
            if not gt_path.exists(): gt_path = ds_dir / "valid" / "labels" / f"{img_path.stem}.txt"

            
            pred_path = PRED_ROOT / ds_name / "labels" / f"{img_path.stem}.txt"

            
            # Draw Ground Truth (Green)
            img_gt = img.copy()
            img_gt = draw_boxes(img_gt, gt_path, class_names, (0, 255, 0))
            cv2.putText(img_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw Prediction (Red)
            img_pred = img.copy()
            img_pred = draw_boxes(img_pred, pred_path, class_names, (0, 0, 255))
            cv2.putText(img_pred, "Grounding DINO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            
            # Combine Side-by-Side
            combined = cv2.hconcat([img_gt, img_pred])
            
            # Save
            save_path = OUTPUT_DIR / f"{ds_name}_{i}.jpg"
            cv2.imwrite(str(save_path), combined)


if __name__ == "__main__":
    main()
