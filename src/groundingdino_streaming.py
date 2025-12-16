import os
import shutil
import csv
import cv2
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from roboflow import Roboflow
from torchvision.ops import box_iou
from tqdm import tqdm
import sys

# --- SETUP PATHS ---
# Add GroundingDINO to python path so imports work
SCRATCH_ROOT = Path("/fs/scratch/PAS3162/ragkasi")
sys.path.append(str(SCRATCH_ROOT / "GroundingDINO"))
from groundingdino.util.inference import load_model, load_image, predict

# --- CONFIGURATION ---
WORKSPACE = "roboflow-100"
EXPORT_FORMAT = "yolov5"

# Directories in your SCRATCH space
TEMP_DATA_DIR = SCRATCH_ROOT / "temp_data"           # Temporary download spot
RESULTS_DIR = SCRATCH_ROOT / "rf100_results"         # Where .txt labels go
VIS_DIR = SCRATCH_ROOT / "vis_results"               # Where debug images go
METRICS_CSV = SCRATCH_ROOT / "groundingdino_metrics.csv"

# Model Paths
WEIGHTS_PATH = SCRATCH_ROOT / "weights/groundingdino_swint_ogc.pth"
CONFIG_PATH = SCRATCH_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# The Full RF100 List
RF100_DATASETS = [
    ("4-fold-defect", 1), ("abdomen-mri", 1), ("acl-x-ray", 1), ("activity-diagrams-qdobr", 1),
    ("aerial-cows", 1), ("aerial-pool", 1), ("aerial-spheres", 1), ("animals-ij5d2", 1),
    ("apex-videogame", 1), ("apples-fvpl5", 1), ("aquarium-qlnqy", 1), ("asbestos", 1),
    ("avatar-recognition-nuexe", 1), ("axial-mri", 1), ("bacteria-ptywi", 1), ("bccd-ouzjz", 1),
    ("bees-jt5in", 1), ("bone-fracture-7fylg", 1), ("brain-tumor-m2pbp", 1), ("cable-damage", 1),
    ("cables-nl42k", 1), ("cavity-rs0uf", 1), ("cell-towers", 1), ("cells-uyemf", 1),
    ("chess-pieces-mjzgj", 1), ("circuit-elements", 1), ("circuit-voltages", 1), ("cloud-types", 1),
    ("coins-1apki", 1), ("construction-safety-gsnvb", 1), ("coral-lwptl", 1), ("corrosion-bi3q3", 1),
    ("cotton-20xz5", 1), ("cotton-plant-disease", 1), ("csgo-videogame", 1), ("currency-v4f8j", 1),
    ("digits-t2eg6", 1), ("document-parts", 1), ("excavators-czvg9", 1), ("farcry6-videogame", 1),
    ("fish-market-ggjso", 4), ("flir-camera-objects", 1), ("furniture-ngpea", 1), ("gauge-u2lwv", 4),
    ("grass-weeds", 1), ("gynecology-mri", 1), ("halo-infinite-angel-videogame", 1), ("hand-gestures-jps7z", 1),
    ("insects-mytwu", 1), ("leaf-disease-nsdsr", 1), ("lettuce-pallets", 1), ("liver-disease", 1),
    ("marbles", 1), ("mask-wearing-608pr", 1), ("mitosis-gjs3g", 1), ("number-ops", 1),
    ("paper-parts", 1), ("paragraphs-co84b", 1), ("parasites-1s07h", 1), ("peanuts-sd4kf", 1),
    ("peixos-fish", 1), ("people-in-paintings", 1), ("pests-2xlvx", 1), ("phages", 1),
    ("pills-sxdht", 1), ("poker-cards-cxcvz", 1), ("printed-circuit-board", 1), ("radio-signal", 1),
    ("road-signs-6ih4y", 1), ("road-traffic", 1), ("robomasters-285km", 1), ("secondary-chains", 1),
    ("sedimentary-features-9eosf", 4), ("shark-teeth-5atku", 1), ("sign-language-sokdr", 1),
    ("signatures-xc8up", 1), ("smoke-uvylj", 1), ("soccer-players-5fuqs", 1), ("soda-bottles", 4),
    ("solar-panels-taxvb", 1), ("stomata-cells", 1), ("street-work", 1), ("tabular-data-wf9uh", 1),
    ("team-fight-tactics", 1), ("thermal-cheetah-my4dp", 1), ("thermal-dogs-and-people-x6ejw", 1),
    ("trail-camera", 1), ("truck-movement", 1), ("tweeter-posts", 1), ("tweeter-profile", 1),
    ("underwater-objects-5v7p8", 1), ("underwater-pipes-4ng4t", 1), ("uno-deck", 1),
    ("valentines-chocolate", 1), ("vehicles-q0x2v", 1), ("wall-damage", 1), ("washroom-rf1fa", 1),
    ("weed-crop-aerial", 1), ("wine-labels", 1), ("x-ray-rheumatology", 1),
]

# --- HELPER FUNCTIONS ---
def load_completed_datasets(csv_path):
    completed = set()
    if not csv_path.exists(): return completed
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "dataset" in row: completed.add(row["dataset"])
    return completed

def parse_yolo_file(file_path):
    if not os.path.exists(file_path): return torch.tensor([]), torch.tensor([])
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                data.append([x1, y1, x2, y2, cls])

    if not data: return torch.tensor([]), torch.tensor([])
    data = torch.tensor(data)
    return data[:, :4], data[:, 4].int()

def calculate_metrics(gt_dir, pred_dir):
    gt_files = list(gt_dir.glob("*.txt"))
    tp, fp, fn = 0, 0, 0

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        gt_boxes, gt_classes = parse_yolo_file(gt_file)
        pred_boxes, pred_classes = parse_yolo_file(pred_file)

        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue

        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue

        iou_matrix = box_iou(gt_boxes, pred_boxes)
        matched_gt = set()
        matched_pred = set()

        for i in range(len(gt_boxes)):
            best_iou, best_j = 0, -1
            for j in range(len(pred_boxes)):
                if j in matched_pred: continue
                if gt_classes[i] != pred_classes[j]: continue
                if iou_matrix[i, j] > 0.5 and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j

            if best_j != -1:
                matched_gt.add(i)
                matched_pred.add(best_j)
                tp += 1
            else:
                fn += 1
        fp += len(pred_boxes) - len(matched_pred)

    epsilon = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision, recall, f1, tp, fp, fn

def save_visualizations(image_dir, gt_dir, pred_dir, class_names, output_dir, ds_name, num_samples=3):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    random.shuffle(images)

    for i, img_path in enumerate(images[:num_samples]):
        img = cv2.imread(str(img_path))
        if img is None: continue
        # Draw GT (Green)
        gt_path = gt_dir / f"{img_path.stem}.txt"
        h, w, _ = img.shape
        img_vis = img.copy()

        if gt_path.exists():
            boxes, classes = parse_yolo_file(gt_path)
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = (box * torch.tensor([w, h, w, h])).int().tolist()
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Pred (Red)
        pred_path = pred_dir / f"{img_path.stem}.txt"
        if pred_path.exists():
            boxes, classes = parse_yolo_file(pred_path)
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = (box * torch.tensor([w, h, w, h])).int().tolist()
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(output_dir / f"{ds_name}_sample_{i}.jpg"), img_vis)

def main():
    # Check Environment
    if "ROBOFLOW_API_KEY" not in os.environ:
        raise ValueError("Please export ROBOFLOW_API_KEY")

    # Setup Folders
    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Model
    print("Loading GroundingDINO...")
    model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH))

    # Setup CSV
    completed = load_completed_datasets(METRICS_CSV)
    file_exists = METRICS_CSV.exists()
    csv_file = open(METRICS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["dataset", "precision", "recall", "f1", "tp", "fp", "fn"])

    if not file_exists: writer.writeheader()
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    ws = rf.workspace(WORKSPACE)

    print(f"Starting Benchmark on {len(RF100_DATASETS)} datasets...")

    for project_slug, version in RF100_DATASETS:
        if project_slug in completed:
            print(f"[SKIP] {project_slug} already done.")
            continue

        print(f"\n=== Processing: {project_slug} ===")
        dataset_dir = TEMP_DATA_DIR / project_slug
        
        # Download
        try:
            if dataset_dir.exists(): shutil.rmtree(dataset_dir) # Clean start
            project = ws.project(project_slug)
            project.version(version).download(EXPORT_FORMAT, location=str(dataset_dir))

        except Exception as e:
            print(f"[ERROR] Download failed for {project_slug}: {e}")
            continue

        # Setup Inference
        image_dir = dataset_dir / "test" / "images"
        gt_dir = dataset_dir / "test" / "labels"
        if not image_dir.exists(): 
            image_dir = dataset_dir / "valid" / "images"
            gt_dir = dataset_dir / "valid" / "labels"

        # Prepare Prompt
        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, 'r') as f: data = yaml.safe_load(f)
        names = data['names']
        if isinstance(names, dict): names = [names[k] for k in sorted(names.keys())]

        # Strict matching logic (Consolidated for Report Consistency)
        clean_names = [str(n).replace(".", "").replace("_", " ") for n in names]
        prompt = " . ".join(clean_names) + " ."

        # Run Inference
        save_dir = RESULTS_DIR / project_slug / "labels"
        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        for img_path in tqdm(image_paths, desc="Inference"):
            try:
                image_source, image = load_image(str(img_path))
                boxes, logits, phrases = predict(model, image, prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
                
                label_lines = []
                for box, phrase in zip(boxes, phrases):
                    phrase_clean = phrase.replace(".", "").strip().lower()
                    best_match_idx = -1
                    for idx, name in enumerate(clean_names):
                        name_clean = name.lower().strip()
                        if name_clean == phrase_clean or name_clean in phrase_clean:
                            best_match_idx = idx
                            break

                    if best_match_idx != -1:
                        line = f"{best_match_idx} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                        label_lines.append(line)

                with open(save_dir / f"{img_path.stem}.txt", "w") as f:
                    f.write("\n".join(label_lines))
            except Exception as e:
                pass # Skip corrupt images

        # Calculate Metrics (Before Deletion!)
        precision, recall, f1, tp, fp, fn = calculate_metrics(gt_dir, save_dir)
        print(f"   -> F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        writer.writerow({
            "dataset": project_slug,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn
        })
        csv_file.flush() # Save progress immediately

        # Save Visualizations (Before Deletion!)
        save_visualizations(image_dir, gt_dir, save_dir, names, VIS_DIR, project_slug)

        # Cleanup (The Streaming Part)
        print(f"[CLEANUP] Deleting {dataset_dir}")
        shutil.rmtree(dataset_dir)

    csv_file.close()
    print("Benchmark Completed.")

if __name__ == "__main__":
    main()
