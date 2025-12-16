#!/usr/bin/env python
"""
Benchmark Rex-Omni on the Roboflow-100 datasets.
"""

from __future__ import annotations

from pathlib import Path
from os import environ
import json
import csv
import shutil
from typing import List, Dict, Tuple

from roboflow import Roboflow
from tqdm import tqdm
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rex_omni import RexOmniWrapper

WORKSPACE = "roboflow-100"

ROOT = Path("/fs/scratch/PAS3162/siddiqui_159/rf100-benchmark")

BASE_OUT_DIR = ROOT / "data" / "rf100_coco"
RESULTS_DIR = ROOT / "results" / "rex_omni"

EXPORT_FORMAT = "coco"

RESULTS_CSV = RESULTS_DIR / "rf100_rex_omni_results.csv"

RF100_DATASETS: List[Tuple[str, int]] = [
    ("4-fold-defect", 1),
    ("abdomen-mri", 1),
    ("acl-x-ray", 1),
    ("activity-diagrams-qdobr", 1),
    ("aerial-cows", 1),
    ("aerial-pool", 1),
    ("aerial-spheres", 1),
    ("animals-ij5d2", 1),
    ("apex-videogame", 1),
    ("apples-fvpl5", 1),
    ("aquarium-qlnqy", 1),
    ("asbestos", 1),
    ("avatar-recognition-nuexe", 1),
    ("axial-mri", 1),
    ("bacteria-ptywi", 1),
    ("bccd-ouzjz", 1),
    ("bees-jt5in", 1),
    ("bone-fracture-7fylg", 1),
    ("brain-tumor-m2pbp", 1),
    ("cable-damage", 1),
    ("cables-nl42k", 1),
    ("cavity-rs0uf", 1),
    ("cell-towers", 1),
    ("cells-uyemf", 1),
    ("chess-pieces-mjzgj", 1),
    ("circuit-elements", 1),
    ("circuit-voltages", 1),
    ("cloud-types", 1),
    ("coins-1apki", 1),
    ("construction-safety-gsnvb", 1),
    ("coral-lwptl", 1),
    ("corrosion-bi3q3", 1),
    ("cotton-20xz5", 1),
    ("cotton-plant-disease", 1),
    ("csgo-videogame", 1),
    ("currency-v4f8j", 1),
    ("digits-t2eg6", 1),
    ("document-parts", 1),
    ("excavators-czvg9", 1),
    ("farcry6-videogame", 1),
    ("fish-market-ggjso", 4),           # fixed version
    ("flir-camera-objects", 1),
    ("furniture-ngpea", 1),
    ("gauge-u2lwv", 4),                 # fixed version
    ("grass-weeds", 1),
    ("gynecology-mri", 1),
    ("halo-infinite-angel-videogame", 1),
    ("hand-gestures-jps7z", 1),
    ("insects-mytwu", 1),
    ("leaf-disease-nsdsr", 1),
    ("lettuce-pallets", 1),
    ("liver-disease", 1),
    ("marbles", 1),
    ("mask-wearing-608pr", 1),
    ("mitosis-gjs3g", 1),
    ("number-ops", 1),
    ("paper-parts", 1),
    ("paragraphs-co84b", 1),
    ("parasites-1s07h", 1),
    ("peanuts-sd4kf", 1),
    ("peixos-fish", 1),
    ("people-in-paintings", 1),
    ("pests-2xlvx", 1),
    ("phages", 1),
    ("pills-sxdht", 1),
    ("poker-cards-cxcvz", 1),
    ("printed-circuit-board", 1),
    ("radio-signal", 1),
    ("road-signs-6ih4y", 1),
    ("road-traffic", 1),
    ("robomasters-285km", 1),
    ("secondary-chains", 1),
    ("sedimentary-features-9eosf", 4),  # fixed version
    ("shark-teeth-5atku", 1),
    ("sign-language-sokdr", 1),
    ("signatures-xc8up", 1),
    ("smoke-uvylj", 1),
    ("soccer-players-5fuqs", 1),
    ("soda-bottles", 4),                # fixed version
    ("solar-panels-taxvb", 1),
    ("stomata-cells", 1),
    ("street-work", 1),
    ("tabular-data-wf9uh", 1),
    ("team-fight-tactics", 1),
    ("thermal-cheetah-my4dp", 1),
    ("thermal-dogs-and-people-x6ejw", 1),
    ("trail-camera", 1),
    ("truck-movement", 1),
    ("tweeter-posts", 1),
    ("tweeter-profile", 1),
    ("underwater-objects-5v7p8", 1),
    ("underwater-pipes-4ng4t", 1),
    ("uno-deck", 1),
    ("valentines-chocolate", 1),
    ("vehicles-q0x2v", 1),
    ("wall-damage", 1),
    ("washroom-rf1fa", 1),
    ("weed-crop-aerial", 1),
    ("wine-labels", 1),
    ("x-ray-rheumatology", 1),
]


_REX_MODEL: RexOmniWrapper | None = None
_CURRENT_CATEGORIES: List[str] | None = None



def load_completed_datasets(csv_path: Path) -> set[str]:
    """
    Read existing CSV (if any) and return a set of dataset names already processed.
    """
    completed: set[str] = set()
    if not csv_path.exists():
        return completed

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "dataset" in row and row["dataset"]:
                completed.add(row["dataset"])
    return completed


def append_result_row(csv_path: Path, row: dict) -> None:
    """
    Append one row to the results CSV, writing header if the file does not exist.
    """
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "version", "map50", "map50_95"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)



def _get_rex_model() -> RexOmniWrapper:
    """
    Initialize and return a singleton Rex-Omni model wrapper.
    """
    global _REX_MODEL

    if _REX_MODEL is not None:
        return _REX_MODEL

    model_path = environ.get("REX_OMNI_MODEL_PATH", "IDEA-Research/Rex-Omni")
    backend = environ.get("REX_OMNI_BACKEND", "transformers")

    max_tokens = int(environ.get("REX_OMNI_MAX_TOKENS", "2048"))
    temperature = float(environ.get("REX_OMNI_TEMPERATURE", "0.0"))
    top_p = float(environ.get("REX_OMNI_TOP_P", "0.05"))
    top_k = int(environ.get("REX_OMNI_TOP_K", "1"))
    repetition_penalty = float(environ.get("REX_OMNI_REP_PENALTY", "1.05"))

    print(f"[INFO] Initializing Rex-Omni model: {model_path} (backend={backend})")
    _REX_MODEL = RexOmniWrapper(
        model_path=model_path,
        backend=backend,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    return _REX_MODEL


def run_rex_omni_on_image(image_path: Path) -> List[Dict]:
    """
    Run Rex-Omni on a single image and return detections.

    RETURNS a list of dicts in the format:
        {
            "bbox": [x, y, w, h],      # pixel coordinates, COCO format
            "score": float,            # confidence score
            "category_name": str,      # class name as string, must match COCO 'name'
        }
    """
    global _CURRENT_CATEGORIES

    if _CURRENT_CATEGORIES is None:
        raise RuntimeError(
            "Rex-Omni category list is not set. "
            "Make sure `_CURRENT_CATEGORIES` is initialized from the COCO annotations "
            "before calling run_rex_omni_on_image()."
        )

    rex = _get_rex_model()

    image = Image.open(image_path).convert("RGB")

    try:
        results = rex.inference(
            images=image,
            task="detection",
            categories=_CURRENT_CATEGORIES,
        )
    except Exception as e:
        print(f"[WARN] Rex-Omni inference error on {image_path}: {e}")
        return []

    if not results:
        return []

    result = results[0]

    if isinstance(result, dict) and not result.get("success", True):
        err = result.get("error", "Unknown Rex-Omni error")
        print(f"[WARN] Rex-Omni reported failure on {image_path}: {err}")
        return []

    extracted = result.get("extracted_predictions", {})
    detections: List[Dict] = []

    if not isinstance(extracted, dict):
        print(f"[WARN] Unexpected Rex-Omni predictions format for {image_path}: {type(extracted)}")
        return []

    for category_name, preds in extracted.items():
        if not isinstance(preds, list):
            continue

        for pred in preds:
            if not isinstance(pred, dict):
                continue
            if pred.get("type") != "box":
                continue

            coords = pred.get("coords")
            if not coords or len(coords) != 4:
                continue

            try:
                x0, y0, x1, y1 = map(float, coords)
            except Exception:
                continue

            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            if w <= 0 or h <= 0:
                continue

            score = float(pred.get("score", 1.0))

            detections.append(
                {
                    "bbox": [x0, y0, w, h],
                    "score": score,
                    "category_name": str(category_name),
                }
            )

    return detections


def evaluate_dataset_with_rex(valid_images_dir: Path, valid_ann_path: Path) -> Tuple[float | None, float | None]:
    """
    Run Rex-Omni on every image in the validation set and compute mAP50 / mAP50-95.
    """
    if not valid_ann_path.exists():
        print(f"[ERROR] COCO annotations not found: {valid_ann_path}")
        return None, None

    coco_gt = COCO(str(valid_ann_path))

    imgid_by_fname: Dict[str, int] = {}
    for img in coco_gt.dataset["images"]:
        imgid_by_fname[img["file_name"]] = img["id"]

    catid_by_name: Dict[str, int] = {}
    global _CURRENT_CATEGORIES
    categories_for_rex: List[str] = []
    for cat in coco_gt.dataset["categories"]:
        name = cat["name"]
        categories_for_rex.append(name)
        catid_by_name[name] = cat["id"]
    _CURRENT_CATEGORIES = categories_for_rex

    detections: List[Dict] = []

    image_files = sorted(
        [p for p in valid_images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    print(f"[INFO] Valid images: {len(image_files)}")

    for img_path in tqdm(image_files, desc="Rex-Omni inference"):
        file_name = img_path.name
        if file_name not in imgid_by_fname:
            continue
        image_id = imgid_by_fname[file_name]

        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            print(f"[WARN] Failed to open image {img_path}: {e}")
            continue

        try:
            preds = run_rex_omni_on_image(img_path)
        except NotImplementedError:
            raise
        except Exception as e:
            print(f"[WARN] Rex-Omni inference failed for {img_path}: {e}")
            continue

        for det in preds:
            bbox = det.get("bbox")
            score = det.get("score")
            cat_name = det.get("category_name")

            if bbox is None or score is None or cat_name is None:
                continue

            cat_id = catid_by_name.get(cat_name)
            if cat_id is None:
                continue

            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cat_id),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "score": float(score),
                }
            )

    if not detections:
        print("[WARN] No detections collected; returning None metrics.")
        return None, None

    coco_dt = coco_gt.loadRes(detections)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map5095 = float(coco_eval.stats[0])
    map50 = float(coco_eval.stats[1])

    return map50, map5095


def main() -> None:
    try:
        api_key = environ["ROBOFLOW_API_KEY"]
    except KeyError:
        raise SystemExit(
            "ERROR: ROBOFLOW_API_KEY not set.\n"
            'Export it first, e.g.\n\n'
            '  export ROBOFLOW_API_KEY="your_key_here"\n'
        )

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(WORKSPACE)

    completed = load_completed_datasets(RESULTS_CSV)
    print(f"[INFO] Already completed datasets in CSV: {len(completed)}")

    print(f"[INFO] Streaming RF100 through Rex-Omni (COCO export)")
    print(f"[INFO] Workspace: {WORKSPACE}, export format: {EXPORT_FORMAT}")
    print(f"[INFO] Temporary dataset folder: {BASE_OUT_DIR}")
    print(f"[INFO] Results CSV: {RESULTS_CSV}")
    print(f"[INFO] Total datasets: {len(RF100_DATASETS)}\n")

    for project_slug, version in RF100_DATASETS:
        if project_slug in completed:
            print(f"[SKIP] {project_slug} (already in results CSV)")
            continue

        dataset_dir = BASE_OUT_DIR / project_slug
        valid_dir = dataset_dir / "valid"
        valid_ann = valid_dir / "_annotations.coco.json"

        print("\n" + "=" * 80)
        print(f"[DATASET] {project_slug} (v{version})")
        print("=" * 80)

        if dataset_dir.exists():
            print(f"[INFO] Dataset folder already exists, reusing: {dataset_dir}")
        else:
            print(f"[INFO] Downloading {project_slug} (v{version}) to {dataset_dir}")
            try:
                project = ws.project(project_slug)
                project.version(version).download(
                    EXPORT_FORMAT,
                    location=str(dataset_dir),
                )
            except Exception as e:
                print(f"[ERROR] Failed to download {project_slug} v{version}: {e}")
                if dataset_dir.exists():
                    shutil.rmtree(dataset_dir, ignore_errors=True)
                continue

        if not valid_dir.exists() or not valid_ann.exists():
            print(f"[ERROR] Validation data not found for {project_slug}.")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)
            continue

        print(f"[INFO] Running Rex-Omni on validation split for {project_slug}")
        try:
            map50, map5095 = evaluate_dataset_with_rex(valid_dir, valid_ann)
        except Exception as e:
            print(f"[ERROR] Evaluation failed for {project_slug}: {e}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)
            continue

        print(f"[METRICS] {project_slug}: mAP50={map50}, mAP50-95={map5095}")

        append_result_row(
            RESULTS_CSV,
            {
                "dataset": project_slug,
                "version": version,
                "map50": map50 if map50 is not None else "",
                "map50_95": map5095 if map5095 is not None else "",
            },
        )

        print(f"[CLEANUP] Removing dataset dir: {dataset_dir}")
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)

        print(f"[DONE] Finished {project_slug}")

    print("\n[ALL DONE] RF100 Rex-Omni benchmarking script finished.")


if __name__ == "__main__":
    main()
