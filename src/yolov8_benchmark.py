from pathlib import Path
from os import environ
import subprocess
import shutil
import csv
from roboflow import Roboflow
WORKSPACE = "roboflow-100"
ROOT = Path("/fs/scratch/PAS3162/siddiqui_159/rf100-benchmark")
BASE_OUT_DIR = ROOT / "data" / "rf100_yolov5"
MODELS_DIR = ROOT / "models" / "yolov8_stream"
RESULTS_DIR = ROOT / "results" / "yolov8_stream"
EXPORT_FORMAT = "yolov5"
RESULTS_CSV = RESULTS_DIR / "rf100_yolov8_results.csv"
RF100_DATASETS = [
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
    ("fish-market-ggjso", 4),
    ("flir-camera-objects", 1),
    ("furniture-ngpea", 1),
    ("gauge-u2lwv", 4),
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
    ("sedimentary-features-9eosf", 4),
    ("shark-teeth-5atku", 1),
    ("sign-language-sokdr", 1),
    ("signatures-xc8up", 1),
    ("smoke-uvylj", 1),
    ("soccer-players-5fuqs", 1),
    ("soda-bottles", 4),
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


def run_cmd(cmd: list, log_file: Path | None = None) -> int:
    """
    Run a shell command via subprocess. If log_file is given, redirect
    both stdout and stderr to that file.
    """
    print(f"[CMD] {' '.join(cmd)}")
    if log_file:
        with log_file.open("w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd)
    return result.returncode


def parse_map_from_results_csv(results_csv: Path) -> tuple[float | None, float | None]:
    """
    Parse mAP50 and mAP50-95 from YOLOv8's results.csv inside the run directory.

    Ultralytics YOLOv8 writes a CSV with columns like:
        metrics/mAP50(B), metrics/mAP50-95(B), ...

    We take the last row (final epoch).
    """
    if not results_csv.exists():
        print(f"[WARN] results.csv not found: {results_csv}")
        return None, None

    last_row = None
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    if last_row is None:
        print(f"[WARN] results.csv is empty: {results_csv}")
        return None, None

    try:
        map50 = last_row.get("metrics/mAP50(B)")
        map5095 = last_row.get("metrics/mAP50-95(B)")

        map50 = float(map50) if map50 not in (None, "") else None
        map5095 = float(map5095) if map5095 not in (None, "") else None

        return map50, map5095
    except Exception as e:
        print(f"[WARN] failed to parse mAP from {results_csv}: {e}")
        return None, None

def load_completed_datasets(csv_path: Path) -> set[str]:
    """
    Read existing CSV (if any) and return a set of dataset names already processed.
    """
    completed = set()
    if not csv_path.exists():
        return completed

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "dataset" in row:
                completed.add(row["dataset"])
    return completed


def append_result_row(csv_path: Path, row: dict):
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


def main():
    try:
        api_key = environ["ROBOFLOW_API_KEY"]
    except KeyError:
        raise SystemExit(
            "ERROR: ROBOFLOW_API_KEY not set.\n"
            'Export it first, e.g.\n\n'
            '  export ROBOFLOW_API_KEY="your_key_here"\n'
        )

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(WORKSPACE)

    completed = load_completed_datasets(RESULTS_CSV)
    print(f"[INFO] Already completed datasets in CSV: {len(completed)}")

    print(f"[INFO] Streaming RF100 through YOLOv8")
    print(f"[INFO] Workspace: {WORKSPACE}, export format: {EXPORT_FORMAT}")
    print(f"[INFO] Temporary dataset folder: {BASE_OUT_DIR}")
    print(f"[INFO] Models folder: {MODELS_DIR}")
    print(f"[INFO] Results CSV: {RESULTS_CSV}")
    print(f"[INFO] Total datasets: {len(RF100_DATASETS)}\n")

    for project_slug, version in RF100_DATASETS:
        if project_slug in completed:
            print(f"[SKIP] {project_slug} (already in results CSV)")
            continue

        dataset_dir = BASE_OUT_DIR / project_slug
        data_yaml = dataset_dir / "data.yaml"
        train_name = f"{project_slug}_train"
        model_run_dir = MODELS_DIR / train_name

        print("\n" + "=" * 80)
        print(f"[DATASET] {project_slug} (v{version})")
        print("=" * 80)

        if dataset_dir.exists():
            print(f"[INFO] Dataset folder already exists, reusing: {dataset_dir}")
        else:
            print(f"[INFO] Downloading {project_slug} to {dataset_dir}")
            try:
                project = ws.project(project_slug)
                project.version(version).download(
                    EXPORT_FORMAT,
                    location=str(dataset_dir),
                )
            except Exception as e:
                print(f"[ERROR] Failed to download {project_slug} v{version}: {e}")
                continue

        if not data_yaml.exists():
            print(f"[ERROR] data.yaml not found in {dataset_dir}, skipping.")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)
            continue

        print(f"[INFO] Training YOLOv8 on {project_slug}")
        train_cmd = [
            "yolo",
            "detect",
            "train",
            f"data={data_yaml}",
            "model=yolov8n.pt",
            "epochs=1",
            "batch=16",
            "imgsz=640",
            f"project={MODELS_DIR}",
            f"name={train_name}",
        ]
        rc_train = run_cmd(train_cmd)
        if rc_train != 0:
            print(f"[ERROR] Training failed for {project_slug} (rc={rc_train}), skipping.")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)
            continue

        best_weights = model_run_dir / "weights" / "best.pt"
        if not best_weights.exists():
            print(f"[ERROR] best.pt not found for {project_slug}, skipping.")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)
            if model_run_dir.exists():
                shutil.rmtree(model_run_dir, ignore_errors=True)
            continue

        results_csv_path = model_run_dir / "results.csv"
        map50, map5095 = parse_map_from_results_csv(results_csv_path)
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

        print(f"[CLEANUP] Removing model run dir: {model_run_dir}")
        if model_run_dir.exists():
            shutil.rmtree(model_run_dir, ignore_errors=True)

        print(f"[DONE] Finished {project_slug}")

    print("\n[ALL DONE] Streaming RF100 YOLOv8 benchmarking script finished.")


if __name__ == "__main__":
    main()
