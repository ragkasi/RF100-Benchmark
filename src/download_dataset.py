from pathlib import Path
from os import environ

from roboflow import Roboflow

WORKSPACE = "roboflow-100"

BASE_OUT_DIR = Path(
    "/fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark/data/rf100_yolov5"
)

EXPORT_FORMAT = "yolov5"

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
    ("fish-market-ggjso", 1),
    ("flir-camera-objects", 1),
    ("furniture-ngpea", 1),
    ("gauge-u2lwv", 1),
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
    ("sedimentary-features-9eosf", 1),
    ("shark-teeth-5atku", 1),
    ("sign-language-sokdr", 1),
    ("signatures-xc8up", 1),
    ("smoke-uvylj", 1),
    ("soccer-players-5fuqs", 1),
    ("soda-bottles", 1),
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

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(WORKSPACE)

    print(f"[INFO] Downloading RF100 to: {BASE_OUT_DIR}")
    print(f"[INFO] Workspace: {WORKSPACE}, format: {EXPORT_FORMAT}")
    print(f"[INFO] Total datasets: {len(RF100_DATASETS)}\n")

    for project_slug, version in RF100_DATASETS:
        out_dir = BASE_OUT_DIR / project_slug

        if out_dir.exists():
            print(f"[SKIP] {project_slug} (folder already exists: {out_dir})")
            continue

        print(f"[INFO] -> {project_slug} (v{version})")

        try:
            project = ws.project(project_slug)
            project.version(version).download(
                EXPORT_FORMAT,
                location=str(out_dir),
            )
        except Exception as e:
            print(f"[ERROR] Failed to download {project_slug} v{version}: {e}")
            continue

        print(f"[OK]   {project_slug} downloaded to {out_dir}\n")

    print("[DONE] RF100 download script finished.")


if __name__ == "__main__":
    main()
