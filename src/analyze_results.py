import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- CONFIGURATION ---
# Output Directory
OUTPUT_DIR = Path("/fs/scratch/PAS3162/ragkasi/final_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
PATH_DINO = Path("/fs/scratch/PAS3162/ragkasi/groundingdino_metrics.csv")
PATH_YOLO = Path("/fs/scratch/PAS3162/siddiqui_159/rf100-benchmark/results/yolov8_stream/rf100_yolov8_results.csv")
PATH_REX  = Path("/fs/scratch/PAS3162/siddiqui_159/rf100-benchmark/results/rex_omni/rf100_rex_omni_results.csv")

def load_and_prep():
    print("Loading datasets...")
    
    # Load Grounding DINO (Metric: F1)
    if not PATH_DINO.exists():
        print(f"ERROR: DINO file not found at {PATH_DINO}")
        return None
    
    df_dino = pd.read_csv(PATH_DINO)
    # Standardize columns
    df_dino = df_dino[['dataset', 'f1']].rename(columns={'f1': 'DINO_F1'})
    # Load YOLOv8 (Metric: mAP50)

    if not PATH_YOLO.exists():
        print(f"ERROR: YOLO file not found at {PATH_YOLO}")
        return None
    
    df_yolo = pd.read_csv(PATH_YOLO)
    df_yolo = df_yolo[['dataset', 'map50']].rename(columns={'map50': 'YOLO_mAP'})

    # Load Rex Omni (Metric: Check columns)
    if not PATH_REX.exists():
        print(f"ERROR: Rex Omni file not found at {PATH_REX}")
        return None

    df_rex = pd.read_csv(PATH_REX)
    print(f"Rex Omni columns detected: {df_rex.columns.tolist()}")

    # dynamic column finder for Rex
    rex_metric_col = None
    for col in ['map50', 'mAP', 'accuracy', 'score', 'f1']:
        if col in df_rex.columns:
            rex_metric_col = col
            break

    if rex_metric_col:
        df_rex = df_rex[['dataset', rex_metric_col]].rename(columns={rex_metric_col: 'REX_Score'})

    else:
        # Fallback: assume the second column is the score if not named standardly
        print("WARNING: Could not identify Rex metric column by name. Using 2nd column.")
        df_rex = df_rex.iloc[:, [0, 1]]
        df_rex.columns = ['dataset', 'REX_Score']

    # --- MERGE (INTERSECTION ONLY) ---
    print("\nMerging data...")
    print(f"Counts before merge -> DINO: {len(df_dino)}, YOLO: {len(df_yolo)}, REX: {len(df_rex)}")

    # Merge DINO and YOLO (Inner Join)
    merged = pd.merge(df_dino, df_yolo, on='dataset', how='inner')

    # Merge result with REX (Inner Join)
    merged = pd.merge(merged, df_rex, on='dataset', how='inner')

    print(f"Counts after merge  -> Common Datasets: {len(merged)}")

    if len(merged) == 0:
        print("CRITICAL ERROR: No common datasets found. Check dataset naming conventions (e.g., '4-fold-defect' vs '4_fold_defect').")
        return None

    return merged

def generate_plots(df):
    sns.set_theme(style="whitegrid")
    num_datasets = len(df)

    # BAR CHART: Average Performance (Fair Comparison)
    plt.figure(figsize=(10, 6))

    # Calculate means
    means = df[['DINO_F1', 'YOLO_mAP', 'REX_Score']].mean().reset_index()
    means.columns = ['Model', 'Score']

    sns.barplot(data=means, x='Model', y='Score', palette="viridis")
    plt.title(f"Average Performance on Common Datasets (N={num_datasets})")
    plt.ylim(0, 1.0)

    # Add number labels on bars
    for index, row in means.iterrows():
        plt.text(index, row.Score + 0.02, f"{row.Score:.3f}", color='black', ha="center")

    plt.savefig(OUTPUT_DIR / "fair_comparison_bar.png")
    plt.close()

    # SCATTER PLOT: YOLO vs DINO (Colored by Rex Performance)
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=df, 
        x='YOLO_mAP', 
        y='DINO_F1', 
        size='REX_Score', 
        hue='REX_Score',
        sizes=(20, 200),
        palette="magma",
        alpha=0.8
    )

    # Draw y=x line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label="DINO = YOLO")
    plt.title(f"Model Correlation (Size/Color = Rex Score)\nN={num_datasets} Common Datasets")
    plt.xlabel("Supervised (YOLOv8) mAP")
    plt.ylabel("Zero-Shot (DINO) F1")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_3d_comparison.png")
    plt.close()

    # TOP DINO WINS (Relative to YOLO)
    # We only care about wins in the common subset now
    df['DINO_Gap'] = df['DINO_F1'] - df['YOLO_mAP']
    top_wins = df.sort_values('DINO_Gap', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_wins, x='dataset', y='DINO_Gap', palette='Greens_r')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top Domains where DINO outperforms YOLO (Subset N={num_datasets})")
    plt.ylabel("Performance Gap (DINO - YOLO)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bar_dino_wins_subset.png")
    plt.close()

def main():
    df = load_and_prep()
    if df is None:
        return

    # Save the Fair Master CSV
    master_csv_path = OUTPUT_DIR / "rf100_common_intersection.csv"
    df.to_csv(master_csv_path, index=False)
    print(f"\nSUCCESS: Intersection data saved to {master_csv_path}")

    # Generate Plots
    print("Generating plots...")
    generate_plots(df)
    print(f"Plots saved to {OUTPUT_DIR}")
    
    # Print Text Summary

    print("\n=== FINAL SUMMARY STATISTICS (INTERSECTION ONLY) ===")
    print(df.describe())
    print("\n=== TOP 3 OVERALL WINNERS ===")

    # Who had the highest single score on any dataset?
    print(df.sort_values('DINO_F1', ascending=False)[['dataset', 'DINO_F1']].head(1))
    print(df.sort_values('YOLO_mAP', ascending=False)[['dataset', 'YOLO_mAP']].head(1))
    print(df.sort_values('REX_Score', ascending=False)[['dataset', 'REX_Score']].head(1))

if __name__ == "__main__":
    main()
