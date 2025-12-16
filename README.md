# Benchmarking Classic vs. Open-Vocabulary Detectors (RF100)

**Authors:** Sarim Siddiqui, Adam Haikal, Raghav Kasibhatla
**Course:** Computer Vision - Final Project
**Due Date:** 12/17/2025

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/ragkasi/RF100-Benchmark/blob/main/demo.ipynb)

## 1. Project Overview
This project benchmarks the performance of **Supervised Learning (YOLOv8)** against **Open-Vocabulary Foundation Models (Grounding DINO)** across the **Roboflow 100 (RF100)** dataset. We evaluated performance across 56 diverse domains (Aerial, Medical, Gaming, etc.) to analyze the trade-off between the "annotation tax" of supervised models and the "generalization gap" of zero-shot models.

**Advanced Algorithm Submitted:** Grounding DINO (Open-Set Object Detector).
**Key Contribution:** A comparative analysis showing where Zero-Shot models fail (Medical/Microscopic) vs where they succeed (Common Objects).

## 2. Repository Structure & Description

- **`demo.ipynb`**
  [MAIN] Runnable Code. This Jupyter Notebook implements the Grounding DINO inference pipeline. It is designed to run in Google Colab (T4 GPU).

- **`samples/`**
  [DATA] Contains validation images representing three distinct domains:
  1. `aerial_demo.jpg` (Aerial View)
  2. `game_demo.jpg` (Synthetic/Gaming)
  3. `medical_demo.jpg` (Medical Imaging)

- **`src/`**
  [SOURCE] The complete evaluation scripts used on the Ohio Supercomputer Center (OSC):
  - `yolov8_benchmark.py`: Runs the supervised training loop.
  - `groundingdino_streaming.py`: Runs the Zero-Shot inference loop.
  - `analyze_results.py`: Generates the comparative plots and statistics.

- **`requirements.txt`**
  [ENV] List of Python dependencies for local execution.

  ## 3. Installation & Usage Instructions

**Note to Graders:** The "Advanced Algorithm" (Grounding DINO) requires specific CUDA kernels that are difficult to compile on local machines. **We strongly recommend using the Google Colab link below**, which pre-installs the correct environment automatically.

### **Option A: Google Colab (Recommended)**

1.  **Open the Notebook:**
    Click the badge at the top or [Click Here](https://github.com/ragkasi/RF100-Benchmark/blob/main/demo.ipynb).

2.  **Setup Runtime:**
    - Go to `Runtime` -> `Change runtime type`.
    - Select **T4 GPU** (Required for Grounding DINO).

3.  **Upload Test Examples:**
    - Download the `samples/` folder from this repository to your computer.
    - In Colab, click the **Folder Icon** (Files) on the left sidebar.
    - Drag and drop the images (e.g., `aerial_demo.jpg`) into the Colab file space.

4.  **Run the Algorithm:**
    - Run **Cell 1** to install the environment (Grounding DINO, Supervision, PyTorch).
    - Run **Cell 2 & 3** to load the model and define inference logic.
    - Run **Cell 4** to generate the bounding box predictions on the test examples.

### **Option B: Local Installation**

If you prefer to run locally, you require `Python 3.9+` and a `CUDA-enabled GPU`.

```bash
# 1. Clone the repository
git clone [https://github.com/ragkasi/RF100-Benchmark.git](https://github.com/ragkasi/RF100-Benchmark.git)
cd RF100-Benchmark

# 2. Install dependencies
# Note: Ensure your NVIDIA drivers match your PyTorch CUDA version.
pip install -r requirements.txt

# 3. Run the Jupyter Notebook
jupyter notebook demo.ipynb