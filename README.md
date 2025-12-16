# Benchmarking Classic vs. Open-Vocabulary Detectors (RF100)

**Authors:** Sarim Siddiqui, Adam Haikal, Raghav Kasibhatla  
**Course:** Computer Vision - Final Project  

## Project Overview
This project benchmarks the performance of **Supervised Learning (YOLOv8)** against **Open-Vocabulary Foundation Models (Grounding DINO, Rex Omni)** across the **Roboflow 100 (RF100)** dataset. We evaluated performance across 56 diverse domains (Aerial, Medical, Gaming, etc.) to analyze the trade-off between the "annotation tax" of supervised models and the "generalization gap" of zero-shot models.

## Repository Structure
* `demo.ipynb`: A Jupyter Notebook (Google Colab compatible) that demonstrates the Zero-Shot detection pipeline on sample images.
* `samples/`: Sample images from the RF100 dataset for testing.
* `models/`: Contains a sample fine-tuned YOLOv8 model (best.pt) from our training runs.
* `src/`: The complete benchmarking scripts used on the Ohio Supercomputer Center (OSC).

## Quick Start (Run the Demo)
To verify our "Advanced Algorithm" (Grounding DINO Inference) and view results:

1.  Open `demo.ipynb` in GitHub.
2.  Click the **"Open in Colab"** badge (or download and upload to Google Colab).
3.  Connect to a GPU Runtime (Runtime > Change runtime type > T4 GPU).
4.  Upload the images from the `samples/` folder to the Colab file explorer.
5.  Run all cells to see Grounding DINO detect objects in Aerial, Game, and Medical domains.

## Installation (Local)
If you prefer to run locally, you will need `python 3.9+` and `CUDA 11.x`.

```bash
# Clone repository
git clone https://github.com/ragkasi/RF100-Benchmark.git
cd RF100-Benchmark

# Install dependencies
pip install -r requirements.txt