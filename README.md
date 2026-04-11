# IndoorLocalization

A visual place recognition (VPR) pipeline for indoor localization using the [NYC-Indoor-VPR](https://huggingface.co/datasets/ai4ce/NYC-Indoor-VPR-Data) dataset. This project investigates whether a purely visual, infrastructure-free approach — using only a smartphone camera and a state-of-the-art 3D vision model — can localize a user inside a building without any specialized hardware such as BLE beacons or WiFi fingerprinting.

We benchmark three retrieval conditions using DINOv2 and VGGT (Visual Geometry Grounded Transformer), evaluated on recall@K metrics across multiple indoor scenes in New York City.

## Problem Statement

GPS fails indoors. Existing solutions require expensive pre-installed infrastructure. Our pipeline asks: given query images from a smartphone, can VGGT's camera pose estimation accurately localize a user inside a known building using only visual information?

## Dataset

**NYC-Indoor-VPR** (~38 GB) — [ai4ce/NYC-Indoor-VPR-Data](https://huggingface.co/datasets/ai4ce/NYC-Indoor-VPR-Data)

The dataset contains over 36,000 images from 13 crowded indoor scenes (e.g., World Trade Center) captured under varying lighting conditions. It is split into `train`, `val`, and `test` sets, each with `database` and `query` subsets.

> The dataset is downloaded automatically when the notebook runs — do not commit it to the repository.

## Methods

Three experimental conditions are compared:

| Condition | Description |
|-----------|-------------|
| **A — DINOv2 Zero-Shot** | `facebook/dinov2-base` embeds all images; FAISS cosine similarity retrieves top-K candidates |
| **B — VGGT Geometric Re-Ranking** | DINOv2 candidates are re-ranked using `facebook/VGGT-1B` geometric scores derived from camera pose alignment |
| **C — Fine-Tuned VGGT Re-Ranking** | VGGT fine-tuned with triplet margin loss on indoor training pairs, then used for re-ranking |

Recall is measured at K ∈ {1, 5, 10, 25} and distance thresholds of 3m, 5m, and 10m.

## Results

Evaluated on 1,880 test queries across 9 indoor scenes.

| Condition | R@1 | R@5 | R@10 | R@25 |
|-----------|-----|-----|------|------|
| **A — DINOv2 Zero-Shot** (≤3m) | 7.3% | 26.9% | 38.2% | 54.0% |
| **A — DINOv2 Zero-Shot** (≤5m) | 10.7% | 38.1% | 52.3% | 72.2% |
| **A — DINOv2 Zero-Shot** (≤10m) | 17.5% | 54.8% | 72.7% | 91.3% |
| **B — VGGT Re-Ranked** (≤5m) | 11.1% | 36.6% | 52.3% | 72.2% |
| **C — VGGT Fine-Tuned** (≤5m) | **12.1%** | 37.4% | 52.3% | **72.2%** |

All three conditions converge at R@25 = 72.2% at 5m, indicating the bottleneck is top-1 ranking precision rather than candidate coverage. VGGT re-ranking adds ~2.15s latency per query (31× overhead over retrieval alone).

## Repository Structure

```
IndoorLocalization/
├── indoor_localization.ipynb   # Main notebook (all experiments)
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

Runtime-generated folders (`results/`, `visualizations/`, `models/`, `data/`) are excluded from version control via `.gitignore`.

## Setup and Usage

This project is designed to run on **Google Colab** with an A100 GPU.

1. Open `indoor_localization.ipynb` in Google Colab
2. Set runtime to **A100 GPU**: Runtime > Change runtime type > A100
3. Run **Cell 0 once** to install dependencies — the kernel will restart automatically, then skip back to Cell 1
4. Run cells sequentially — the dataset (~38 GB) downloads automatically on first run

## Requirements

- Python 3.10+
- CUDA GPU (A100 strongly recommended)

Key dependencies (see `requirements.txt`):

```
torch
torchvision
transformers
faiss-gpu
Pillow
pandas
numpy
matplotlib
tqdm
huggingface_hub
```

## Team Contributions

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **Hyeonu (Eric) Kim** | TBD | TBD |
| **Jacky Chen** | TBD | TBD |
| **Henry Chen** | TBD | TBD |
