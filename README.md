# MoRe-UAV

This repository hosts the official resources for **MoRe-UAV: A Large-Scale Benchmark for Motion-Aware Visual Grounding in UAV Videos**.

## Repository Structure

- `Main/`: primary release materials and project resources.
- `Method/`: evaluation scripts and baseline folder structure.

## Dataset Download

The full dataset is currently available through Google Drive:

- **Google Drive**: [Download Link](https://drive.google.com/drive/folders/1KDKjIvxHtZYHHxQxKrxzxs368npEc1Y1?usp=drive_link)

To help reviewers quickly download and verify the dataset format and quality, we also provide a small-scale dataset for reviewer access:

- **Reviewer Subset (Google Drive)**: [Download Link](https://drive.google.com/drive/folders/1HOMfquXIHSN5HpVI8LBhAlIzoX73feLV?usp=drive_link)

## Dataset Format

The released dataset is organized into `train/`, `val/`, and `test/` splits.

```text
dataset/
├── train/
│   ├── expression.json
│   ├── case_00000001/
│   │   ├── images/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   └── bboxes.json
│   └── ...
├── val/
│   ├── expression.json
│   └── ...
└── test/
    ├── expression.json
    └── ...
```

Each split contains one `expression.json` file for the cases in that split.

Each case folder contains:

- `images/` with ordered video frames
- `bboxes.json` with frame-level target annotations for the clip
