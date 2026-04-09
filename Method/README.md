# Method

This folder contains the runnable evaluation and baseline inference structure for **MoRe-UAV**.

## Layout

- `evaluate.sh`
- `metric.py`
- `common.py`
- `qwen/`
- `CPM/`

## Included Baselines

- `qwen/`: direct grounding baseline built on `Qwen/Qwen2.5-VL-7B-Instruct`
- `CPM/`: direct grounding baseline built on `openbmb/MiniCPM-V-2_6`

Both baselines read one split from the released dataset, run multimodal grounding case by case, export predictions as JSON, and can be evaluated with the shared metric script.

The Qwen baseline follows the official `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` loading path.

The MiniCPM baseline follows the official `AutoModel.from_pretrained(..., trust_remote_code=True)` + `model.chat(...)` loading path.

## Evaluation Metrics

The benchmark reports:

- `mIoU`
- `Acc@0.5`
- `Norm Precision`
- `IoU AUC`

## Dataset Assumption

The dataset root is expected to follow:

```text
./dataset/
├── train/
├── val/
└── test/
```

Each split should contain:

- one folder per sample, such as `case_00000001/`
- `images/`
- `bboxes.json`
- `expression.txt`

The loaders in this folder read the referring expression from each case-level `expression.txt` file.

## Run Qwen Baseline

```bash
pip install -r Method/qwen/requirements.txt
bash Method/qwen/run.sh val ./dataset ./outputs/qwen_val_predictions.json
```

## Run MiniCPM Baseline

```bash
pip install -r Method/CPM/requirements.txt
bash Method/CPM/run.sh val ./dataset ./outputs/cpm_val_predictions.json
```

## Evaluate Predictions

```bash
bash Method/evaluate.sh val ./dataset ./outputs/qwen_val_predictions.json
```
