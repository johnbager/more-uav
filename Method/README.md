# Method

This folder contains the runnable evaluation and baseline inference structure for **MoRe-UAV**.

## Layout

- `evaluate.sh`
- `metric.py`
- `common.py`
- `more_uav/`
- `qwen/`
- `CPM/`

## Included Implementations

- `more_uav/`: reference MoRe-UAV method with a frozen Qwen2.5-VL backbone, MPA, MVA, and training / prediction scripts
- `qwen/`: direct grounding baseline built on `Qwen/Qwen2.5-VL-7B-Instruct`
- `CPM/`: direct grounding baseline built on `openbmb/MiniCPM-V-2_6`

All released implementations read one split from the dataset, export predictions as JSON, and can be evaluated with the shared metric script.

The Qwen baseline follows the official `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` loading path.

The MiniCPM baseline follows the official `AutoModel.from_pretrained(..., trust_remote_code=True)` + `model.chat(...)` loading path.

The `more_uav/` package implements the paper-aligned reference method on top of the Qwen2.5-VL backbone with parameter-efficient tuning.

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

- `expression.json`
- one folder per sample, such as `case_00000001/`
- `images/`
- `bboxes.json`

The loaders in this folder first read the split-level `expression.json`. If it is absent, they fall back to case-level `expression.txt`.

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

## Train MoRe-UAV Method

```bash
pip install -r Method/more_uav/requirements.txt
bash Method/more_uav/run_train.sh ./dataset ./checkpoints/more_uav_qwen
```

## Predict With MoRe-UAV Method

```bash
bash Method/more_uav/run_predict.sh ./dataset val ./checkpoints/more_uav_qwen/best ./outputs/more_uav_val_predictions.json
```

## Evaluate Predictions

```bash
bash Method/evaluate.sh val ./dataset ./outputs/more_uav_val_predictions.json
```
