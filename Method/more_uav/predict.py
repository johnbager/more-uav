from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data import context_window_indices, fill_missing_boxes, load_case_records
from model import load_qwen_from_checkpoint


def denormalize_box(box, width: int, height: int):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return [
        int(round(max(0.0, min(1.0, x1)) * width)),
        int(round(max(0.0, min(1.0, y1)) * height)),
        int(round(max(0.0, min(1.0, x2)) * width)),
        int(round(max(0.0, min(1.0, y2)) * height)),
    ]


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    model = load_qwen_from_checkpoint(args.model_name, args.checkpoint_dir, args.gradient_checkpointing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    records = load_case_records(args.dataset_root, args.split, args.max_cases)
    predictions = {}

    for record in tqdm(records, desc="predict"):
        case_predictions = {}
        for target_index in range(0, len(record.frame_paths), max(1, args.frame_stride)):
            window_indices = context_window_indices(len(record.frame_paths), target_index, args.window_size)
            window_paths = [str(record.frame_paths[index]) for index in window_indices]
            outputs = model([record.expression], [window_paths])
            predicted_box = outputs["pred_boxes"][0, -1].detach().cpu().tolist()
            visibility_logit = outputs["visibility_logits"][0, -1].item()
            box = denormalize_box(predicted_box, record.width, record.height) if visibility_logit >= 0.0 else None
            case_predictions[record.frame_paths[target_index].stem] = box
        case_predictions = fill_missing_boxes(record.frame_paths, case_predictions)
        predictions[record.sample_id] = case_predictions

    save_json(
        Path(args.output),
        {
            "model": args.model_name,
            "split": args.split,
            "checkpoint_dir": args.checkpoint_dir,
            "predictions": predictions,
        },
    )


if __name__ == "__main__":
    main()
