from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import candidate_case_names, load_json, load_samples


def parse_box(value):
    if value is None or value == []:
        return None
    if isinstance(value, dict):
        if "xyxy" in value:
            value = value["xyxy"]
        elif "bbox" in value:
            value = value["bbox"]
        elif all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
        else:
            return None
    if not isinstance(value, list) or len(value) != 4:
        return None
    x1, y1, x2, y2 = [float(item) for item in value]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def iou(box_a, box_b):
    if box_a is None and box_b is None:
        return 1.0
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def normalized_error(box_a, box_b):
    if box_a is None and box_b is None:
        return 0.0
    if box_a is None or box_b is None:
        return 1.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    aw = max(ax2 - ax1, 1e-6)
    ah = max(ay2 - ay1, 1e-6)
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return max(abs(acx - bcx) / aw, abs(acy - bcy) / ah)


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def threshold_auc(values, thresholds, lower_is_better):
    scores = []
    for threshold in thresholds:
        if lower_is_better:
            scores.append(sum(value <= threshold for value in values) / len(values) if values else 0.0)
        else:
            scores.append(sum(value >= threshold for value in values) / len(values) if values else 0.0)
    return mean(scores)


def make_lookup(raw_boxes):
    lookup = {}
    if isinstance(raw_boxes, dict):
        for key, value in raw_boxes.items():
            name = Path(str(key)).stem
            box = parse_box(value)
            lookup[name] = box
            if name.isdigit():
                lookup[str(int(name))] = box
    elif isinstance(raw_boxes, list):
        for index, value in enumerate(raw_boxes):
            box = parse_box(value)
            lookup[str(index)] = box
            lookup[str(index + 1)] = box
            lookup[f"{index:06d}"] = box
            lookup[f"{index + 1:06d}"] = box
    return lookup


def align_boxes(raw_boxes, frame_stems):
    aligned = {}
    lookup = make_lookup(raw_boxes)
    for frame_stem in frame_stems:
        box = None
        aliases = [frame_stem]
        if frame_stem.isdigit():
            number = int(frame_stem)
            aliases.extend([str(number), f"{number:06d}"])
            if number > 0:
                aliases.extend([str(number - 1), f"{number - 1:06d}"])
        for alias in aliases:
            if alias in lookup:
                box = lookup[alias]
                break
        aligned[frame_stem] = box
    return aligned


def load_ground_truth(gt_root: Path, split: str):
    samples = load_samples(gt_root, split)
    ground_truth = {}
    for sample in samples:
        bboxes_path = sample.case_dir / "bboxes.json"
        if not bboxes_path.exists():
            continue
        raw_boxes = load_json(bboxes_path)
        frame_stems = [path.stem for path in sample.frame_paths]
        ground_truth[sample.sample_id] = align_boxes(raw_boxes, frame_stems)
    return ground_truth


def normalize_prediction_records(raw):
    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]
    if isinstance(raw, list):
        records = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            sample_id = item.get("id") or item.get("sample_id") or item.get("case_id")
            if sample_id is None:
                continue
            records[str(sample_id)] = item.get("boxes") or item.get("bboxes") or item.get("predictions") or {}
        return records
    if isinstance(raw, dict):
        return raw
    return {}


def find_prediction_entry(records, sample_id):
    for candidate in candidate_case_names(sample_id):
        if candidate in records:
            value = records[candidate]
            if isinstance(value, dict) and ("boxes" in value or "bboxes" in value):
                return value.get("boxes") or value.get("bboxes") or {}
            return value
    return {}


def load_predictions(pred_file: Path, ground_truth):
    raw = load_json(pred_file)
    records = normalize_prediction_records(raw)
    predictions = {}
    for sample_id, gt_boxes in ground_truth.items():
        raw_boxes = find_prediction_entry(records, sample_id)
        predictions[sample_id] = align_boxes(raw_boxes, list(gt_boxes.keys()))
    return predictions


def evaluate(ground_truth, predictions):
    all_ious = []
    all_errors = []
    sample_metrics = {}
    for sample_id, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(sample_id, {})
        sample_ious = []
        sample_errors = []
        for frame_stem, gt_box in gt_boxes.items():
            pred_box = pred_boxes.get(frame_stem)
            sample_ious.append(iou(gt_box, pred_box))
            sample_errors.append(normalized_error(gt_box, pred_box))
        sample_metrics[sample_id] = {
            "frames": len(sample_ious),
            "mIoU": mean(sample_ious),
            "Acc@0.5": sum(value >= 0.5 for value in sample_ious) / len(sample_ious) if sample_ious else 0.0,
        }
        all_ious.extend(sample_ious)
        all_errors.extend(sample_errors)
    summary = {
        "samples": len(ground_truth),
        "frames": len(all_ious),
        "mIoU": mean(all_ious),
        "Acc@0.5": sum(value >= 0.5 for value in all_ious) / len(all_ious) if all_ious else 0.0,
        "Norm Precision": threshold_auc(all_errors, [step / 100 for step in range(51)], True),
        "IoU AUC": threshold_auc(all_ious, [step / 100 for step in range(101)], False),
    }
    return {"summary": summary, "samples": sample_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--save", default="")
    args = parser.parse_args()

    ground_truth = load_ground_truth(Path(args.gt_root), args.split)
    predictions = load_predictions(Path(args.pred_file), ground_truth)
    report = evaluate(ground_truth, predictions)

    output = json.dumps(report, indent=2)
    print(output)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
