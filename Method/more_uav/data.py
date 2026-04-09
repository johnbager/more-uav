from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class CaseRecord:
    sample_id: str
    expression: str
    case_dir: Path
    frame_paths: list[Path]
    aligned_boxes: dict[str, list[float] | None]
    width: int
    height: int


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_frames(images_dir: Path):
    if not images_dir.exists():
        return []
    return sorted(
        [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda item: item.name,
    )


def candidate_case_names(raw_id: str):
    value = str(raw_id).strip()
    names = [value]
    if value.startswith("case_"):
        suffix = value[5:]
        if suffix.isdigit():
            names.append(suffix)
            names.append(str(int(suffix)))
    elif value.isdigit():
        names.append(f"case_{int(value):08d}")
        names.append(f"case_{value.zfill(8)}")
    return list(dict.fromkeys(names))


def resolve_case_dir(split_dir: Path, raw_id: str):
    for candidate in candidate_case_names(raw_id):
        case_dir = split_dir / candidate
        if case_dir.exists() and case_dir.is_dir():
            return case_dir
    return None


def normalize_expression_entries(raw):
    entries = []
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            return normalize_expression_entries(raw["data"])
        for key, value in raw.items():
            if isinstance(value, str):
                entries.append((str(key), value))
            elif isinstance(value, dict):
                case_id = value.get("id", key)
                expression = value.get("expression") or value.get("text") or value.get("caption") or ""
                entries.append((str(case_id), str(expression)))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            case_id = item.get("id") or item.get("sample_id") or item.get("case_id")
            expression = item.get("expression") or item.get("text") or item.get("caption") or ""
            if case_id is not None:
                entries.append((str(case_id), str(expression)))
    return entries


def load_expression_map(split_dir: Path):
    expression_path = split_dir / "expression.json"
    mapping = {}
    if expression_path.exists():
        for raw_id, expression in normalize_expression_entries(load_json(expression_path)):
            case_dir = resolve_case_dir(split_dir, raw_id)
            if case_dir is not None:
                mapping[case_dir.name] = expression
    if mapping:
        return mapping
    for case_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        expression_path = case_dir / "expression.txt"
        if expression_path.exists():
            mapping[case_dir.name] = expression_path.read_text(encoding="utf-8").strip()
    return mapping


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


def load_case_records(dataset_root: str | Path, split: str, limit: int = 0):
    split_dir = Path(dataset_root) / split
    expression_map = load_expression_map(split_dir)
    records = []
    for case_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        frame_paths = list_frames(case_dir / "images")
        if not frame_paths:
            continue
        bboxes_path = case_dir / "bboxes.json"
        if not bboxes_path.exists():
            continue
        raw_boxes = load_json(bboxes_path)
        aligned_boxes = align_boxes(raw_boxes, [path.stem for path in frame_paths])
        with Image.open(frame_paths[0]) as image:
            width, height = image.size
        records.append(
            CaseRecord(
                sample_id=case_dir.name,
                expression=expression_map.get(case_dir.name, ""),
                case_dir=case_dir,
                frame_paths=frame_paths,
                aligned_boxes=aligned_boxes,
                width=width,
                height=height,
            )
        )
        if limit and len(records) >= limit:
            break
    return records


def normalize_box(box, width: int, height: int):
    if box is None:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    x1, y1, x2, y2 = box
    return [x1 / width, y1 / height, x2 / width, y2 / height], 1.0


def uniform_sample_indices(length: int, num_samples: int):
    if length <= 0:
        return []
    if num_samples <= 1 or length == 1:
        return [length - 1]
    if length <= num_samples:
        indices = list(range(length))
        while len(indices) < num_samples:
            indices.append(indices[-1])
        return indices
    return [round(step * (length - 1) / (num_samples - 1)) for step in range(num_samples)]


def context_window_indices(length: int, target_index: int, window_size: int):
    indices = list(range(max(0, target_index - window_size + 1), target_index + 1))
    while len(indices) < window_size:
        indices.insert(0, indices[0] if indices else target_index)
    return indices[-window_size:]


def fill_missing_boxes(frame_paths: list[Path], frame_boxes: dict[str, list[int] | None]):
    ordered_names = [path.stem for path in frame_paths]
    last_box = None
    for frame_name in ordered_names:
        if frame_name not in frame_boxes:
            frame_boxes[frame_name] = last_box
        elif frame_boxes[frame_name] is not None:
            last_box = frame_boxes[frame_name]
    next_box = None
    for frame_name in reversed(ordered_names):
        if frame_boxes.get(frame_name) is None and next_box is not None:
            frame_boxes[frame_name] = next_box
        elif frame_boxes.get(frame_name) is not None:
            next_box = frame_boxes[frame_name]
    return frame_boxes


class GroundingTrainDataset(Dataset):
    def __init__(self, records: list[CaseRecord], num_sampled_frames: int):
        self.records = records
        self.num_sampled_frames = num_sampled_frames

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        indices = uniform_sample_indices(len(record.frame_paths), self.num_sampled_frames)
        frame_paths = [record.frame_paths[item] for item in indices]
        frame_names = [path.stem for path in frame_paths]
        boxes = []
        visible = []
        for frame_name in frame_names:
            box, is_visible = normalize_box(record.aligned_boxes.get(frame_name), record.width, record.height)
            boxes.append(box)
            visible.append(is_visible)
        return {
            "sample_id": record.sample_id,
            "expression": record.expression,
            "frame_paths": [str(path) for path in frame_paths],
            "frame_names": frame_names,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "visible": torch.tensor(visible, dtype=torch.float32),
            "width": record.width,
            "height": record.height,
        }


def collate_batch(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "expressions": [item["expression"] for item in batch],
        "frame_paths": [item["frame_paths"] for item in batch],
        "frame_names": [item["frame_names"] for item in batch],
        "boxes": torch.stack([item["boxes"] for item in batch], dim=0),
        "visible": torch.stack([item["visible"] for item in batch], dim=0),
        "widths": [item["width"] for item in batch],
        "heights": [item["height"] for item in batch],
    }
