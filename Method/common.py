from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    sample_id: str
    expression: str
    case_dir: Path
    frame_paths: list[Path]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


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


def load_expression_text(case_dir: Path):
    expression_path = case_dir / "expression.txt"
    if not expression_path.exists():
        return ""
    return expression_path.read_text(encoding="utf-8").strip()


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


def load_samples(dataset_root: str | Path, split: str, limit: int = 0):
    split_dir = Path(dataset_root) / split
    samples = []
    expression_path = split_dir / "expression.json"
    if expression_path.exists():
        seen = set()
        raw_entries = normalize_expression_entries(load_json(expression_path))
        for raw_id, expression in raw_entries:
            case_dir = resolve_case_dir(split_dir, raw_id)
            if case_dir is None:
                continue
            sample_id = case_dir.name
            if sample_id in seen:
                continue
            frame_paths = list_frames(case_dir / "images")
            if not frame_paths:
                continue
            samples.append(Sample(sample_id=sample_id, expression=expression, case_dir=case_dir, frame_paths=frame_paths))
            seen.add(sample_id)
            if limit and len(samples) >= limit:
                return samples
        if samples:
            return samples
    for case_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        frame_paths = list_frames(case_dir / "images")
        if not frame_paths:
            continue
        expression = load_expression_text(case_dir)
        samples.append(Sample(sample_id=case_dir.name, expression=expression, case_dir=case_dir, frame_paths=frame_paths))
        if limit and len(samples) >= limit:
            break
    return samples


def select_context_paths(frame_paths: list[Path], target_index: int, context_frames: int, frame_stride: int):
    count = max(1, context_frames)
    step = max(1, frame_stride)
    start = max(0, target_index - step * (count - 1))
    indices = list(range(start, target_index + 1, step))
    if not indices or indices[-1] != target_index:
        indices.append(target_index)
    indices = indices[-count:]
    return [frame_paths[index] for index in indices]


def build_prompt(expression: str, target_frame_name: str, width: int, height: int):
    expression_text = expression.strip() if expression else "the referred target"
    return (
        "You are given ordered UAV video frames. "
        "The last image is the target frame. "
        "Localize the object referred to by the expression in the last image only. "
        f"Referring expression: {expression_text}. "
        f"Target frame: {target_frame_name}. "
        f"Image size: width={width}, height={height}. "
        'Return only one JSON object in one line with the format {"bbox":[x1,y1,x2,y2]}. '
        'Use integer pixel coordinates. If the target is absent or cannot be localized, return {"bbox": null}.'
    )


def parse_box(value, width: int, height: int):
    if value is None:
        return None
    if isinstance(value, dict):
        if "bbox" in value:
            value = value["bbox"]
        elif "box" in value:
            value = value["box"]
        elif all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
        else:
            return None
    if not isinstance(value, list) or len(value) != 4:
        return None
    numbers = [float(item) for item in value]
    if all(0.0 <= item <= 1.5 for item in numbers):
        numbers = [numbers[0] * width, numbers[1] * height, numbers[2] * width, numbers[3] * height]
    x1, y1, x2, y2 = numbers
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = max(0.0, min(left, width - 1))
    right = max(0.0, min(right, width - 1))
    top = max(0.0, min(top, height - 1))
    bottom = max(0.0, min(bottom, height - 1))
    if right <= left or bottom <= top:
        return None
    return [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))]


def parse_bbox_text(text: str, width: int, height: int):
    stripped = text.strip()
    if not stripped:
        return None
    match = re.search(r"\{.*\}", stripped, re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            box = parse_box(payload, width, height)
            if box is not None:
                return box
        except json.JSONDecodeError:
            pass
    matches = re.findall(r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]", stripped)
    for candidate in matches:
        try:
            box = parse_box(json.loads(candidate), width, height)
            if box is not None:
                return box
        except json.JSONDecodeError:
            continue
    return None


def build_prediction_store():
    return {}


def attach_prediction(store, sample_id: str, frame_stem: str, box):
    sample_store = store.setdefault(sample_id, {})
    sample_store[frame_stem] = box


def fill_missing_frames(frame_paths: list[Path], frame_boxes: dict[str, list[int] | None]):
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
