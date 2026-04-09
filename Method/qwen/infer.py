from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import attach_prediction, build_prediction_store, build_prompt, fill_missing_frames, load_samples, parse_bbox_text, save_json, select_context_paths


def load_model(model_name: str, device_map: str, attn_implementation: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "torch_dtype": dtype,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if attn_implementation and torch.cuda.is_available():
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    if not device_map:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, processor


def move_inputs(inputs, model):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return inputs
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def infer_one(model, processor, context_paths, expression: str, max_new_tokens: int):
    from PIL import Image

    pil_images = [Image.open(path).convert("RGB") for path in context_paths]
    width, height = pil_images[-1].size
    prompt = build_prompt(expression, context_paths[-1].name, width, height)
    content = []
    for index in range(len(pil_images)):
        content.append({"type": "text", "text": f"Frame {index + 1}"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_images, padding=True, return_tensors="pt")
    inputs = move_inputs(inputs, model)
    output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    box = parse_bbox_text(response, width, height)
    for image in pil_images:
        image.close()
    return box, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    model, processor = load_model(args.model_name, args.device_map, args.attn_implementation)
    samples = load_samples(args.dataset_root, args.split, args.max_cases)
    predictions = build_prediction_store()
    raw_outputs = {}

    for sample in samples:
        frame_boxes = {}
        sample_logs = {}
        for index in range(0, len(sample.frame_paths), max(1, args.frame_stride)):
            context_paths = select_context_paths(sample.frame_paths, index, args.context_frames, args.frame_stride)
            box, response = infer_one(model, processor, context_paths, sample.expression, args.max_new_tokens)
            frame_stem = sample.frame_paths[index].stem
            frame_boxes[frame_stem] = box
            sample_logs[frame_stem] = response
        frame_boxes = fill_missing_frames(sample.frame_paths, frame_boxes)
        for frame_name, box in frame_boxes.items():
            attach_prediction(predictions, sample.sample_id, frame_name, box)
        raw_outputs[sample.sample_id] = sample_logs

    payload = {
        "model": args.model_name,
        "split": args.split,
        "predictions": predictions,
        "raw_outputs": raw_outputs,
    }
    save_json(Path(args.output), payload)


if __name__ == "__main__":
    main()
