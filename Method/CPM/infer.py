from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import attach_prediction, build_prediction_store, build_prompt, fill_missing_frames, load_samples, parse_bbox_text, save_json, select_context_paths


def load_model(model_name: str, attn_implementation: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
    model = AutoModel.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def infer_one(model, tokenizer, context_paths, expression: str):
    from PIL import Image

    pil_images = [Image.open(path).convert("RGB") for path in context_paths]
    width, height = pil_images[-1].size
    prompt = build_prompt(expression, context_paths[-1].name, width, height)
    msgs = [{"role": "user", "content": pil_images + [prompt]}]
    response = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
    if not isinstance(response, str):
        response = "".join(response)
    box = parse_bbox_text(response, width, height)
    for image in pil_images:
        image.close()
    return box, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default="openbmb/MiniCPM-V-2_6")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, args.attn_implementation)
    samples = load_samples(args.dataset_root, args.split, args.max_cases)
    predictions = build_prediction_store()
    raw_outputs = {}

    for sample in samples:
        frame_boxes = {}
        sample_logs = {}
        for index in range(0, len(sample.frame_paths), max(1, args.frame_stride)):
            context_paths = select_context_paths(sample.frame_paths, index, args.context_frames, args.frame_stride)
            box, response = infer_one(model, tokenizer, context_paths, sample.expression)
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
