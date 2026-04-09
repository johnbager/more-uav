from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data import GroundingTrainDataset, collate_batch, load_case_records
from model import build_qwen_backbone, generalized_iou_loss, MoReUAVQwen


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_batch(batch, device: torch.device):
    batch["boxes"] = batch["boxes"].to(device)
    batch["visible"] = batch["visible"].to(device)
    return batch


def compute_losses(outputs, boxes, visible, motion_weight: float, box_l1_weight: float, giou_weight: float, visibility_weight: float):
    visibility_logits = outputs["visibility_logits"]
    pred_boxes = outputs["pred_boxes"]
    visibility_loss = F.binary_cross_entropy_with_logits(visibility_logits, visible)
    visible_mask = visible > 0.5
    if visible_mask.any():
        l1_loss = F.l1_loss(pred_boxes[visible_mask], boxes[visible_mask])
        giou_loss = generalized_iou_loss(pred_boxes[visible_mask], boxes[visible_mask]).mean()
    else:
        l1_loss = pred_boxes.new_tensor(0.0)
        giou_loss = pred_boxes.new_tensor(0.0)
    motion_loss = outputs["motion_loss"]
    total_loss = (
        visibility_weight * visibility_loss
        + box_l1_weight * l1_loss
        + giou_weight * giou_loss
        + motion_weight * motion_loss
    )
    return {
        "loss": total_loss,
        "visibility_loss": visibility_loss.detach(),
        "l1_loss": l1_loss.detach(),
        "giou_loss": giou_loss.detach(),
        "motion_loss": motion_loss.detach(),
    }


def mean_metrics(items: list[dict[str, float]]):
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(float(item[key]) for item in items) / len(items) for key in keys}


def save_checkpoint(output_dir: Path, name: str, model: MoReUAVQwen, processor, args, epoch: int, val_loss: float):
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(checkpoint_dir / "adapter")
    processor.save_pretrained(checkpoint_dir / "processor")
    payload = model.export_method_state()
    payload["decoder_layers"] = args.decoder_layers
    payload["decoder_heads"] = args.decoder_heads
    payload["dropout"] = args.dropout
    payload["epoch"] = epoch
    payload["val_loss"] = val_loss
    torch.save(payload, checkpoint_dir / "more_uav.pt")
    with (checkpoint_dir / "train_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
        handle.write("\n")


def count_trainable_parameters(model: torch.nn.Module):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_one_epoch(model, loader, optimizer, scheduler, device, args):
    model.train()
    metrics = []
    progress = tqdm(loader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(progress, start=1):
        batch = prepare_batch(batch, device)
        outputs = model(batch["expressions"], batch["frame_paths"])
        losses = compute_losses(
            outputs,
            batch["boxes"],
            batch["visible"],
            motion_weight=args.motion_loss_weight,
            box_l1_weight=args.box_l1_weight,
            giou_weight=args.giou_weight,
            visibility_weight=args.visibility_weight,
        )
        (losses["loss"] / args.gradient_accumulation_steps).backward()
        if step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        metrics.append({key: float(value) for key, value in losses.items()})
        progress.set_postfix(loss=f"{losses['loss'].item():.4f}")
    if len(loader) % args.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return mean_metrics(metrics)


@torch.no_grad()
def evaluate_one_epoch(model, loader, device, args):
    model.eval()
    metrics = []
    progress = tqdm(loader, desc="val", leave=False)
    for batch in progress:
        batch = prepare_batch(batch, device)
        outputs = model(batch["expressions"], batch["frame_paths"])
        losses = compute_losses(
            outputs,
            batch["boxes"],
            batch["visible"],
            motion_weight=args.motion_loss_weight,
            box_l1_weight=args.box_l1_weight,
            giou_weight=args.giou_weight,
            visibility_weight=args.visibility_weight,
        )
        metrics.append({key: float(value) for key, value in losses.items()})
        progress.set_postfix(loss=f"{losses['loss'].item():.4f}")
    return mean_metrics(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_sampled_frames", type=int, default=8)
    parser.add_argument("--max_cases_train", type=int, default=0)
    parser.add_argument("--max_cases_val", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--prefix_tokens", type=int, default=8)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text_max_length", type=int, default=128)
    parser.add_argument("--motion_loss_weight", type=float, default=0.2)
    parser.add_argument("--box_l1_weight", type=float, default=5.0)
    parser.add_argument("--giou_weight", type=float, default=2.0)
    parser.add_argument("--visibility_weight", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_case_records(args.dataset_root, args.train_split, args.max_cases_train)
    val_records = load_case_records(args.dataset_root, args.val_split, args.max_cases_val)

    if not train_records:
        raise ValueError(f"No training cases were found in {Path(args.dataset_root) / args.train_split}")
    if not val_records:
        raise ValueError(f"No validation cases were found in {Path(args.dataset_root) / args.val_split}")

    train_dataset = GroundingTrainDataset(train_records, args.num_sampled_frames)
    val_dataset = GroundingTrainDataset(val_records, args.num_sampled_frames)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    backbone, processor, hidden_size = build_qwen_backbone(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    model = MoReUAVQwen(
        backbone=backbone,
        processor=processor,
        hidden_size=hidden_size,
        prefix_tokens=args.prefix_tokens,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        dropout=args.dropout,
        text_max_length=args.text_max_length,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, args.epochs * max(1, updates_per_epoch))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    summary = {
        "train_cases": len(train_records),
        "val_cases": len(val_records),
        "trainable_parameters": count_trainable_parameters(model),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, args)
        val_metrics = evaluate_one_epoch(model, val_loader, device, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
            handle.write("\n")
        save_checkpoint(output_dir, "last", model, processor, args, epoch, val_metrics["loss"])
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(output_dir, "best", model, processor, args, epoch, val_metrics["loss"])
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
