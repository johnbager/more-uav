from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from motion import build_motion_mask


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom


def xyxy_to_cxcywh(boxes: torch.Tensor):
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor):
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = (cx - w / 2.0).clamp(0.0, 1.0)
    y1 = (cy - h / 2.0).clamp(0.0, 1.0)
    x2 = (cx + w / 2.0).clamp(0.0, 1.0)
    y2 = (cy + h / 2.0).clamp(0.0, 1.0)
    left = torch.minimum(x1, x2)
    right = torch.maximum(x1, x2)
    top = torch.minimum(y1, y2)
    bottom = torch.maximum(y1, y2)
    return torch.stack([left, top, right, bottom], dim=-1)


def box_area(boxes: torch.Tensor):
    return (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0) * (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)


def generalized_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
    inter_x1 = torch.maximum(pred_boxes[..., 0], target_boxes[..., 0])
    inter_y1 = torch.maximum(pred_boxes[..., 1], target_boxes[..., 1])
    inter_x2 = torch.minimum(pred_boxes[..., 2], target_boxes[..., 2])
    inter_y2 = torch.minimum(pred_boxes[..., 3], target_boxes[..., 3])
    inter_area = (inter_x2 - inter_x1).clamp_min(0.0) * (inter_y2 - inter_y1).clamp_min(0.0)
    area_pred = box_area(pred_boxes)
    area_target = box_area(target_boxes)
    union = area_pred + area_target - inter_area
    iou = inter_area / union.clamp_min(1e-6)
    enc_x1 = torch.minimum(pred_boxes[..., 0], target_boxes[..., 0])
    enc_y1 = torch.minimum(pred_boxes[..., 1], target_boxes[..., 1])
    enc_x2 = torch.maximum(pred_boxes[..., 2], target_boxes[..., 2])
    enc_y2 = torch.maximum(pred_boxes[..., 3], target_boxes[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp_min(0.0) * (enc_y2 - enc_y1).clamp_min(0.0)
    giou = iou - (enc_area - union) / enc_area.clamp_min(1e-6)
    return 1.0 - giou


def move_to_device(inputs, device: torch.device):
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class MotionAwarePrefixAdapter(nn.Module):
    def __init__(self, hidden_size: int, prefix_tokens: int, dropout: float):
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1)
        self.prefix = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * prefix_tokens),
        )
        self.prefix_tokens = prefix_tokens
        self.hidden_size = hidden_size

    def forward(self, text_embeddings: torch.Tensor, attention_mask: torch.Tensor, motion_mask: torch.Tensor):
        scores = self.scorer(text_embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bn,bnh->bh", alpha, text_embeddings)
        prefix = self.prefix(pooled).view(text_embeddings.size(0), self.prefix_tokens, self.hidden_size)
        target = motion_mask * attention_mask.float()
        positive = target.sum(dim=-1)
        target = target / positive.unsqueeze(-1).clamp_min(1.0)
        motion_loss = -(target * torch.log(alpha.clamp_min(1e-8))).sum(dim=-1)
        valid = positive > 0
        if valid.any():
            motion_loss = motion_loss[valid].mean()
        else:
            motion_loss = prefix.new_tensor(0.0)
        return prefix, motion_loss


class MultiViewAlignmentAdapter(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.align = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, frame_features: torch.Tensor):
        state = torch.zeros_like(frame_features[:, 0, :])
        outputs = []
        for index in range(frame_features.size(1)):
            current = frame_features[:, index, :]
            aligned = self.align(torch.cat([current, state], dim=-1))
            gate = self.gate(torch.cat([current, aligned], dim=-1))
            enhanced = gate * aligned + (1.0 - gate) * current
            outputs.append(enhanced)
            state = enhanced
        return torch.stack(outputs, dim=1)


class TemporalGroundingDecoder(nn.Module):
    def __init__(self, hidden_size: int, prefix_tokens: int, decoder_layers: int, decoder_heads: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=decoder_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=decoder_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 5),
        )
        self.prefix_tokens = prefix_tokens

    def forward(self, prefix_prompts: torch.Tensor, frame_features: torch.Tensor):
        tokens = torch.cat([prefix_prompts, frame_features], dim=1)
        encoded = self.encoder(tokens)
        frame_tokens = encoded[:, self.prefix_tokens :, :]
        head_output = self.head(frame_tokens)
        visibility_logits = head_output[..., 0]
        box_xyxy = cxcywh_to_xyxy(torch.sigmoid(head_output[..., 1:5]))
        return box_xyxy, visibility_logits


class MoReUAVQwen(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        processor,
        hidden_size: int,
        prefix_tokens: int,
        decoder_layers: int,
        decoder_heads: int,
        dropout: float,
        text_max_length: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.hidden_size = hidden_size
        self.text_max_length = text_max_length
        self.mpa = MotionAwarePrefixAdapter(hidden_size, prefix_tokens, dropout)
        self.mva = MultiViewAlignmentAdapter(hidden_size, dropout)
        self.decoder = TemporalGroundingDecoder(hidden_size, prefix_tokens, decoder_layers, decoder_heads, dropout)

    @property
    def device(self):
        return next(self.parameters()).device

    def frame_prompt(self, expression: str, frame_name: str, width: int, height: int):
        text = expression.strip() if expression else "the referred target"
        return (
            "Ground the referred target in this UAV frame according to the motion-centric query. "
            f"Expression: {text}. "
            f"Frame: {frame_name}. "
            f"Image size: width={width}, height={height}."
        )

    def encode_text(self, expressions: list[str]):
        tokenizer = self.processor.tokenizer
        tokenized, motion_mask = build_motion_mask(tokenizer, expressions, self.text_max_length)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        motion_mask = motion_mask.to(self.device)
        with torch.no_grad():
            text_embeddings = self.backbone.get_input_embeddings()(input_ids)
        return self.mpa(text_embeddings, attention_mask, motion_mask)

    def encode_frame_batch(self, expressions: list[str], frame_paths: list[str]):
        texts = []
        images = []
        for expression, frame_path in zip(expressions, frame_paths):
            image = Image.open(frame_path).convert("RGB")
            width, height = image.size
            prompt = self.frame_prompt(expression, Path(frame_path).name, width, height)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            images.append(image)
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = move_to_device(inputs, self.device)
        outputs = self.backbone(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states[-1]
        pooled = masked_mean(hidden_states, inputs["attention_mask"])
        for image in images:
            image.close()
        return pooled

    def forward(self, expressions: list[str], frame_paths: list[list[str]]):
        prefix_prompts, motion_loss = self.encode_text(expressions)
        frame_features = []
        sequence_length = len(frame_paths[0])
        for time_index in range(sequence_length):
            step_paths = [sample_paths[time_index] for sample_paths in frame_paths]
            frame_features.append(self.encode_frame_batch(expressions, step_paths))
        frame_features = torch.stack(frame_features, dim=1)
        aligned = self.mva(frame_features)
        pred_boxes, visibility_logits = self.decoder(prefix_prompts, aligned)
        return {
            "pred_boxes": pred_boxes,
            "visibility_logits": visibility_logits,
            "motion_loss": motion_loss,
        }

    def export_method_state(self):
        return {
            "mpa": self.mpa.state_dict(),
            "mva": self.mva.state_dict(),
            "decoder": self.decoder.state_dict(),
            "settings": {
                "hidden_size": self.hidden_size,
                "prefix_tokens": self.decoder.prefix_tokens,
                "text_max_length": self.text_max_length,
            },
        }

    def load_method_state(self, payload):
        self.mpa.load_state_dict(payload["mpa"])
        self.mva.load_state_dict(payload["mva"])
        self.decoder.load_state_dict(payload["decoder"])


def build_qwen_backbone(
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_name)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    backbone = get_peft_model(backbone, lora_config)
    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None and hasattr(backbone.config, "text_config"):
        hidden_size = backbone.config.text_config.hidden_size
    return backbone, processor, int(hidden_size)


def load_qwen_from_checkpoint(
    model_name: str,
    checkpoint_dir: str | Path,
    gradient_checkpointing: bool,
):
    checkpoint_dir = Path(checkpoint_dir)
    method_payload = torch.load(checkpoint_dir / "more_uav.pt", map_location="cpu")
    settings = method_payload["settings"]
    processor_dir = checkpoint_dir / "processor"
    processor = AutoProcessor.from_pretrained(processor_dir if processor_dir.exists() else model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    backbone = PeftModel.from_pretrained(backbone, checkpoint_dir / "adapter", is_trainable=False)
    model = MoReUAVQwen(
        backbone=backbone,
        processor=processor,
        hidden_size=settings["hidden_size"],
        prefix_tokens=settings["prefix_tokens"],
        decoder_layers=method_payload["decoder_layers"],
        decoder_heads=method_payload["decoder_heads"],
        dropout=method_payload["dropout"],
        text_max_length=settings["text_max_length"],
    )
    model.load_method_state(method_payload)
    return model
