from __future__ import annotations

import re

import torch


DIRECTIONAL_PATTERNS = [
    "turn right",
    "turn left",
    "go straight",
    "move straight",
    "move behind",
    "move ahead",
    "behind",
    "ahead",
    "left",
    "right",
    "straight",
    "toward",
    "towards",
    "away",
    "around",
    "across",
    "along",
    "through",
]


def directional_spans(text: str):
    lowered = text.lower()
    spans = []
    for pattern in DIRECTIONAL_PATTERNS:
        for match in re.finditer(rf"\b{re.escape(pattern)}\b", lowered):
            spans.append((match.start(), match.end()))
    spans.sort()
    return spans


def overlaps(span_a, span_b):
    return span_a[0] < span_b[1] and span_b[0] < span_a[1]


def nltk_verb_spans(text: str):
    try:
        import nltk
        from nltk.tokenize import TreebankWordTokenizer
    except Exception:
        return []
    tokenizer = TreebankWordTokenizer()
    spans = list(tokenizer.span_tokenize(text))
    tokens = [text[start:end] for start, end in spans]
    if not tokens:
        return []
    try:
        tagged = nltk.pos_tag(tokens)
    except LookupError:
        try:
            tagged = nltk.pos_tag(tokens, lang="eng")
        except Exception:
            return []
    except Exception:
        return []
    verb_spans = []
    for (token, tag), span in zip(tagged, spans):
        if isinstance(tag, str) and tag.startswith("VB"):
            verb_spans.append(span)
    return verb_spans


def motion_spans(text: str):
    spans = nltk_verb_spans(text)
    spans.extend(directional_spans(text))
    spans.sort()
    deduped = []
    for span in spans:
        if span not in deduped:
            deduped.append(span)
    return deduped


def build_motion_mask(tokenizer, expressions: list[str], max_length: int):
    try:
        tokenized = tokenizer(
            expressions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = tokenized.pop("offset_mapping")
    except Exception:
        tokenized = tokenizer(
            expressions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = None
    attention_mask = tokenized["attention_mask"]
    motion_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    for batch_index, expression in enumerate(expressions):
        spans = motion_spans(expression)
        if not spans:
            continue
        if offsets is not None:
            for token_index, span in enumerate(offsets[batch_index].tolist()):
                if attention_mask[batch_index, token_index].item() == 0:
                    continue
                token_span = (int(span[0]), int(span[1]))
                if token_span[0] == token_span[1]:
                    continue
                if any(overlaps(token_span, motion_span) for motion_span in spans):
                    motion_mask[batch_index, token_index] = 1.0
        else:
            input_ids = tokenized["input_ids"][batch_index]
            for token_index, token_id in enumerate(input_ids.tolist()):
                if attention_mask[batch_index, token_index].item() == 0:
                    continue
                token_text = tokenizer.decode([token_id]).lower()
                if any(pattern in token_text for pattern in DIRECTIONAL_PATTERNS):
                    motion_mask[batch_index, token_index] = 1.0
    return tokenized, motion_mask
