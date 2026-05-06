# ─────────────────────────────────────────────
# Reward computation: accuracy + format
# ─────────────────────────────────────────────
import re
from config import ALPHA


def compute_accuracy_reward(prediction: str, ground_truth: str) -> float:
    pred = extract_answer(prediction).strip().lower()
    gt   = ground_truth.strip().lower()
    return 1.0 if pred == gt else 0.0


def compute_format_reward(response: str) -> float:
    penalties = 0.0
    if "<answer>" not in response:
        penalties += 0.5
    if "<think>" not in response:
        penalties += 0.15
    for tag in ["answer", "think", "search"]:
        opens  = response.count(f"<{tag}>")
        closes = response.count(f"</{tag}>")
        if opens != closes:
            penalties += 0.2
    if "<search>" not in response:
        penalties += 0.1
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match and not answer_match.group(1).strip():
        penalties += 0.2
    return max(0.0, 1.0 - penalties)


def compute_reward(prediction: str, ground_truth: str) -> float:
    acc = compute_accuracy_reward(prediction, ground_truth)
    fmt = compute_format_reward(prediction)
    return ALPHA * acc + (1 - ALPHA) * fmt


def extract_answer(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def mask_tool_tokens(input_ids, response_ids, retrieval_spans):
    import torch
    mask = torch.ones(len(response_ids), dtype=torch.bool)
    for start, end in retrieval_spans:
        mask[start:end] = False
    return mask
