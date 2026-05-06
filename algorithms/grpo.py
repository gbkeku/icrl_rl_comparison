# ─────────────────────────────────────────────
# GRPO — Group Relative Policy Optimization
# Baseline algorithm from the ICRL paper
# ─────────────────────────────────────────────
import torch
from config import KL_COEFF


def compute_grpo_advantages(rewards: list) -> torch.Tensor:
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean = rewards_t.mean()
    std  = rewards_t.std().clamp(min=1e-8)
    return (rewards_t - mean) / std


def grpo_loss(log_probs_new, log_probs_old, advantages,
              loss_mask, ref_log_probs, epsilon=0.2):
    ratio  = torch.exp(log_probs_new - log_probs_old)
    adv    = advantages.unsqueeze(1).expand_as(ratio)
    surr1  = ratio * adv
    surr2  = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    policy_loss = -torch.min(surr1, surr2)
    kl          = log_probs_new - ref_log_probs
    combined    = (policy_loss + KL_COEFF * kl) * loss_mask
    return combined.sum() / loss_mask.sum().clamp(min=1)
