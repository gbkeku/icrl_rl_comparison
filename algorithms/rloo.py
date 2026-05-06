# ─────────────────────────────────────────────
# RLOO — Leave-One-Out Policy Gradient
# ─────────────────────────────────────────────
import torch
from config import KL_COEFF


def compute_rloo_advantages(rewards: list) -> torch.Tensor:
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    n     = len(rewards_t)
    total = rewards_t.sum()
    loo_means = (total - rewards_t) / (n - 1)
    return rewards_t - loo_means


def rloo_loss(log_probs_new, log_probs_old, advantages,
              loss_mask, ref_log_probs, epsilon=0.2):
    ratio  = torch.exp(log_probs_new - log_probs_old)
    adv    = advantages.unsqueeze(1).expand_as(ratio)
    surr1  = ratio * adv
    surr2  = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    policy_loss = -torch.min(surr1, surr2)
    kl          = log_probs_new - ref_log_probs
    combined    = (policy_loss + KL_COEFF * kl) * loss_mask
    return combined.sum() / loss_mask.sum().clamp(min=1)
