# ─────────────────────────────────────────────
# REINFORCE with moving average baseline
# ─────────────────────────────────────────────
import torch


class ReinforceWithBaseline:
    def __init__(self, alpha: float = 0.99):
        self.baseline = 0.0
        self.alpha    = alpha

    def update_baseline(self, reward: float):
        self.baseline = self.alpha * self.baseline + (1 - self.alpha) * reward

    def compute_advantage(self, reward: float) -> float:
        return reward - self.baseline

    def loss(self, log_probs, rewards, loss_mask, ref_log_probs):
        from config import KL_COEFF
        advantages = torch.tensor(
            [self.compute_advantage(r) for r in rewards],
            dtype=torch.float32
        )
        self.update_baseline(sum(rewards) / len(rewards))
        adv      = advantages.unsqueeze(1).expand_as(log_probs)
        policy   = -log_probs * adv
        kl       = log_probs - ref_log_probs
        combined = (policy + KL_COEFF * kl) * loss_mask
        return combined.sum() / loss_mask.sum().clamp(min=1)
