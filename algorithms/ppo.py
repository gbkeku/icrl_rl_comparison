# ─────────────────────────────────────────────
# PPO — Proximal Policy Optimization
# Via TRL's PPOTrainer with ICRL loss masking
# ─────────────────────────────────────────────
from config import LEARNING_RATE, KL_COEFF, BATCH_SIZE


def build_ppo_config():
    from trl import PPOConfig
    return PPOConfig(
        learning_rate   = LEARNING_RATE,
        batch_size      = BATCH_SIZE,
        kl_penalty      = "kl",
        init_kl_coef    = KL_COEFF,
        adap_kl_ctrl    = False,
        cliprange       = 0.2,
        cliprange_value = 0.2,
        vf_coef         = 0.1,
        log_with        = None,
    )


def build_ppo_trainer(model, ref_model, tokenizer, dataset):
    from trl import PPOTrainer
    return PPOTrainer(
        config    = build_ppo_config(),
        model     = model,
        ref_model = ref_model,
        tokenizer = tokenizer,
        dataset   = dataset,
    )
