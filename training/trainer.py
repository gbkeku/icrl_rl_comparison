# ─────────────────────────────────────────────
# Unified ICRL Trainer
# ─────────────────────────────────────────────
import re
import gc
import json
import random
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    MODEL_NAME, DTYPE, DEVICE, CURRICULUM,
    STEPS_PER_STAGE, NUM_ROLLOUTS,
    MAX_PROMPT_LENGTH, TEMPERATURE, KL_COEFF
)
from utils.rewards import compute_reward, extract_answer
from algorithms.grpo import compute_grpo_advantages
from algorithms.rloo import compute_rloo_advantages
from algorithms.reinforce import ReinforceWithBaseline

MAX_LEN  = 1024
MAX_NEW  = 150
CLIP_EPS = 0.2


# ── Model loading ─────────────────────────────
def load_policy_model(model_name=MODEL_NAME):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_ref_model(model_name=MODEL_NAME):
    ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"),
        device_map=DEVICE, low_cpu_mem_usage=True)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def clear():
    """Aggressive VRAM cleanup — call between stages and steps."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def vram():
    u = torch.cuda.memory_allocated() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{u:.2f}/{t:.2f} GB"


# ── Prompt builder ────────────────────────────
def build_rollout_prompt(question, demos, n_shot):
    system = (
        "Solve the following problem step by step. "
        "Reason inside <think>...</think>. "
        "Search using <search>query</search>. "
        "Results appear in <information>...</information>. "
        "Give your final answer in <answer>...</answer>."
    )
    prompt = system + "\n\n"
    if n_shot > 0:
        prompt += "Here are some examples:\n\n"
        for demo in demos[:n_shot]:
            prompt += f"Example Problem: {demo['question']}\n"
            prompt += f"Example Solution: {demo['solution']}\n\n"
    prompt += f"Now solve the following problem:\n{question}"
    return prompt


def build_token_mask_exact(full_text, retrieval_spans,
                            tokenizer, prompt_len_tokens,
                            total_tokens):
    """
    Build per-token binary mask using tokenizer offset mapping.

    Uses return_offsets_mapping=True for exact
       char→token alignment instead of approximate scaling.

    Mask rules:
      - Prompt tokens          → 0 (never train on prompt)
      - Model response tokens  → 1 (train on these)
      - Retrieved info tokens  → 0 (excluded: not model-generated)

    Args:
        full_text        : prompt + response string
        retrieval_spans  : list of (char_start, char_end) for
                           retrieved <information> content
        tokenizer        : HuggingFace tokenizer
        prompt_len_tokens: number of tokens in prompt
        total_tokens     : total tokens in full sequence

    Returns:
        FloatTensor [total_tokens - 1] (shifted for next-token pred)
    """
    # Start with all zeros (mask everything)
    mask = torch.zeros(total_tokens - 1, dtype=torch.float32,
                       device=DEVICE)

    # Enable all response tokens (after prompt)
    resp_start = max(0, prompt_len_tokens - 1)  # -1 for shift
    mask[resp_start:] = 1.0

    # Use offset mapping for exact char→token alignment
    if retrieval_spans:
        try:
            enc_with_offsets = tokenizer(
                full_text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            offsets = enc_with_offsets["offset_mapping"]
            # offsets[i] = (char_start, char_end) for token i

            for char_start, char_end in retrieval_spans:
                for tok_idx, (tok_char_start, tok_char_end) in enumerate(offsets):
                    # Token overlaps with retrieved span
                    if (tok_char_start < char_end and
                            tok_char_end > char_start):
                        # Shift by 1 for next-token prediction alignment
                        shifted = tok_idx - 1
                        if 0 <= shifted < total_tokens - 1:
                            mask[shifted] = 0.0  # ✅ exact masking
        except Exception:
            # Fallback to approximate if offset mapping fails
            response_chars = max(len(full_text) - len(
                full_text.split("Now solve")[0]), 1)
            response_len   = total_tokens - prompt_len_tokens
            for char_start, char_end in retrieval_spans:
                tok_s = resp_start + int(
                    (char_start / response_chars) * response_len)
                tok_e = resp_start + int(
                    (char_end   / response_chars) * response_len)
                tok_s = max(resp_start, min(tok_s, total_tokens - 2))
                tok_e = max(tok_s,      min(tok_e, total_tokens - 1))
                mask[tok_s:tok_e] = 0.0

    return mask


# ── ICRLTrainer ───────────────────────────────
class ICRLTrainer:
    """
    Unified ICRL trainer
    run_rollout()  : multi-turn tool loop, exact retrieval_spans
    compute_loss() : exact masked gradient, corrected KL
    train_step()   : EM + tool_success logged per step
    train_stage()  : stage-level summary
    train()        : full 3->2->0 curriculum
    save()         : checkpoint + logs

    Usage:
        trainer = ICRLTrainer("grpo", retriever, train_data, demos)
        trainer.train()
        trainer.save()
    """

    def __init__(self, algorithm, retriever, train_data, demos):
        assert algorithm in ("grpo", "ppo", "reinforce", "rloo"), \
            f"Unknown algorithm: {algorithm}"

        self.algorithm  = algorithm
        self.retriever  = retriever
        self.train_data = train_data
        self.demos      = demos

        print(f"\nLoading policy model ({algorithm.upper()})...")
        self.model, self.tokenizer = load_policy_model()
        print(f"Policy model ready  | VRAM: {vram()}")

        print("Loading reference model (4-bit)...")
        self.ref_model = load_ref_model()
        print(f"Ref model ready     | VRAM: {vram()}")

        # ── FIX 1: PPO value head for proper advantage ──
        if algorithm == "ppo":
            self.value_head = torch.nn.Linear(
                self.model.config.hidden_size, 1, bias=False
            ).to(DEVICE).to(torch.bfloat16)
            self.optimizer = torch.optim.AdamW(
                list(self.model.parameters()) +
                list(self.value_head.parameters()),
                lr=1e-5, eps=1e-5)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-5, eps=1e-5)

        if algorithm == "reinforce":
            self.rf_agent = ReinforceWithBaseline(alpha=0.99)

        # ── FIX 6 & 7: Extended logs ─────────────────
        self.logs = {
            "reward":          [],   # avg reward per step
            "response_length": [],   # FIX 4: in tokens
            "valid_searches":  [],   # search calls per step
            "tool_success":    [],   # FIX 3: search helped?
            "em":              [],   # FIX 6: exact match rate
            "stage":           [],   # curriculum stage
        }

    # ── Rollout ──────────────────────────────
    def run_rollout(self, question, n_shot):
        """
        Multi-turn tool-augmented rollout.
        Tracks retrieval_spans for exact loss masking.
        """
        prompt          = build_rollout_prompt(question, self.demos, n_shot)
        response_text   = ""
        retrieval_spans = []

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_LEN).to(DEVICE)

        for _ in range(6):
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, max_new_tokens=MAX_NEW,
                    do_sample=True, temperature=TEMPERATURE,
                    pad_token_id=self.tokenizer.eos_token_id)

            # Decode before deleting tensors
            prompt_len = inputs["input_ids"].shape[1]
            new_text   = self.tokenizer.decode(
                output[0][prompt_len:],
                skip_special_tokens=True)

            del output
            clear()

            response_text += new_text

            # ── Tool execution ────────────────
            match = re.search(r"<search>(.*?)</search>",
                               new_text, re.DOTALL)
            if match:
                query     = match.group(1).strip()
                docs      = self.retriever.retrieve(query)
                info_text = f"\n<information>{docs}</information>\n"

                # Record exact char span of retrieved content
                span_start = len(response_text)
                response_text += info_text
                retrieval_spans.append((span_start, len(response_text)))

                # Rebuild inputs with retrieved context appended
                new_ctx = prompt + response_text
                del inputs   # free old inputs before building new
                inputs  = self.tokenizer(
                    new_ctx, return_tensors="pt",
                    truncation=True, max_length=MAX_LEN).to(DEVICE)

            else:
                # No search — continue from current context
                new_ctx = prompt + response_text
                del inputs
                inputs  = self.tokenizer(
                    new_ctx, return_tensors="pt",
                    truncation=True, max_length=MAX_LEN).to(DEVICE)

            if "<answer>" in response_text:
                del inputs
                break

        return {
            "response":        response_text,
            "retrieval_spans": retrieval_spans,
            "prompt":          prompt,
        }

    # ── PPO advantage from value head ──
    def _compute_ppo_advantages(self, rollouts, rewards):
        """
        Compute PPO advantages using value head.

        advantage_i = reward_i - V(s_i)

        Without this fix, PPO == REINFORCE (no value baseline).
        With this fix, value head learns to predict expected reward,
        and the advantage measures how much better each rollout was.
        """
        advantages = []
        for rollout, reward in zip(rollouts, rewards):
            prompt    = rollout["prompt"]
            response  = rollout["response"]
            full_text = prompt + response

            enc = self.tokenizer(
                full_text, return_tensors="pt",
                truncation=True, max_length=MAX_LEN).to(DEVICE)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out        = self.model(**enc, output_hidden_states=True)
                    last_hidden = out.hidden_states[-1][:, -1, :]
                    value       = self.value_head(last_hidden).squeeze(-1)
                    value_scalar = value.float().item()

            # advantage = actual reward - predicted value
            adv = reward - value_scalar
            advantages.append(adv)

            del enc, out, last_hidden, value
            clear()

        return advantages

    # ── Loss computation ─────────────────────
    def compute_loss(self, rollout, advantage,
                     gt_answer=None):
        """
        Masked policy gradient loss for one rollout.

        Exact token masking via offset mapping
        Corrected KL direction: KL(π_new || π_ref)
                  = E[log π_new - log π_ref]
                  Penalizes new policy deviating from ref.
        """
        prompt          = rollout["prompt"]
        response        = rollout["response"]
        retrieval_spans = rollout["retrieval_spans"]
        full_text       = prompt + response

        enc = self.tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=MAX_LEN).to(DEVICE)

        prompt_len = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_LEN
        )["input_ids"].shape[1]
        total_len = enc["input_ids"].shape[1]

        # ── Policy model forward ──────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if self.algorithm == "ppo":
                out = self.model(**enc, output_hidden_states=True)
                last_h = out.hidden_states[-1][:, -1, :]
                value  = self.value_head(last_h).squeeze(-1)
            else:
                out = self.model(**enc)
            logits = out.logits

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        ids       = enc["input_ids"][:, 1:]
        token_lp  = log_probs.gather(
            2, ids.unsqueeze(-1)).squeeze(-1)[0]

        # ── Reference model forward ───────────
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ref_out    = self.ref_model(**enc)
                ref_logits = ref_out.logits
            ref_lp = F.log_softmax(
                ref_logits[:, :-1, :], dim=-1
            ).gather(2, ids.unsqueeze(-1)).squeeze(-1)[0]

        # ── FIX 2: Exact token mask ───────────
        mask  = build_token_mask_exact(
            full_text         = full_text,
            retrieval_spans   = retrieval_spans,
            tokenizer         = self.tokenizer,
            prompt_len_tokens = prompt_len,
            total_tokens      = total_len,
        )
        n_tok = mask.sum().clamp(min=1)

        adv_t = torch.tensor(float(advantage), device=DEVICE,
                              dtype=torch.bfloat16)

        # ── Algorithm-specific policy loss ────
        if self.algorithm in ("grpo", "rloo"):
            ratio  = torch.exp(token_lp - ref_lp)
            surr1  = ratio * adv_t
            surr2  = torch.clamp(
                ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            policy_loss = -(torch.min(surr1, surr2) * mask).sum() / n_tok

        elif self.algorithm == "reinforce":
            policy_loss = -(token_lp * adv_t * mask).sum() / n_tok

        elif self.algorithm == "ppo":
            # ✅ FIX 1: Use value-based advantage (not raw reward)
            adv_clip = (adv_t - value.detach()).clamp(-5, 5)
            ratio    = torch.exp(token_lp - ref_lp)
            surr1    = ratio * adv_clip
            surr2    = torch.clamp(
                ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_clip
            policy_loss = (
                -(torch.min(surr1, surr2) * mask).sum() / n_tok +
                0.1 * F.mse_loss(
                    value,
                    torch.tensor(float(advantage), device=DEVICE,
                                 dtype=torch.bfloat16).unsqueeze(0))
            )

        # KL(π_new || π_ref) = E[log π_new - log π_ref]
        # This penalizes the NEW policy deviating from reference.
        # Previous version had (token_lp - ref_lp) which is correct
        # BUT the sign convention matters:
        # We want to MINIMIZE KL, so we ADD it as a positive penalty.
        kl_loss = KL_COEFF * (
            (token_lp - ref_lp) * mask
        ).sum() / n_tok   # ✅ positive penalty on deviation

        loss = (policy_loss + kl_loss) / NUM_ROLLOUTS

        del enc, out, logits, log_probs, token_lp
        del ref_out, ref_logits, ref_lp, mask
        clear()
        return loss

    # ── Training step ────────────────────────
    def train_step(self, question, gt_answer, n_shot):
        """
        One full RL update step with all metrics logged.
        tool_success — did search improve answer?
        response_length in tokens
         EM logged per step
        """
        # Generate rollouts
        rollouts = []
        rewards  = []
        for _ in range(NUM_ROLLOUTS):
            r = self.run_rollout(question, n_shot)
            rollouts.append(r)
            rewards.append(compute_reward(r["response"], gt_answer))

        # Compute advantages per algorithm
        if self.algorithm == "grpo":
            advantages = compute_grpo_advantages(rewards).tolist()
        elif self.algorithm == "rloo":
            advantages = compute_rloo_advantages(rewards).tolist()
        elif self.algorithm == "reinforce":
            self.rf_agent.update_baseline(sum(rewards) / len(rewards))
            advantages = [self.rf_agent.compute_advantage(r)
                          for r in rewards]
        elif self.algorithm == "ppo":
            # value-based advantages, not raw rewards
            advantages = self._compute_ppo_advantages(rollouts, rewards)

        # Backward pass
        self.optimizer.zero_grad()
        for rollout, adv in zip(rollouts, advantages):
            self.compute_loss(rollout, adv, gt_answer).backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── Free rollout memory before metrics ──
        # Keep responses for metric extraction, then free
        responses_text = [r["response"] for r in rollouts]
        del rollouts
        clear()

        # ── Metrics ───────────────────────────
        preds = [extract_answer(r) for r in responses_text]

        # Response length in tokens
        avg_len_tokens = sum(
            len(self.tokenizer(r)["input_ids"])
            for r in responses_text
        ) / NUM_ROLLOUTS

        #  Tool success — search used AND answer correct
        tool_success = sum(
            1 for r, p in zip(responses_text, preds)
            if "<search>" in r
            and p.strip().lower() == gt_answer.strip().lower()
        ) / NUM_ROLLOUTS

        # Exact match rate this step
        em = sum(
            p.strip().lower() == gt_answer.strip().lower()
            for p in preds
        ) / NUM_ROLLOUTS

        avg_reward   = sum(rewards) / len(rewards)
        avg_searches = sum(
            r.count("<search>") for r in responses_text
        ) / NUM_ROLLOUTS

        return {
            "avg_reward":    avg_reward,
            "avg_len":       avg_len_tokens,
            "avg_searches":  avg_searches,
            "tool_success":  tool_success,
            "em":            em,
            "preds":         preds,
            "rewards":       rewards,
        }

    # ── Curriculum stage ─────────────────────
    def train_stage(self, n_shot):
        """
        Run STEPS_PER_STAGE update steps.
        Stage-level summary printed + logged.
        """
        name = {3: "3-shot", 2: "2-shot", 0: "0-shot"}[n_shot]

        # ── Aggressive VRAM cleanup before each stage ──
        # Frees lingering tensors from previous stage
        # Prevents VRAM spike at stage transitions
        clear()
        if hasattr(self, "_last_rollouts"):
            del self._last_rollouts
        torch.cuda.reset_peak_memory_stats()

        print(f"\n{'─'*60}")
        print(f"  Stage: {name}  |  Algorithm: {self.algorithm.upper()}")
        print(f"  VRAM after cleanup: {vram()}")
        print(f"{'─'*60}")

        stage_rewards  = []
        stage_em       = []
        stage_searches = []
        stage_tool_suc = []

        for step in range(STEPS_PER_STAGE):
            s       = random.choice(self.train_data)
            metrics = self.train_step(s["question"], s["answer"], n_shot)

            # Log per-step metrics
            self.logs["reward"].append(metrics["avg_reward"])
            self.logs["response_length"].append(metrics["avg_len"])
            self.logs["valid_searches"].append(metrics["avg_searches"])
            self.logs["tool_success"].append(metrics["tool_success"])
            self.logs["em"].append(metrics["em"])
            self.logs["stage"].append(n_shot)

            # Accumulate for stage summary
            stage_rewards.append(metrics["avg_reward"])
            stage_em.append(metrics["em"])
            stage_searches.append(metrics["avg_searches"])
            stage_tool_suc.append(metrics["tool_success"])

            if step % 10 == 0 or step == STEPS_PER_STAGE - 1:
                print(
                    f"  Step {step:3d}/{STEPS_PER_STAGE} | "
                    f"Reward: {metrics['avg_reward']:.3f} | "
                    f"EM: {metrics['em']:.2f} | "
                    f"Searches: {metrics['avg_searches']:.1f} | "
                    f"ToolOK: {metrics['tool_success']:.2f} | "
                    f"Preds: {metrics['preds']}"
                )

        # ── Final cleanup at stage end ───────────────
        clear()
        n = len(stage_rewards)
        print(f"\n  ┌─ Stage {name} Summary ({'─'*30})")
        print(f"  │  Avg Reward   : {sum(stage_rewards)/n:.4f}")
        print(f"  │  Avg EM       : {sum(stage_em)/n:.4f}  "
              f"({sum(stage_em)/n*100:.1f}%)")
        print(f"  │  Avg Searches : {sum(stage_searches)/n:.2f}")
        print(f"  │  Tool Success : {sum(stage_tool_suc)/n:.4f}")
        print(f"  │  VRAM (end)   : {vram()}")
        print(f"  └{'─'*45}")

        # Save stage summary to logs
        if "stage_summary" not in self.logs:
            self.logs["stage_summary"] = {}
        self.logs["stage_summary"][name] = {
            "avg_reward":    round(sum(stage_rewards) / n, 4),
            "avg_em":        round(sum(stage_em) / n, 4),
            "avg_searches":  round(sum(stage_searches) / n, 4),
            "tool_success":  round(sum(stage_tool_suc) / n, 4),
        }

    # ── Full curriculum ───────────────────────
    def train(self):
        """Full 3→2→0 curriculum training."""
        print(f"\n{'='*60}")
        print(f"  ICRL Training — {self.algorithm.upper()} ")
        print(f"  Curriculum: {' -> '.join(str(s)+'-shot' for s in CURRICULUM)}")
        print(f"  Steps/stage: {STEPS_PER_STAGE} | Rollouts: {NUM_ROLLOUTS}")
        print(f"{'='*60}")

        for n_shot in CURRICULUM:
            self.train_stage(n_shot)

        total_steps = len(self.logs["reward"])
        final_em    = self.logs["em"][-1] if self.logs["em"] else 0

        print(f"\n{'='*60}")
        print(f"  Training complete — {self.algorithm.upper()}")
        print(f"  Total steps  : {total_steps}")
        print(f"  Final reward : {self.logs['reward'][-1]:.4f}")
        print(f"  Final EM     : {final_em:.4f} ({final_em*100:.1f}%)")

        # Full curriculum summary
        if "stage_summary" in self.logs:
            print(f"\n  Curriculum Summary:")
            for stage, summ in self.logs["stage_summary"].items():
                print(f"    {stage:7s} | "
                      f"Reward: {summ['avg_reward']:.3f} | "
                      f"EM: {summ['avg_em']*100:.1f}% | "
                      f"Searches: {summ['avg_searches']:.1f} | "
                      f"ToolOK: {summ['tool_success']:.3f}")
        print(f"{'='*60}")

        return self.logs

    # ── Save ─────────────────────────────────
    def save(self, checkpoint_dir=None, log_dir="logs"):
        """Save model checkpoint and training logs."""
        if checkpoint_dir is None:
            checkpoint_dir = f"checkpoints/{self.algorithm}_final"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        log_path = os.path.join(
            log_dir, f"{self.algorithm}_logs.json")
        with open(log_path, "w") as f:
            json.dump({self.algorithm: self.logs}, f, indent=2)

        print(f"\nCheckpoint -> {checkpoint_dir}/")
        print(f"Logs       -> {log_path}")
