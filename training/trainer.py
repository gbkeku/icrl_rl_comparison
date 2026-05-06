# ─────────────────────────────────────────────
# Unified ICRL Trainer
# ─────────────────────────────────────────────
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import (
    MODEL_NAME, DTYPE, DEVICE, CURRICULUM,
    STEPS_PER_STAGE, NUM_ROLLOUTS,
    MAX_PROMPT_LENGTH, TEMPERATURE
)
from utils.rewards import compute_reward
from algorithms.grpo import compute_grpo_advantages
from algorithms.rloo import compute_rloo_advantages
from algorithms.reinforce import ReinforceWithBaseline


def load_model(model_name: str = MODEL_NAME):
    dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_rollout_prompt(question: str, demos: list, n_shot: int) -> str:
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


class ICRLTrainer:
    def __init__(self, algorithm, retriever, train_data, demos):
        self.algorithm  = algorithm
        self.retriever  = retriever
        self.train_data = train_data
        self.demos      = demos
        self.model, self.tokenizer = load_model()
        self.ref_model, _ = load_model()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5
        )
        if algorithm == "reinforce":
            self.reinforce = ReinforceWithBaseline()
        self.logs = {
            "reward": [], "response_length": [], "valid_searches": []
        }

    def run_rollout(self, question: str, n_shot: int) -> dict:
        prompt = build_rollout_prompt(question, self.demos, n_shot)
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_PROMPT_LENGTH
        ).to(DEVICE)
        response_text   = ""
        retrieval_spans = []

        for _ in range(6):
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, max_new_tokens=256,
                    temperature=TEMPERATURE, do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_text = self.tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            response_text += new_text

            import re
            if "<search>" in new_text:
                query = re.search(
                    r"<search>(.*?)</search>", new_text, re.DOTALL
                )
                if query:
                    q    = query.group(1).strip()
                    docs = self.retriever.retrieve(q)
                    info = f"\n<information>{docs}</information>\n"
                    span_start = len(response_text)
                    response_text += info
                    retrieval_spans.append(
                        (span_start, span_start + len(info))
                    )
                    new_input = prompt + response_text
                    inputs    = self.tokenizer(
                        new_input, return_tensors="pt",
                        truncation=True, max_length=MAX_PROMPT_LENGTH
                    ).to(DEVICE)

            if "<answer>" in response_text:
                break

        return {"response": response_text, "retrieval_spans": retrieval_spans}

    def train_stage(self, n_shot: int):
        print(f"\n--- Curriculum stage: {n_shot}-shot ---")
        for step in range(STEPS_PER_STAGE):
            sample    = random.choice(self.train_data)
            question  = sample["question"]
            gt_answer = sample["answer"]
            rollouts  = [self.run_rollout(question, n_shot)
                         for _ in range(NUM_ROLLOUTS)]
            rewards   = [compute_reward(r["response"], gt_answer)
                         for r in rollouts]
            avg_reward   = sum(rewards) / len(rewards)
            avg_len      = sum(len(r["response"]) for r in rollouts) / NUM_ROLLOUTS
            avg_searches = sum(
                r["response"].count("<search>") for r in rollouts
            ) / NUM_ROLLOUTS
            self.logs["reward"].append(avg_reward)
            self.logs["response_length"].append(avg_len)
            self.logs["valid_searches"].append(avg_searches)
            if step % 10 == 0:
                print(f"  Step {step:3d} | Reward: {avg_reward:.3f} | "
                      f"Searches: {avg_searches:.2f}")

    def train(self):
        for n_shot in CURRICULUM:
            self.train_stage(n_shot)
        print("\nTraining complete.")
        return self.logs
