# run_reinforce.py
# Full REINFORCE curriculum training: 3→2→0 shot
import json, torch, random, os, sys
sys.path.insert(0, '.')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from algorithms.reinforce import ReinforceWithBaseline
from utils.rewards import compute_reward, extract_answer
from training.trainer import build_rollout_prompt
from config import MODEL_NAME, DEVICE, KL_COEFF, CURRICULUM, STEPS_PER_STAGE

print("=" * 60)
print("  ICRL — REINFORCE Full Curriculum Training")
print("  Schedule: 3-shot → 2-shot → 0-shot")
print("=" * 60)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def vram():
    u = torch.cuda.memory_allocated() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{u:.2f}/{t:.2f} GB"

def clear():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print("\nLoading policy model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16,
    device_map=DEVICE, low_cpu_mem_usage=True,
)
model.gradient_checkpointing_enable()

print("Loading reference model (4-bit)...")
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    device_map=DEVICE, low_cpu_mem_usage=True,
)
for p in ref_model.parameters():
    p.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

optimizer  = torch.optim.AdamW(
    model.parameters(), lr=1e-5, eps=1e-5
)
rf_agent   = ReinforceWithBaseline(alpha=0.99)

print(f"Models ready | VRAM: {vram()}\n")

demos      = json.load(open('data/demos.json'))
train_data = json.load(open('data/nq_train.json'))

NUM_ROLLOUTS = 4
MAX_LEN      = 1024
MAX_NEW      = 150


def generate_rollout(question, n_shot):
    prompt = build_rollout_prompt(question, demos, n_shot)
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_LEN
    ).to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW,
            do_sample=True, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    del inputs, output
    clear()
    return response


def compute_reinforce_loss(question, response, advantage, n_shot):
    prompt   = build_rollout_prompt(question, demos, n_shot)
    full_seq = prompt + response
    enc      = tokenizer(
        full_seq, return_tensors="pt",
        truncation=True, max_length=MAX_LEN
    ).to(DEVICE)
    prompt_len = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_LEN
    )["input_ids"].shape[1]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out    = model(**enc)
        logits = out.logits
    log_probs = torch.nn.functional.log_softmax(
        logits[:, :-1, :], dim=-1
    )
    ids      = enc["input_ids"][:, 1:]
    token_lp = log_probs.gather(
        2, ids.unsqueeze(-1)
    ).squeeze(-1)[0]

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref_out    = ref_model(**enc)
            ref_logits = ref_out.logits
        ref_lp = torch.nn.functional.log_softmax(
            ref_logits[:, :-1, :], dim=-1
        ).gather(2, ids.unsqueeze(-1)).squeeze(-1)[0]

    mask  = torch.zeros(token_lp.shape[0], device=DEVICE)
    mask[min(prompt_len - 1, mask.shape[0] - 1):] = 1.0
    n_tok = mask.sum().clamp(min=1)

    adv_t = torch.tensor(
        advantage, device=DEVICE, dtype=torch.bfloat16
    )
    policy_loss = -(token_lp * adv_t * mask).sum() / n_tok
    kl_loss     = KL_COEFF * (
        (token_lp - ref_lp) * mask
    ).sum() / n_tok

    loss = (policy_loss + kl_loss) / NUM_ROLLOUTS

    del enc, out, logits, log_probs, token_lp
    del ref_out, ref_logits, ref_lp, mask
    clear()
    return loss


all_logs = {
    "reinforce": {
        "reward": [], "response_length": [],
        "valid_searches": [], "stage": [],
    }
}

stage_names = {3: "3-shot", 2: "2-shot", 0: "0-shot"}
total_steps = 0

for n_shot in CURRICULUM:
    print(f"\n{'─'*60}")
    print(f"  Stage: {stage_names[n_shot]}  ({STEPS_PER_STAGE} steps)")
    print(f"{'─'*60}")

    for step in range(STEPS_PER_STAGE):
        sample    = random.choice(train_data)
        question  = sample["question"]
        gt_answer = sample["answer"]

        responses  = []
        rewards    = []
        for _ in range(NUM_ROLLOUTS):
            resp   = generate_rollout(question, n_shot)
            reward = compute_reward(resp, gt_answer)
            responses.append(resp)
            rewards.append(reward)

        # Update baseline and compute advantages
        avg_reward = sum(rewards) / len(rewards)
        rf_agent.update_baseline(avg_reward)
        advantages = [
            rf_agent.compute_advantage(r) for r in rewards
        ]

        optimizer.zero_grad()
        for resp, adv in zip(responses, advantages):
            loss = compute_reinforce_loss(
                question, resp, adv, n_shot
            )
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        clear()

        avg_len      = sum(len(r) for r in responses) / NUM_ROLLOUTS
        avg_searches = sum(
            r.count("<search>") for r in responses
        ) / NUM_ROLLOUTS

        all_logs["reinforce"]["reward"].append(avg_reward)
        all_logs["reinforce"]["response_length"].append(avg_len)
        all_logs["reinforce"]["valid_searches"].append(avg_searches)
        all_logs["reinforce"]["stage"].append(n_shot)
        total_steps += 1

        if step % 10 == 0 or step == STEPS_PER_STAGE - 1:
            preds = [extract_answer(r) for r in responses]
            print(
                f"  Step {step:3d}/{STEPS_PER_STAGE} | "
                f"Reward: {avg_reward:.3f} | "
                f"Searches: {avg_searches:.1f} | "
                f"Preds: {preds}"
            )

    print(f"  Stage {stage_names[n_shot]} complete!")

print("\nSaving checkpoint...")
model.save_pretrained("checkpoints/reinforce_final")
tokenizer.save_pretrained("checkpoints/reinforce_final")

with open("logs/reinforce_logs.json", "w") as f:
    json.dump(all_logs, f, indent=2)

print("Checkpoint → checkpoints/reinforce_final/")
print("Logs       → logs/reinforce_logs.json")
print()
print("=" * 60)
print("  REINFORCE Training Complete! ✅")
print(f"  Total steps: {total_steps}")
print("=" * 60)