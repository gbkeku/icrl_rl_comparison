# ─────────────────────────────────────────────
# Evaluation on 5 QA benchmarks
# ─────────────────────────────────────────────
import torch
from datasets import load_dataset
from utils.rewards import extract_answer, compute_accuracy_reward
from training.trainer import build_rollout_prompt
from config import MAX_EVAL_SAMPLES, DEVICE

BENCHMARK_CONFIG = {
    "trivia_qa":  {"path": "trivia_qa",  "name": "rc", "split": "validation"},
    "hotpot_qa":  {"path": "hotpot_qa",  "name": "fullwiki", "split": "validation"},
    "2wiki":      {"path": "wiki_hop",   "name": None, "split": "validation"},
    "musique":    {"path": "musique",    "name": None, "split": "validation"},
    "bamboogle":  {"path": "bamboogle",  "name": None, "split": "test"},
}


def evaluate_model(model, tokenizer, retriever, benchmark: str) -> float:
    cfg  = BENCHMARK_CONFIG[benchmark]
    args = {"path": cfg["path"], "split": cfg["split"]}
    if cfg["name"]:
        args["name"] = cfg["name"]
    dataset = load_dataset(**args)
    dataset = dataset.select(range(min(MAX_EVAL_SAMPLES, len(dataset))))

    correct = 0
    total   = len(dataset)

    for sample in dataset:
        question  = sample["question"]
        gt_answer = sample["answer"]
        prompt    = build_rollout_prompt(question, demos=[], n_shot=0)
        inputs    = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        correct += compute_accuracy_reward(response, gt_answer)

    return round((correct / total) * 100, 2)


def evaluate_all(model, tokenizer, retriever) -> dict:
    results = {}
    for benchmark in BENCHMARK_CONFIG:
        print(f"  Evaluating {benchmark}...")
        results[benchmark] = evaluate_model(
            model, tokenizer, retriever, benchmark
        )
        print(f"    EM: {results[benchmark]:.2f}%")
    results["average"] = round(
        sum(results.values()) / len(results), 2
    )
    print(f"\n  Average EM: {results['average']:.2f}%")
    return results
