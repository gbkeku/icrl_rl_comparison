# ICRL RL Algorithm Comparison

**Deep Reinforcement Learning Term Project**

> Comparing RL Optimization Algorithms under the ICRL In-Context Curriculum for Tool-Augmented LLM Reasoning

---

## Overview

This project compares four RL optimization algorithms — **GRPO**, **PPO**, **REINFORCE with baseline**, and **RLOO** — within the same **In-Context Reinforcement Learning (ICRL)** framework. The goal is to isolate the effect of the optimizer while keeping the **model**, **retriever**, **reward**, **training data**, and **curriculum** fixed.

Unlike a setup-only repository, this repo includes the **current comparison results**, **stage-level analysis**, and the scripts used to reproduce the plots and logs.

---

## Research Question

> Does the choice of RL optimization algorithm significantly affect the quality and efficiency of in-context curriculum-based tool-use learning in large language models?

---

## What is ICRL?

**In-Context Reinforcement Learning (ICRL)** trains an LLM to use external tools through **reinforcement learning only**, without supervised fine-tuning.

The model is prompted with a curriculum of demonstrations that are gradually removed:

- **3-shot**: the model imitates tool-use demonstrations
- **2-shot**: the scaffold is reduced
- **0-shot**: the model must use tools autonomously

A single rollout follows this pattern:

`Query -> Prompt(k-shot) -> LLM -> <search> -> Retriever -> <information> -> LLM -> <answer> -> Reward -> RL update`

**Important implementation detail:** retrieved `<information>` tokens are **masked out of the policy gradient**, so only model-generated tokens contribute to learning.

---

## Current Results

These are the **current project results** under the setup implemented in this repository:

- **Backbone:** Qwen2.5-1.5B-Instruct
- **Curriculum:** `3 -> 2 -> 0`
- **Training:** Natural Questions subset with BM25 retrieval
- **Metric:** Exact Match (EM) accuracy
- **Compute:** NVIDIA RTX A3000 (6.4 GB VRAM)

### Main Comparison: Average EM (%)

| Algorithm | Average EM | Summary |
|-----------|-----------:|---------|
| **PPO** | **45.62** | Best overall final accuracy |
| **RLOO** | **43.60** | Most stable across the curriculum |
| **GRPO** | **32.82** | Consistent but underperforms PPO/RLOO |
| **REINFORCE** | **0.80** | Collapses to near-zero performance |

### Per-Benchmark EM (%)

| Algorithm | TriviaQA | HotpotQA | 2Wiki | Musique | Bamboogle | Average |
|-----------|---------:|---------:|------:|--------:|----------:|--------:|
| **PPO** | 64.7 | 42.0 | 46.9 | 29.2 | 45.3 | **45.62** |
| **RLOO** | 65.3 | 37.7 | 44.9 | 27.2 | 42.9 | **43.60** |
| **GRPO** | 48.0 | 28.1 | 33.1 | 20.0 | 34.9 | **32.82** |
| **REINFORCE** | 2.2 | 2.8 | ~0.0 | 0.7 | ~0.0 | **0.80** |

> **Key result:** PPO achieves the best average EM, while RLOO is the most curriculum-stable. REINFORCE fails to maintain tool use and collapses.

---

## Stage-Level Performance

How performance changes as demonstrations are removed:

| Algorithm | 3-shot | 2-shot | 0-shot | Interpretation |
|-----------|-------:|-------:|-------:|----------------|
| **PPO** | 56.0 | 52.5 | 35.5 | Strong early performance, fragile at 0-shot |
| **RLOO** | 50.0 | 47.0 | 43.0 | Best stability across curriculum transitions |
| **GRPO** | 47.0 | 47.0 | 40.0 | Stable but lower ceiling |
| **REINFORCE** | 16.0 | 2.0 | 0.0 | Policy collapse |

### Reward Trend by Stage

| Algorithm | 3-shot Reward | 2-shot Reward | 0-shot Reward |
|-----------|--------------:|--------------:|--------------:|
| **PPO** | 0.61 | 0.57 | 0.42 |
| **RLOO** | 0.55 | 0.53 | 0.50 |
| **GRPO** | 0.54 | 0.53 | 0.47 |
| **REINFORCE** | 0.26 | 0.12 | 0.06 |

**Interpretation:**

- **PPO** gives the highest peak performance when demonstrations are still present.
- **RLOO** degrades the least as the curriculum transitions to autonomy.
- **GRPO** is stable but less competitive than PPO and RLOO in this implementation.
- **REINFORCE** cannot adapt to the curriculum shifts and abandons the tool.

---

## Tool-Use Analysis

Tool use was tracked explicitly during training:

- `avg_searches`: average number of search calls per rollout
- `tool_success`: search used **and** final answer correct

### Average Search Calls per Stage

| Algorithm | 3-shot | 2-shot | 0-shot |
|-----------|-------:|-------:|-------:|
| **GRPO** | 1.52 | 1.39 | 0.89 |
| **PPO** | 1.03 | 1.31 | 0.70 |
| **REINFORCE** | 0.95 | 0.08 | 0.01 |
| **RLOO** | 1.97 | 0.61 | 0.74 |

### Tool Success Rate (%) per Stage

| Algorithm | 3-shot | 2-shot | 0-shot |
|-----------|-------:|-------:|-------:|
| **GRPO** | 36.0 | 29.5 | 33.5 |
| **PPO** | 47.0 | 34.0 | 24.0 |
| **REINFORCE** | 13.0 | 0.5 | 0.0 |
| **RLOO** | 30.5 | 15.5 | 27.0 |

> **Key diagnostic:** tool usage is a leading indicator of policy collapse. REINFORCE stops calling the retriever almost entirely by 2-shot, which explains its near-zero EM.

---

## Key Findings

1. **PPO wins overall** on average EM.
2. **RLOO is the most stable** across the full 3-shot -> 2-shot -> 0-shot curriculum.
3. **GRPO remains consistent**, but underperforms PPO and RLOO in this implementation.
4. **REINFORCE collapses** because its moving-average baseline adapts too slowly to curriculum transitions.
5. **Algorithm choice is not a minor optimization detail** — it determines whether tool-use learning succeeds or fails.

---

## Repository Structure

```text
icrl_rl_comparison/
├── config.py                  # Central hyperparameter config
├── run_grpo.py                # GRPO full training run
├── run_ppo.py                 # PPO full training run
├── run_reinforce.py           # REINFORCE full training run
├── run_rloo.py                # RLOO full training run
├── compare_all.py             # Load logs + generate all figures
├── requirements.txt
├── README.md
├── algorithms/
│   ├── grpo.py                # GRPO baseline
│   ├── ppo.py                 # PPO implementation
│   ├── reinforce.py           # REINFORCE + moving avg baseline
│   └── rloo.py                # Leave-One-Out estimator
├── training/
│   └── trainer.py             # Unified ICRL trainer + curriculum
├── evaluation/
│   └── evaluator.py           # EM evaluation on 5 QA benchmarks
├── environment/
│   └── retriever.py           # BM25 + MockRetriever
├── utils/
│   ├── rewards.py             # Reward computation + loss masking
│   ├── plot_training.py       # Training dynamics plots
│   └── plot_results.py        # Results table + bar charts
├── data/
│   ├── demos.json             # 3 fixed few-shot demonstrations
│   └── nq_train.json          # NQ training samples (mock)
├── figures/                   # Generated plots
└── logs/                      # Training logs + results
```

---

## Algorithms Compared

| Algorithm | Type | Reference |
|-----------|------|-----------|
| GRPO | Group Relative Policy Optimization | Shao et al., 2024 |
| PPO | Proximal Policy Optimization | Schulman et al., 2017 |
| REINFORCE | Policy gradient + moving average baseline | Williams, 1992 |
| RLOO | Leave-One-Out estimator | Kool et al., 2019 |

---

## Setup

```bash
# 1. Clone repo
git clone https://github.com/gbkeku/icrl_rl_comparison
cd icrl_rl_comparison

# 2. Create virtual environment
python3 -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/RUC-NLPIR/FlashRAG.git

# 4. Install CUDA PyTorch if using GPU
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If you want to run on CPU, set the device in `config.py`:

```python
DEVICE = "cpu"
```

---

## Running Experiments

```bash
# Run all algorithms one by one
python run_grpo.py
python run_ppo.py
python run_reinforce.py
python run_rloo.py

# Generate comparison figures after logs are produced
python compare_all.py
```

Outputs:

- training logs -> `logs/`
- checkpoints -> `checkpoints/`
- figures -> `figures/`

---

## Reproducing the Reported Results

The reported comparison numbers above come from the current project configuration:

- `MODEL_NAME = Qwen2.5-1.5B-Instruct`
- `CURRICULUM = [3, 2, 0]`
- `STEPS_PER_STAGE = 50`
- `NUM_ROLLOUTS = 4`
- reward = `0.8 * accuracy + 0.2 * format`

To reproduce the comparison workflow:

```bash
python run_grpo.py
python run_ppo.py
python run_reinforce.py
python run_rloo.py
python compare_all.py
```

---

## Configuration

All main hyperparameters are defined in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| MODEL_NAME | Qwen2.5-1.5B-Instruct | LLM backbone |
| CURRICULUM | [3, 2, 0] | Shot counts per stage |
| STEPS_PER_STAGE | 50 | RL steps per stage |
| NUM_ROLLOUTS | 4 | Trajectories per query |
| LEARNING_RATE | 1e-5 | AdamW learning rate |
| KL_COEFF | 0.001 | KL penalty coefficient |
| ALPHA | 0.8 | Accuracy weight in reward |

---

## Evaluation Benchmarks

| Benchmark | Type | Samples |
|-----------|------|---------|
| TriviaQA | Single-hop, factual | 500 |
| HotpotQA | Multi-hop, Wikipedia | 500 |
| 2WikiMultiHopQA | Multi-hop, cross-doc | 500 |
| Musique | Multi-hop, compositional | 500 |
| Bamboogle | Out-of-domain, adversarial | 500 |

**Metric:** Exact Match (EM) accuracy (%)

---

## Hardware Requirements

| Component | Minimum | Used in this project |
|-----------|---------|----------------------|
| GPU | 6 GB VRAM | NVIDIA RTX A3000 (6.4 GB) |
| RAM | 16 GB | 16+ GB recommended |
| CUDA | 11.8+ | 12.8 |

---

## Limitations

These results should be interpreted in the context of the current project scope:

- **Small-scale training:** only 150 steps total (50 per stage)
- **Mock retrieval setup:** not a full Wikipedia-scale index
- **Single backbone:** only Qwen2.5-1.5B-Instruct tested
- **Approximate / implementation-level constraints:** final rankings may change with more compute, larger models, or full-scale retrieval

---

## Repository

GitHub: `https://github.com/gbkeku/icrl_rl_comparison`

This repository includes:

- full ICRL trainer
- algorithm-specific run scripts
- logs and plotting utilities
- evaluation scripts
- current comparison results

---

## References

- Ye et al. (2026) — ICRL: In-Context Reinforcement Learning for Tool Use in LLMs
- Shao et al. (2024) — DeepSeekMath / GRPO
- Schulman et al. (2017) — Proximal Policy Optimization
- Ahmadian et al. (2024) — RLOO
- Kool et al. (2019) — Leave-One-Out policy gradient baseline
- Jin et al. (2025) — Search-R1
- Kwiatkowski et al. (2019) — Natural Questions
