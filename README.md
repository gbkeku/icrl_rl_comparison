# ICRL RL Algorithm Comparison

**Deep Reinforcement Learning Term Project**

> Comparing RL Optimization Algorithms Under the ICRL In-Context
> Curriculum for Tool-Augmented LLM Reasoning

---

## Overview

This project systematically compares four RL optimization algorithms —
**GRPO**, **PPO**, **REINFORCE with baseline**, and **RLOO** — within
the fixed ICRL curriculum framework (Ye et al., 2026), holding all
other variables constant.

---

## Research Question

> Does the choice of RL optimization algorithm significantly affect
> the quality and efficiency of in-context curriculum-based tool-use
> learning in large language models?

---

## Project Structure

```
icrl_rl_comparison/
├── config.py                  # Central hyperparameter config
├── main.py                    # Entry point
├── run_grpo.py                # GRPO full training run
├── run_ppo.py                 # PPO full training run
├── run_reinforce.py           # REINFORCE full training run
├── run_rloo.py                # RLOO full training run
├── compare_all.py             # Load logs + generate all figures
├── requirements.txt
├── README.md
├── algorithms/
│   ├── grpo.py                # GRPO baseline
│   ├── ppo.py                 # PPO via TRL
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
| PPO  | Proximal Policy Optimization | Schulman et al., 2017 |
| REINFORCE | Policy gradient + moving avg baseline | Williams, 1992 |
| RLOO | Leave-One-Out estimator | Kool et al., 2019 |

---

## Setup

```bash
# 1. Clone repo
git clone https://github.com/gbkeku/icrl_rl_comparison
cd icrl_rl_comparison

# 2. Create virtual environment
python3 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/RUC-NLPIR/FlashRAG.git

# 4. Install CUDA PyTorch (GPU)
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Running Experiments

```bash
# Run all algorithms one by one
python run_grpo.py        # baseline (~30-40 min)
python run_ppo.py
python run_reinforce.py
python run_rloo.py

python main.py              # runs all 4 sequentially
python main.py --algo grpo  # runs just one

# Generate comparison figures (run anytime after any training)
python compare_all.py
```

---

## Configuration

All hyperparameters in `config.py`:

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

| Component | Minimum | Used |
|-----------|---------|------|
| GPU | 6 GB VRAM | NVIDIA RTX A3000 (6.4GB) |
| RAM | 16 GB | - |
| CUDA | 11.8+ | 12.8 |

---

## References

- Ye et al. (2026) — ICRL paper: https://arxiv.org/abs/2603.08068
- Shao et al. (2024) — DeepSeekMath / GRPO
- Schulman et al. (2017) — PPO
- Ahmadian et al. (2024) — RLOO
- Jin et al. (2025) — Search-R1
