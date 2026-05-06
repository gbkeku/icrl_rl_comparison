# ─────────────────────────────────────────────
# Central configuration — optimized for 6.4GB VRAM
# ─────────────────────────────────────────────

MODEL_NAME          = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE              = "cuda"
DTYPE               = "bfloat16"

# Training — reduced for 6GB VRAM
LEARNING_RATE       = 1e-5
BATCH_SIZE          = 16
MAX_PROMPT_LENGTH   = 2048
MAX_RESPONSE_LENGTH = 512
NUM_ROLLOUTS        = 4
KL_COEFF            = 0.001
MAX_SEARCH_TURNS    = 4
TEMPERATURE         = 1.0

# Reward
ALPHA               = 0.8

# Curriculum
CURRICULUM          = [3, 2, 0]
STEPS_PER_STAGE     = 50

# Retriever
TOP_K_DOCS          = 3

# Evaluation
EVAL_BENCHMARKS     = [
    "trivia_qa",
    "hotpot_qa",
    "2wiki",
    "musique",
    "bamboogle",
]
MAX_EVAL_SAMPLES    = 500

# Algorithms to compare
ALGORITHMS          = ["grpo", "ppo", "reinforce", "rloo"]
