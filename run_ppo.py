# run_ppo.py
# ─────────────────────────────────────────────
# PPO training — uses ICRLTrainer exclusively
# ✅ Correct ICRL: tool loop + retrieval masking
# ─────────────────────────────────────────────
import json
import sys
sys.path.insert(0, ".")

from training.trainer import ICRLTrainer
from environment.retriever import MockRetriever
from utils.plot_training import plot_training_dynamics


def main():
    # Load data
    demos      = json.load(open("data/demos.json"))
    train_data = json.load(open("data/nq_train.json"))
    retriever  = MockRetriever()

    # Build trainer — all logic lives in ICRLTrainer
    trainer = ICRLTrainer(
        algorithm  = "ppo",
        retriever  = retriever,
        train_data = train_data,
        demos      = demos,
    )

    # Run full 3->2->0 curriculum
    logs = trainer.train()

    # Save checkpoint + logs
    trainer.save()

    # Plot training dynamics for this algorithm
    # Pass logs nested under algo key as plot function expects
    print("\nGenerating training plot...")
    plot_training_dynamics(
        logs            = {"ppo": logs},
        steps_per_stage = 50,
        save_path       = "figures/ppo_training.png"
    )

    print("\nDone! Run compare_all.py after all algorithms finish.")


if __name__ == "__main__":
    main()
