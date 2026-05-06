# main.py — Master runner for all 4 algorithms
import subprocess
import sys
import argparse

ALGORITHMS = ["grpo", "ppo", "reinforce", "rloo"]

def run(algo):
    print(f"\n{'='*60}")
    print(f"  Starting {algo.upper()} training...")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, f"run_{algo}.py"],
        check=True
    )
    return result.returncode

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=ALGORITHMS + ["all"], default="all")
args = parser.parse_args()

algos = ALGORITHMS if args.algo == "all" else [args.algo]
for algo in algos:
    run(algo)

print("\nAll done! Run: python compare_all.py")