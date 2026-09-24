"""Read-only cloud policy audit with hash-bound mounted input and unique artifact ID."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def locate_checkpoint(roots, digest):
    matches = []
    for root in roots:
        if root.is_dir():
            for path in root.rglob("model_1000*.pt"):
                if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                    matches.append(path.resolve())
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError("No mounted model1000 matches the requested SHA256")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-task", choices=("TASK_20260924_059",), required=True)
    args = parser.parse_args()
    checkpoint = locate_checkpoint((ROOT, Path("/workspace"), Path("/personal")), args.checkpoint_sha256)
    output = ROOT/"logs/f1_amp_audit/exported_data/source059/model_9901000.pt"
    # Different numeric ID from input checkpoint avoids platform artifact aliasing.
    subprocess.run([sys.executable, str(ROOT/"humanoid/scripts/record_amp_policy.py"),
        "--checkpoint-file", str(checkpoint), "--checkpoint-sha256", args.checkpoint_sha256,
        "--expected-commit", args.expected_commit, "--experiment", "recovery",
        "--output", str(output), "--duration", "20", "--num_envs", "16", "--seed", "5",
        "--headless", "--task", "f1_amp_walk02_recovery", "--sim_device", "cuda:0",
        "--rl_device", "cuda:0"], check=True, cwd=str(ROOT), timeout=900)
    print("[amp-mounted-audit-complete] source=%s artifact=%s" % (args.source_task, output.name), flush=True)


if __name__ == "__main__":
    main()
