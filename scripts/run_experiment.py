import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STEPS = {
    "knowledge": "run_knowledge.py",
    "generation": "run_generation.py",
    "linguistic": "run_linguistic.py",
    "analyze": "analyze_results.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pipeline step.")
    parser.add_argument("--step", choices=sorted(STEPS), required=True)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script = SCRIPTS_DIR / STEPS[args.step]
    completed = subprocess.run(
        [sys.executable, str(script), *[a for a in args.extra if a != "--"]],
        cwd=PROJECT_ROOT,
        check=False,
    )

    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
