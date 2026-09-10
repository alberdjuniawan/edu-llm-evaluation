import random
import subprocess
from pathlib import Path


def seed_everything(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_commit(project_root: str | Path | None = None) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def collect_runtime() -> dict:
    runtime: dict = {}

    try:
        import torch

        runtime["torch"] = torch.__version__
        runtime["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            runtime["device_name"] = torch.cuda.get_device_name(0)
            runtime["cuda_runtime"] = torch.version.cuda
    except ImportError:
        runtime["torch"] = None
        runtime["cuda_available"] = False

    try:
        import transformers

        runtime["transformers"] = transformers.__version__
    except ImportError:
        runtime["transformers"] = None

    return runtime
