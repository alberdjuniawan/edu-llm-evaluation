"""Print what to paste into configs/models.yaml so a model identity is reproducible.

HF repo id  -> current commit sha (network; run right after you finish evaluating, and
               note it was resolved after the run if the cache is gone)
local dir   -> sha256 over all weight files (sorted), plus per-file sizes
"""

import argparse
import hashlib
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")


def hash_local_weights(directory: Path) -> tuple[str, list[tuple[str, int]]]:
    files = sorted(p for p in directory.rglob("*") if p.suffix in WEIGHT_SUFFIXES)

    if not files:
        raise FileNotFoundError(f"No weight files under {directory}.")

    digest = hashlib.sha256()
    listing: list[tuple[str, int]] = []

    for path in files:
        relative = str(path.relative_to(directory))
        digest.update(relative.encode())

        with path.open("rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                digest.update(block)

        listing.append((relative, path.stat().st_size))

    return digest.hexdigest(), listing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="HF repo id or local directory")
    args = parser.parse_args()
    local = Path(args.source).expanduser()

    if local.is_dir():
        sha, listing = hash_local_weights(local)
        print(f"weights_sha256: {sha}")

        for name, size in listing:
            print(f"  {name}  {size} bytes")
    else:
        from huggingface_hub import HfApi

        info = HfApi().model_info(args.source)
        print(f"revision: {info.sha}")
        print(f"last_modified: {info.last_modified}")


if __name__ == "__main__":
    main()
