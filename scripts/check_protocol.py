import argparse
import json
from pathlib import Path

EQUAL_PATHS = [
    ("generation", "prompt_version"),
    ("generation", "max_new_tokens"),
    ("generation", "do_sample"),
    ("generation", "use_cache"),
    ("dataset", "dataset_hash"),
    ("seed",),
]


def _get(metadata: dict, *path: str):
    node = metadata

    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None, False

        node = node[key]

    return node, True


def check_consistency(labeled: list[tuple[str, dict]]) -> tuple[list[str], list[str]]:
    if not labeled:
        raise ValueError("No metadata provided.")

    errors: list[str] = []
    warnings: list[str] = []
    first_label, first = labeled[0]

    for label, metadata in labeled[1:]:
        for path in EQUAL_PATHS:
            want, found = _get(first, *path)
            got, found_got = _get(metadata, *path)
            name = ".".join(path)

            if not found or not found_got:
                errors.append(f"{label}: missing {name}.")
            elif got != want:
                errors.append(f"{label}: {name}={got!r} != {first_label} {want!r}.")

    commits: dict[str | None, list[str]] = {}

    for label, metadata in labeled:
        value, _ = _get(metadata, "git_commit")
        commits.setdefault(value, []).append(label)

    if len(commits) > 1:
        warnings.append(f"git_commit differs across runs: {commits}.")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Check protocol consistency.")
    parser.add_argument("metadata", nargs="+", help="run_metadata.json files.")
    args = parser.parse_args()

    labeled = []

    for raw in args.metadata:
        path = Path(raw)

        if not path.exists():
            print(f"ERROR: missing {path}.")

            raise SystemExit(1)

        with path.open("r", encoding="utf-8") as file:
            content = json.load(file)

        if not isinstance(content, dict):
            raise TypeError(f"Invalid metadata at {path}.")

        model, _ = _get(content, "model_id")
        revision, _ = _get(content, "model_revision")
        print(f"{path}: model={model} revision={revision}")
        labeled.append((str(path), content))

    errors, warnings = check_consistency(labeled)

    for warning in warnings:
        print(f"WARN: {warning}")

    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        raise SystemExit(1)

    print(f"consistent: {len(labeled)} runs")


if __name__ == "__main__":
    main()
