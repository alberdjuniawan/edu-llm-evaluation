import argparse
import json
from pathlib import Path

COMMON_PATHS = [("seed",)]

TRACK_PATHS = {
    "generation": [
        ("generation", "prompt_version"),
        ("generation", "max_new_tokens"),
        ("generation", "do_sample"),
        ("generation", "use_cache"),
        ("generation", "enable_thinking"),
        ("dataset", "dataset_hash"),
    ],
    "knowledge": [
        ("scoring", "score_mode"),
        ("scoring", "score_span"),
        ("scoring", "letter_style"),
        ("scoring", "prompt_format"),
        ("scoring", "exclude_fewshot"),
        ("scoring", "limit"),
        ("dataset", "primary_questions"),
        ("dataset", "retention_questions"),
    ],
}

WARN_PATHS = [
    ("dtype",),
    ("scoring", "batch_size"),
    ("runtime", "torch"),
    ("runtime", "transformers"),
]


def _get(metadata: dict, *path: str):
    node = metadata

    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None, False

        node = node[key]

    return node, True


def detect_track(metadata: dict) -> str:
    if "generation" in metadata:
        return "generation"

    if "scoring" in metadata:
        return "knowledge"

    return "unknown"


def check_consistency(labeled: list[tuple[str, dict]]) -> tuple[list[str], list[str]]:
    if not labeled:
        raise ValueError("No metadata provided.")

    errors: list[str] = []
    warnings: list[str] = []
    first_label, first = labeled[0]
    track = detect_track(first)

    if track == "unknown":
        return [f"{first_label}: bukan metadata knowledge maupun generation."], []

    for label, metadata in labeled:
        if detect_track(metadata) != track:
            errors.append(f"{label}: track {detect_track(metadata)} != {track}.")

    for label, metadata in labeled:
        revision, _ = _get(metadata, "model_revision")
        weights, _ = _get(metadata, "model_weights_sha256")

        if not revision and not weights:
            errors.append(
                f"{label}: identitas model belum di-pin "
                "(model_revision dan model_weights_sha256 kosong)."
            )

    if errors:
        return errors, warnings

    for label, metadata in labeled[1:]:
        for path in COMMON_PATHS + TRACK_PATHS[track]:
            want, found = _get(first, *path)
            got, found_got = _get(metadata, *path)
            name = ".".join(path)

            if not found or not found_got:
                errors.append(f"{label}: missing {name}.")
            elif got != want:
                errors.append(f"{label}: {name}={got!r} != {first_label} {want!r}.")

        for path in WARN_PATHS:
            want, _ = _get(first, *path)
            got, _ = _get(metadata, *path)

            if got != want:
                warnings.append(
                    f"{label}: {'.'.join(path)}={got!r} differs from {want!r}."
                )

    for path in COMMON_PATHS + TRACK_PATHS[track]:
        if not _get(first, *path)[1]:
            errors.append(f"{first_label}: missing {'.'.join(path)}.")

    commits: dict[str, list[str]] = {}

    for label, metadata in labeled:
        value, _ = _get(metadata, "git_commit")
        commits.setdefault(value if isinstance(value, str) else "missing", []).append(
            label
        )

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
