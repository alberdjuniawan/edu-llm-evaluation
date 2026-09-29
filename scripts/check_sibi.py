import os
from typing import Any

from huggingface_hub import HfApi

SIBI_DATASETS = [
    "aitf-ub-2026/ub-sr-02-sibi-all-pdf",
    "aitf-ub-2026/ub-sr-02-sibi-siswa-md-cleaned",
    "aitf-ub-2026/ub-sr-02-sibi-siswa-jsonl-cpt-final",
]


def classify_error(exc: Exception) -> tuple[str, str]:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__
    detail = f"{name}: {exc}"[:160]

    if status in (401, 403) or "Gated" in name:
        return "gated", detail

    if status == 404 or "NotFound" in name:
        return "missing", detail

    return "error", detail


def check_one(api: Any, dataset_id: str) -> tuple[str, str]:
    try:
        info = api.dataset_info(dataset_id)
    except Exception as exc:  # noqa: BLE001
        return classify_error(exc)

    siblings = getattr(info, "siblings", None) or []

    return "ok", f"{len(siblings)} files"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    print(f"auth={'token' if token else 'anonymous'}")
    api = HfApi(token=token)
    failed = 0

    for dataset_id in SIBI_DATASETS:
        status, detail = check_one(api, dataset_id)
        print(f"{dataset_id}: {status} ({detail})")

        if status != "ok":
            failed += 1

    if failed:
        raise SystemExit(1)

    print(f"reachable: {len(SIBI_DATASETS)}/{len(SIBI_DATASETS)}")


if __name__ == "__main__":
    main()
