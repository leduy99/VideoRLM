from __future__ import annotations

import re
from pathlib import Path

from rlm.video.media import get_repo_output_root


def get_model_output_root() -> Path:
    output_root = get_repo_output_root() / "models"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", repo_id)


def default_local_model_dir(repo_id: str) -> Path:
    model_dir = get_model_output_root() / sanitize_repo_id(repo_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def download_snapshot(
    repo_id: str,
    *,
    local_dir: str | Path | None = None,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    target_dir = Path(local_dir) if local_dir is not None else default_local_model_dir(repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        revision=revision,
        allow_patterns=allow_patterns,
    )
    return target_dir
