#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_REPO = "OpenGVLab/InternVideo2-Stage2_6B"


def sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", repo_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot into output/models."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo id.")
    parser.add_argument(
        "--output-root",
        default="output/models",
        help="Directory where local model snapshots are stored.",
    )
    parser.add_argument("--revision", help="Optional model repo revision.")
    parser.add_argument(
        "--token",
        help="Optional Hugging Face token. If omitted, huggingface_hub uses its login cache.",
    )
    parser.add_argument(
        "--skip-text-encoder",
        action="store_true",
        help="Do not pre-cache the InternVideo text encoder dependency.",
    )
    parser.add_argument(
        "--text-encoder-repo",
        help="Override the text encoder repo to cache. Defaults to config.json.",
    )
    parser.add_argument(
        "--text-encoder-revision",
        help="Optional text encoder repo revision.",
    )
    parser.add_argument(
        "--mirror-text-encoder-to-output",
        action="store_true",
        help=(
            "Also download the text encoder into output/models. InternVideo still needs "
            "the Hugging Face cache because its remote code loads by repo id."
        ),
    )
    parser.add_argument(
        "--no-validate-internvideo",
        action="store_true",
        help="Skip validation for InternVideo2 config and safetensor shard files.",
    )
    return parser.parse_args()


def download_repo_to_output(
    repo_id: str,
    output_root: Path,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    target_dir = output_root / sanitize_repo_id(repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        revision=revision,
        token=token,
    )
    return target_dir


def download_repo_to_hf_cache(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, revision=revision, token=token))


def infer_internvideo_text_encoder_repo(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing InternVideo config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_encoder = config.get("model", {}).get("text_encoder", {})
    pretrained = text_encoder.get("pretrained")
    if pretrained:
        return str(pretrained)
    text_encoders = config.get("TextEncoders", {})
    bert_large = text_encoders.get("bert_large", {})
    pretrained = bert_large.get("pretrained")
    if pretrained:
        return str(pretrained)
    raise ValueError(f"Could not infer text encoder repo from {config_path}")


def validate_internvideo_model_dir(model_dir: Path) -> None:
    required_files = [
        model_dir / "config.json",
        model_dir / "modeling_internvideo2.py",
        model_dir / "configs" / "config_bert_large.json",
        model_dir / "model.safetensors.index.json",
    ]
    missing_required = [path for path in required_files if not path.is_file()]
    if missing_required:
        missing = "\n".join(str(path) for path in missing_required)
        raise FileNotFoundError(f"Incomplete InternVideo download; missing:\n{missing}")

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = set(index.get("weight_map", {}).values())
    missing_shards = sorted(name for name in shard_names if not (model_dir / name).is_file())
    if missing_shards:
        missing = "\n".join(str(model_dir / name) for name in missing_shards)
        raise FileNotFoundError(f"Incomplete InternVideo download; missing shards:\n{missing}")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model_dir = download_repo_to_output(
        args.repo,
        output_root,
        revision=args.revision,
        token=args.token,
    )
    print(f"model: {model_dir}")

    is_internvideo = "InternVideo2" in args.repo
    if is_internvideo and not args.no_validate_internvideo:
        validate_internvideo_model_dir(model_dir)
        print("validated: InternVideo2 config and safetensor shards are present")

    if is_internvideo and not args.skip_text_encoder:
        text_encoder_repo = args.text_encoder_repo or infer_internvideo_text_encoder_repo(
            model_dir
        )
        cache_dir = download_repo_to_hf_cache(
            text_encoder_repo,
            revision=args.text_encoder_revision,
            token=args.token,
        )
        print(f"text_encoder_cache: {cache_dir}")
        if args.mirror_text_encoder_to_output:
            text_encoder_dir = download_repo_to_output(
                text_encoder_repo,
                output_root,
                revision=args.text_encoder_revision,
                token=args.token,
            )
            print(f"text_encoder_output: {text_encoder_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
