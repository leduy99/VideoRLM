from pathlib import Path

import rlm.video.huggingface as video_hf


def test_sanitize_repo_id_replaces_slashes():
    assert video_hf.sanitize_repo_id("Qwen/Qwen3-8B") == "Qwen__Qwen3-8B"


def test_default_local_model_dir_is_under_output():
    model_dir = video_hf.default_local_model_dir("Qwen/Qwen3-8B")

    assert model_dir.exists()
    assert Path("output") in model_dir.relative_to(model_dir.parents[2]).parents


def test_download_snapshot_uses_target_dir(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    output_dir = video_hf.download_snapshot("Qwen/Qwen3-8B", local_dir=tmp_path / "model")

    assert output_dir == tmp_path / "model"
    assert captured["repo_id"] == "Qwen/Qwen3-8B"
