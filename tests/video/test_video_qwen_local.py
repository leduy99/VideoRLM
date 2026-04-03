from pathlib import Path

import rlm.video.qwen as video_qwen


def test_local_model_config_download_uses_snapshot(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(video_qwen, "download_snapshot", lambda repo_id, local_dir=None: tmp_path / "model")
    config = video_qwen.LocalModelConfig(model_name="Qwen/Qwen3-8B", model_path=str(tmp_path / "model"))

    path = config.download()

    assert path == str(tmp_path / "model")


def test_qwen_local_stack_default_uses_official_model_ids():
    config = video_qwen.QwenLocalVideoStackConfig.default()

    assert config.controller.model_name == "Qwen/Qwen3-8B"
    assert config.visual.model_name == "Qwen/Qwen3-VL-8B-Instruct"
    assert config.speech.model_name == "Qwen/Qwen3-ASR-0.6B"
    assert config.forced_aligner is not None
    assert "output/models" in config.controller.model_path
