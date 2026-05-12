import json
import os
import uuid
from datetime import datetime
from typing import Any

from rlm.video.types import TraceStep


class VideoRLMLogger:
    def __init__(
        self,
        log_dir: str | None = None,
        file_name: str = "videorlm",
        console: bool = False,
    ):
        self._save_to_disk = log_dir is not None
        self.console = console
        self.log_dir = log_dir
        self.log_file_path: str | None = None
        if self._save_to_disk and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = str(uuid.uuid4())[:8]
            self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")

        self._metadata: dict[str, Any] | None = None
        self._steps: list[dict[str, Any]] = []

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        self._metadata = dict(metadata)
        entry = {"type": "metadata", "timestamp": datetime.now().isoformat(), **metadata}
        self._print_entry(entry)
        self._write_entry(entry)

    def log_step(self, step: TraceStep) -> None:
        entry = {"type": "step", "timestamp": datetime.now().isoformat(), **step.to_dict()}
        self._steps.append(entry)
        self._print_entry(entry)
        self._write_entry(entry)

    def clear_steps(self) -> None:
        self._steps = []

    def get_trace(self) -> dict[str, Any] | None:
        if self._metadata is None:
            return None
        return {"metadata": dict(self._metadata), "steps": list(self._steps)}

    def _write_entry(self, entry: dict[str, Any]) -> None:
        if not self._save_to_disk or not self.log_file_path:
            return
        with open(self.log_file_path, "a", encoding="utf-8") as handle:
            json.dump(entry, handle)
            handle.write("\n")

    def _print_entry(self, entry: dict[str, Any]) -> None:
        if not self.console:
            return
        if entry["type"] == "metadata":
            print(
                "[VideoRLM] run "
                f"model={entry.get('controller_model')} "
                f"video_id={entry.get('video_id')} "
                f"max_steps={entry.get('max_steps')} "
                f"search_top_k={entry.get('search_top_k')}",
                flush=True,
            )
            return
        if entry["type"] != "step":
            return

        action = entry.get("action", {})
        observation = entry.get("observation", {})
        action_parts = [
            f"step={entry.get('step_index')}",
            f"action={action.get('action_type')}",
        ]
        for key in ("query", "modality", "node_id", "target_slot"):
            value = action.get(key)
            if value:
                action_parts.append(f"{key}={value}")
        print("[VideoRLM] " + " ".join(action_parts), flush=True)
        print(
            "[VideoRLM] observation "
            f"kind={observation.get('kind')} "
            f"summary={observation.get('summary')}",
            flush=True,
        )
