import json
from pathlib import Path

from rlm.video.memory import PreparedVideoArtifacts
from rlm.video.types import AudioEvent, OCRSpan, SpeechSpan, VisualSummarySpan


class PreparedArtifactStore:
    """
    Save/load prepared video artifacts as a small directory of sidecar files.

    This gives preprocessing and runtime a clean handshake:
    preprocessing can write out stable JSON sidecars, while runtime can load
    them later without re-running ASR or visual summarization.
    """

    MANIFEST_FILE = "manifest.json"
    SPEECH_FILE = "speech.jsonl"
    VISUAL_FILE = "visual.jsonl"
    OCR_FILE = "ocr.jsonl"
    AUDIO_FILE = "audio.jsonl"

    def save(self, artifacts: PreparedVideoArtifacts, directory: str | Path) -> Path:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "video_id": artifacts.video_id,
            "duration_seconds": artifacts.duration_seconds,
            "metadata": dict(artifacts.metadata),
            "counts": {
                "speech_spans": len(artifacts.speech_spans),
                "visual_summaries": len(artifacts.visual_summaries),
                "ocr_spans": len(artifacts.ocr_spans),
                "audio_events": len(artifacts.audio_events),
            },
            "schema_version": 1,
        }
        (output_dir / self.MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        self._write_jsonl(output_dir / self.SPEECH_FILE, artifacts.speech_spans)
        self._write_jsonl(output_dir / self.VISUAL_FILE, artifacts.visual_summaries)
        self._write_jsonl(output_dir / self.OCR_FILE, artifacts.ocr_spans)
        self._write_jsonl(output_dir / self.AUDIO_FILE, artifacts.audio_events)
        return output_dir

    def load(self, directory: str | Path) -> PreparedVideoArtifacts:
        input_dir = Path(directory)
        manifest = json.loads((input_dir / self.MANIFEST_FILE).read_text(encoding="utf-8"))
        return PreparedVideoArtifacts(
            video_id=manifest["video_id"],
            duration_seconds=float(manifest["duration_seconds"]),
            speech_spans=self._read_jsonl(input_dir / self.SPEECH_FILE, SpeechSpan),
            visual_summaries=self._read_jsonl(input_dir / self.VISUAL_FILE, VisualSummarySpan),
            ocr_spans=self._read_jsonl(input_dir / self.OCR_FILE, OCRSpan),
            audio_events=self._read_jsonl(input_dir / self.AUDIO_FILE, AudioEvent),
            metadata=dict(manifest.get("metadata", {})),
        )

    def _write_jsonl(self, path: Path, items: list[object]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                json.dump(item.to_dict(), handle)
                handle.write("\n")

    def _read_jsonl(self, path: Path, item_type) -> list[object]:
        if not path.exists():
            return []
        items = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(item_type.from_dict(json.loads(line)))
        return items
