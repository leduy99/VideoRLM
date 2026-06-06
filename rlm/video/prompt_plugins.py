from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_PROMPT_PLUGIN_DIR = REPO_ROOT / "prompts" / "benchmark_plugins"
BUILTIN_PROMPT_PLUGINS = {
    "vrrqa": BUILTIN_PROMPT_PLUGIN_DIR / "vrrqa.json",
    "tlqa": BUILTIN_PROMPT_PLUGIN_DIR / "tlqa.json",
    "timelogic": BUILTIN_PROMPT_PLUGIN_DIR / "tlqa.json",
}


@dataclass(frozen=True)
class BenchmarkPromptPlugin:
    name: str
    instructions: tuple[str, ...]
    source: str | None = None

    def context_for_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "instructions": list(self.instructions),
            "question_id": str(sample.get("question_id", "")),
        }


def load_benchmark_prompt_plugin(
    plugin: str | None = None,
    *,
    plugin_file: str | Path | None = None,
) -> BenchmarkPromptPlugin | None:
    path = prompt_plugin_path(plugin, plugin_file=plugin_file)
    if path is None:
        return None
    return load_prompt_plugin_file(path)


def prompt_plugin_path(
    plugin: str | None = None,
    *,
    plugin_file: str | Path | None = None,
) -> Path | None:
    if plugin_file is not None:
        return Path(plugin_file)
    if plugin is None:
        return None
    plugin_name = plugin.strip().lower()
    if plugin_name in {"", "none", "off", "disabled"}:
        return None
    if plugin_name not in BUILTIN_PROMPT_PLUGINS:
        raise ValueError(
            f"Unsupported benchmark prompt plugin {plugin!r}. "
            f"Use one of: {', '.join(sorted(BUILTIN_PROMPT_PLUGINS))}, "
            "or pass --prompt-plugin-file."
        )
    return BUILTIN_PROMPT_PLUGINS[plugin_name]


def load_prompt_plugin_file(path: str | Path) -> BenchmarkPromptPlugin:
    plugin_path = Path(path)
    if not plugin_path.exists():
        raise FileNotFoundError(f"Prompt plugin file does not exist: {plugin_path}")
    if plugin_path.suffix.lower() in {".txt", ".md"}:
        return BenchmarkPromptPlugin(
            name=plugin_path.stem,
            instructions=(plugin_path.read_text(encoding="utf-8").strip(),),
            source=str(plugin_path),
        )
    payload = json.loads(plugin_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt plugin JSON must be an object: {plugin_path}")
    name = str(payload.get("name") or plugin_path.stem)
    instructions = normalize_instruction_lines(
        payload.get("instructions") or payload.get("prompt") or payload.get("text")
    )
    return BenchmarkPromptPlugin(
        name=name,
        instructions=tuple(instructions),
        source=str(plugin_path),
    )


def normalize_instruction_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines()]
        return [line for line in lines if line]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Unsupported prompt plugin instructions type: {type(value).__name__}")


def prompt_plugin_context(
    plugin: BenchmarkPromptPlugin | None,
    sample: Mapping[str, Any],
) -> dict[str, Any] | None:
    if plugin is None:
        return None
    return {"benchmark_prompt_plugin": plugin.context_for_sample(sample)}


def render_prompt_plugin_section(plugin_payload: Mapping[str, Any] | None) -> str:
    if not plugin_payload:
        return ""
    instructions = [
        str(item).strip()
        for item in plugin_payload.get("instructions", [])
        if str(item).strip()
    ]
    if not instructions:
        return ""

    lines = [f"Benchmark prompt plugin: {plugin_payload.get('name') or 'custom'}"]
    lines.extend(["Instructions:", *[f"- {line}" for line in instructions]])
    return "\n".join(lines)
