from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingRecorder:
    components: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def record(self, component: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(component, time.perf_counter() - start)

    def add(self, component: str, seconds: float, *, calls: int = 1) -> None:
        self.components[component] += float(seconds)
        self.calls[component] += int(calls)

    def snapshot(self) -> dict[str, Any]:
        components: dict[str, dict[str, float | int]] = {}
        for component, seconds in sorted(self.components.items()):
            call_count = self.calls.get(component, 0)
            components[component] = {
                "seconds": round(seconds, 6),
                "calls": call_count,
                "avg_seconds": round(seconds / call_count, 6) if call_count else 0.0,
            }
        return {
            "components": components,
            "total_recorded_seconds": round(sum(self.components.values()), 6),
        }


def merge_timing_summaries(*summaries: dict[str, Any] | None) -> dict[str, Any]:
    recorder = TimingRecorder()
    for summary in summaries:
        if not summary:
            continue
        components = summary.get("components", {})
        if not isinstance(components, dict):
            continue
        for component, payload in components.items():
            if not isinstance(payload, dict):
                continue
            recorder.add(
                str(component),
                float(payload.get("seconds", 0.0) or 0.0),
                calls=int(payload.get("calls", 0) or 0),
            )
    return recorder.snapshot()
