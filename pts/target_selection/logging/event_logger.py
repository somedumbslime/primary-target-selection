from __future__ import annotations

import json
from pathlib import Path

from ..domain.models import EventRecord


class JsonlEventLogger:
    def __init__(self, output_path: Path | None = None, enabled: bool = True):
        self.output_path = output_path
        self.enabled = enabled

    def set_output_path(self, path: Path) -> None:
        self.output_path = path

    def log(self, event: EventRecord) -> None:
        if not self.enabled or self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
