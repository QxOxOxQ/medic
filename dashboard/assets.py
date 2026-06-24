from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrontendAssets:
    script: str
    styles: tuple[str, ...]


class AssetManifest:
    def __init__(
        self,
        *,
        manifest_path: Path,
        static_prefix: str = "/static/dist",
    ) -> None:
        self._manifest_path = manifest_path
        self._static_prefix = static_prefix.rstrip("/")

    def frontend(self) -> FrontendAssets:
        payload = self._read()
        entry = payload.get("frontend/main.tsx")
        if not isinstance(entry, dict):
            return FrontendAssets(
                script=f"{self._static_prefix}/main.js",
                styles=(),
            )
        script = str(entry.get("file") or "main.js")
        css = entry.get("css")
        styles = tuple(str(item) for item in css) if isinstance(css, list) else ()
        return FrontendAssets(
            script=f"{self._static_prefix}/{script}",
            styles=tuple(f"{self._static_prefix}/{item}" for item in styles),
        )

    def _read(self) -> dict[str, Any]:
        if not self._manifest_path.is_file():
            return {}
        try:
            value = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
