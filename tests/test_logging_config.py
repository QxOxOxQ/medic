from __future__ import annotations

import logging
from pathlib import Path

from observability.logging_config import configure_logging


def test_configure_logging_writes_detailed_errors_to_file(tmp_path: Path) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    log_file = tmp_path / "logs" / "medic.log"

    try:
        configured = configure_logging(
            level=logging.INFO,
            log_file=log_file,
            use_color=False,
        )
        assert configured == log_file

        try:
            raise RuntimeError("provider token secret-123 failed")
        except RuntimeError:
            logging.getLogger("medic.test").exception("Detailed agent failure")

        for handler in logging.getLogger().handlers:
            handler.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "Detailed agent failure" in content
        assert "RuntimeError: provider token secret-123 failed" in content
        assert "Traceback" in content
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
