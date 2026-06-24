from __future__ import annotations

import json
import os
from pathlib import Path

from dashboard.auth import AUTH_ENV_NAMES
from rag.config import SETTINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "frontend" / "shared" / "api" / "openapi.json"


def main() -> None:
    _configure_schema_environment()
    from dashboard.app import create_app

    schema = create_app().openapi()
    SCHEMA_PATH.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _configure_schema_environment() -> None:
    os.environ.setdefault(
        SETTINGS["env"]["database_url"],
        "sqlite:////tmp/medic-openapi-schema.db",
    )
    os.environ.setdefault(AUTH_ENV_NAMES["username"], "schema")
    os.environ.setdefault(AUTH_ENV_NAMES["password"], "schema-password")
    os.environ.setdefault(AUTH_ENV_NAMES["session_secret"], "schema-session-secret")
    os.environ.setdefault(AUTH_ENV_NAMES["cookie_secure"], "false")


if __name__ == "__main__":
    main()
