from pathlib import Path

import pytest

import rag.config as settings_module
from rag.qdrant import Qdrant


ENV_NAMES = settings_module.SETTINGS["env"]


def test_qdrant_client_uses_qdrant_env_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv(ENV_NAMES["qdrant_url"], raising=False)
    monkeypatch.delenv(ENV_NAMES["qdrant_api_key"], raising=False)
    (tmp_path / ".env").write_text(
        f"{ENV_NAMES['qdrant_url']}=https://qdrant.example:7443\n"
        f"{ENV_NAMES['qdrant_api_key']}=qdrant-key\n"
    )

    qdrant = Qdrant()
    client = qdrant.client

    assert client._client.openapi_client.client.host == "https://qdrant.example:7443"
    assert client._client.openapi_client.client._client.headers["api-key"] == "qdrant-key"
