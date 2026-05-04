from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    "fastapi",
    "langfuse",
    "sqlalchemy",
    "qdrant_client",
    "ragas",
    "openai",
)


def test_domain_and_application_do_not_import_infrastructure_frameworks() -> None:
    violations: list[str] = []
    for layer in ("domain", "application"):
        for path in (PROJECT_ROOT / "evaluation" / layer).glob("*.py"):
            for imported in _imports(path):
                if imported.startswith(FORBIDDEN_PREFIXES):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} imports {imported}"
                    )

    assert violations == []


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return tuple(names)
