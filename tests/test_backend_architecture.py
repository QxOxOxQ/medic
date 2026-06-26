from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USE_CASE_FILES = (
    PROJECT_ROOT / "backend" / "use_cases.py",
    PROJECT_ROOT / "backend" / "chat_use_cases.py",
)
FORBIDDEN_IMPORT_PREFIXES = ("fastapi", "sqlalchemy", "rag.database")


def test_backend_use_cases_do_not_import_presentation_or_database_frameworks() -> None:
    violations: list[str] = []
    for path in USE_CASE_FILES:
        for imported in _imports(path):
            if imported.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {imported}")

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
