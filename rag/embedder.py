from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embedding.embedder import Embedder, embed_texts, main

__all__ = ["Embedder", "embed_texts", "main"]


if __name__ == "__main__":
    main()
