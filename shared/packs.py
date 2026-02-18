from __future__ import annotations

from pathlib import Path
from typing import Iterable

from shared.constants import BUILTIN_PACK_IDS, PACK_DIR, RUN_LIMITS
from shared.types import PackDescriptor

PACK_METADATA = {
    "regex": {
        "title": "Regex Patterns",
        "description": "Common practical regular expression snippets.",
    },
    "abc_music": {
        "title": "ABC Music",
        "description": "Small melodic snippets in ABC notation.",
    },
    "chess_pgn": {
        "title": "Chess PGN",
        "description": "Short opening and tactical move sequences.",
    },
    "sql_snippets": {
        "title": "SQL Snippets",
        "description": "Short query patterns and clauses.",
    },
    "arithmetic": {
        "title": "Arithmetic",
        "description": "Digit-level addition and subtraction templates.",
    },
    "json": {
        "title": "JSON Objects",
        "description": "Small fixed-schema JSON lines with typed fields.",
    },
}


def _read_docs(path: Path) -> list[str]:
    docs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    docs = [doc for doc in docs if doc]
    joined = "\n".join(docs)
    if len(joined) > RUN_LIMITS["corpus_max_chars"]:
        raise ValueError(f"Corpus in {path.name} exceeds max size")
    return docs


def load_builtin_pack_docs(pack_id: str) -> list[str]:
    if pack_id not in BUILTIN_PACK_IDS:
        raise ValueError(f"Unknown pack: {pack_id}")
    path = PACK_DIR / f"{pack_id}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_docs(path)


def build_pack_descriptors() -> list[PackDescriptor]:
    descriptors: list[PackDescriptor] = []
    for pack_id in BUILTIN_PACK_IDS:
        docs = load_builtin_pack_docs(pack_id)
        joined = "\n".join(docs)
        meta = PACK_METADATA[pack_id]
        descriptors.append(
            PackDescriptor(
                pack_id=pack_id,
                title=meta["title"],
                description=meta["description"],
                document_count=len(docs),
                character_count=len(joined),
            )
        )
    return descriptors


def docs_from_text(text: str) -> list[str]:
    docs = [line.strip() for line in text.splitlines() if line.strip()]
    if not docs:
        raise ValueError("Uploaded corpus contains no non-empty documents")
    joined = "\n".join(docs)
    if len(joined) > RUN_LIMITS["corpus_max_chars"]:
        raise ValueError("Uploaded corpus exceeds character limit")
    return docs


def resolve_docs(pack_id: str, upload_text: str | None) -> list[str]:
    if pack_id in BUILTIN_PACK_IDS:
        return load_builtin_pack_docs(pack_id)
    if pack_id.startswith("upload:"):
        if upload_text is None:
            raise ValueError("Upload not found")
        return docs_from_text(upload_text)
    raise ValueError(f"Unknown pack: {pack_id}")
