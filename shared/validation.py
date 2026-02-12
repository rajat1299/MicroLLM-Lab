from __future__ import annotations

from pathlib import Path

from shared.constants import CONTENT_BLOCKLIST, RUN_LIMITS, UPLOAD_ALLOWED_EXTENSIONS


class UploadValidationError(ValueError):
    pass


def validate_upload(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in UPLOAD_ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(UPLOAD_ALLOWED_EXTENSIONS))
        raise UploadValidationError(f"Only {allowed} files are allowed")

    if len(content) > RUN_LIMITS["upload_max_bytes"]:
        raise UploadValidationError(
            f"File exceeds {RUN_LIMITS['upload_max_bytes']} bytes"
        )

    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadValidationError("File must be UTF-8 text") from exc

    text = decoded.strip()
    if not text:
        raise UploadValidationError("File is empty")

    unique_chars = len(set(text))
    if unique_chars > RUN_LIMITS["upload_max_unique_chars"]:
        raise UploadValidationError(
            f"Too many unique characters: {unique_chars} > {RUN_LIMITS['upload_max_unique_chars']}"
        )

    upper_text = text.upper()
    for blocked in CONTENT_BLOCKLIST:
        if blocked.upper() in upper_text:
            raise UploadValidationError(f"Blocked content detected: {blocked}")

    if len(text) > RUN_LIMITS["corpus_max_chars"]:
        raise UploadValidationError(
            f"Corpus exceeds {RUN_LIMITS['corpus_max_chars']} characters"
        )

    return text
