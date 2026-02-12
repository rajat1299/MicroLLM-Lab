from __future__ import annotations

import pytest

from shared.validation import UploadValidationError, validate_upload


def test_validate_upload_accepts_valid_text() -> None:
    text = b"line1\nline2\n"
    result = validate_upload("corpus.txt", text)
    assert "line1" in result


def test_validate_upload_rejects_wrong_extension() -> None:
    with pytest.raises(UploadValidationError, match="allowed"):
        validate_upload("corpus.csv", b"a,b")


def test_validate_upload_rejects_blocked_content() -> None:
    with pytest.raises(UploadValidationError, match="Blocked"):
        validate_upload("corpus.txt", b"DROP DATABASE users;")
