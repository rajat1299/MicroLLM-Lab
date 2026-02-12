from __future__ import annotations

import pytest

from shared.types import RunConfig


def test_run_config_rejects_head_embedding_mismatch() -> None:
    with pytest.raises(ValueError, match="divisible"):
        RunConfig(n_embd=30, n_head=8)


def test_run_config_rejects_excessive_num_steps() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        RunConfig(num_steps=5000)
