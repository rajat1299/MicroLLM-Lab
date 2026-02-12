from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from shared.constants import RUN_LIMITS


class RunConfig(BaseModel):
    n_embd: int = Field(default=32, ge=8)
    n_head: int = Field(default=4, ge=1)
    n_layer: int = Field(default=1, ge=1)
    block_size: int = Field(default=16, ge=4)
    num_steps: int = Field(default=300, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    temperature: float = Field(default=0.8, gt=0)
    seed: int = Field(default=42)
    sample_count: int = Field(default=5, ge=1, le=20)
    sample_interval: int = Field(default=100, ge=1)
    top_k: int = Field(default=5, ge=1, le=20)
    op_graph_token_index: int = Field(default=0, ge=0)
    op_graph_step_interval: int = Field(default=25, ge=1)

    @model_validator(mode="after")
    def validate_limits(self) -> "RunConfig":
        if self.n_embd > RUN_LIMITS["n_embd_max"]:
            raise ValueError(f"n_embd exceeds {RUN_LIMITS['n_embd_max']}")
        if self.n_head > RUN_LIMITS["n_head_max"]:
            raise ValueError(f"n_head exceeds {RUN_LIMITS['n_head_max']}")
        if self.n_layer > RUN_LIMITS["n_layer_max"]:
            raise ValueError(f"n_layer exceeds {RUN_LIMITS['n_layer_max']}")
        if self.block_size > RUN_LIMITS["block_size_max"]:
            raise ValueError(f"block_size exceeds {RUN_LIMITS['block_size_max']}")
        if self.num_steps > RUN_LIMITS["num_steps_max"]:
            raise ValueError(f"num_steps exceeds {RUN_LIMITS['num_steps_max']}")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        return self


class RunCreateRequest(BaseModel):
    pack_id: str
    config: RunConfig = Field(default_factory=RunConfig)


class RunSummary(BaseModel):
    run_id: str
    status: str
    pack_id: str
    config: RunConfig
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class PackDescriptor(BaseModel):
    pack_id: str
    title: str
    description: str
    document_count: int
    character_count: int


class RunEvent(BaseModel):
    seq: int
    type: str
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    upload_id: str
    document_count: int
    character_count: int
    expires_at: datetime
