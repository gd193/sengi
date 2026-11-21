"""Host tree build settings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Static build settings."""

    max_stack_depth: int = 32
    leaf_size_threshold: int = 8
