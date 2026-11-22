"""Host tree build settings."""
from dataclasses import dataclass
from operator import index


@dataclass(frozen=True, kw_only=True)
class Config:
    """Static build settings."""

    leaf_size_threshold: int = 8

    def __post_init__(self):
        try:
            leaf_size = index(self.leaf_size_threshold)
        except TypeError as exc:
            raise ValueError("leaf size must be a positive integer") from exc
        if leaf_size < 1:
            raise ValueError("leaf size must be a positive integer")
