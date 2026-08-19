"""Index strategy definitions."""

from enum import Enum


class IndexMethod(str, Enum):
    STANDARD = "standard"
    FAST = "fast"
    STANDARD_UPDATE = "standard-update"
    FAST_UPDATE = "fast-update"

    def __str__(self) -> str:
        return self.value
