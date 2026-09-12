"""Runtime states for the KAG client."""

from enum import Enum


class ClientState(str, Enum):
    CREATED = "created"
    READY = "ready"
    CLOSED = "closed"
