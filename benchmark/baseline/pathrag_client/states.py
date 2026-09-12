"""Runtime states for the PathRAG client."""

from enum import Enum


class ClientState(str, Enum):
    CREATED = "created"
    READY = "ready"
    CLOSED = "closed"
