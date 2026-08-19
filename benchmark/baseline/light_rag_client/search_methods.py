"""Search strategy definitions and backend mode resolution."""

from enum import Enum


class SearchMethod(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    HYBRID = "hybrid"
    MIX = "mix"
    NAIVE = "naive"
    BASIC = "basic"
    DRIFT = "drift"

    def __str__(self) -> str:
        return self.value


def resolve_search_method(method: str | SearchMethod) -> tuple[str, str]:
    public = method.value if isinstance(method, SearchMethod) else str(method)
    public = public.casefold()
    backend = {"basic": "naive", "drift": "hybrid"}.get(public, public)
    if backend not in {"global", "local", "hybrid", "mix", "naive"}:
        raise ValueError(f"unsupported search method: {method}")
    return backend, public
