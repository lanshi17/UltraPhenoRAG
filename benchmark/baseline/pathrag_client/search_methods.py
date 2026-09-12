"""Search strategy definitions and backend mode resolution."""

from enum import Enum


class SearchMethod(str, Enum):
    """PathRAG query modes.

    The vendored ``kg_query`` only implements the ``hybrid`` pipeline: it
    extracts low- and high-level keywords, then mixes entity- and
    relationship-centric retrieval.  The legacy ``local``/``global``
    switching happens inside the context builder, not as a public mode, so
    this enum exposes a single public method.
    """

    HYBRID = "hybrid"

    def __str__(self) -> str:
        return self.value


def resolve_search_method(method: str | SearchMethod) -> tuple[str, str]:
    public = method.value if isinstance(method, SearchMethod) else str(method)
    public = public.casefold()
    if public != "hybrid":
        raise ValueError(f"unsupported search method: {method}")
    return "hybrid", public
