"""Search strategy definitions and backend pipeline resolution."""

from enum import Enum


class SearchMethod(str, Enum):
    SOLVER = "solver"
    """KAG 逻辑表单静态规划求解管线（planner + 混合检索 + 推理 + 生成）。"""

    NAIVE = "naive"
    """纯向量检索 RAG 管线（chunk 检索 + 生成），不含逻辑规划与图推理。"""

    BASIC = "basic"
    """naive 的别名，保持与其它基线客户端的 API 兼容。"""

    def __str__(self) -> str:
        return self.value


def resolve_search_method(method: str | SearchMethod) -> tuple[str, str]:
    """Map a public search method onto a KAG solver pipeline key."""
    public = method.value if isinstance(method, SearchMethod) else str(method)
    public = public.casefold()
    backend = {"basic": "naive"}.get(public, public)
    if backend not in {"solver", "naive"}:
        raise ValueError(f"unsupported search method: {method}")
    return backend, public
