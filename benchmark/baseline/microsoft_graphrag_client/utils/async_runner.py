# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""异步工具类。

将 GraphRAG 的异步 API 转为同步调用，兼容已有事件循环的场景（如 Jupyter）。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any


class AsyncRunner:
    """异步协程同步执行器。

    自动检测当前是否已有运行中的事件循环：
    - 无事件循环时直接使用 ``asyncio.run``。
    - 已在事件循环中时（如 Jupyter Notebook），通过线程池执行以避免嵌套循环错误。
    """

    @staticmethod
    def run(coro: Any) -> Any:
        """同步运行一个协程。

        Parameters
        ----------
        coro : Coroutine
            待执行的协程对象。

        Returns
        -------
        Any
            协程的返回值。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
