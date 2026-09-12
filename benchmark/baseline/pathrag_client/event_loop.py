"""Synchronous execution of asynchronous PathRAG operations."""

import asyncio
import threading
from typing import Any


class EventLoopRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def run(self, coroutine: Any) -> Any:
        if threading.get_ident() == self._thread.ident:
            raise RuntimeError("synchronous operation called from client event loop")
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def close(self) -> None:
        async def cancel_tasks() -> None:
            current = asyncio.current_task()
            pending = [
                task for task in asyncio.all_tasks(self.loop) if task is not current
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(cancel_tasks(), self.loop).result(
                timeout=10
            )
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        self.loop.close()
