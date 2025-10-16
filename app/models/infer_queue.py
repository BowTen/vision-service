import asyncio
from typing import Callable, TypeVar

T = TypeVar("T")


class InferQueue:
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._is_running = False

    async def submit(self, infer_sync: Callable[[], T]) -> T:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((infer_sync, future))
        if not self._is_running:
            self._is_running = True
            asyncio.create_task(self._run_loop())
        return await future

    async def _run_loop(self):
        try:
            while True:
                infer_sync, future = await self._queue.get()
                try:
                    result = await asyncio.to_thread(infer_sync)
                except Exception as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)
                if self._queue.empty():
                    await asyncio.sleep(0)
                    if self._queue.empty():
                        break
        finally:
            self._is_running = False
