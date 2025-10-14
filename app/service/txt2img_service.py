from diffusers import DiffusionPipeline
from app.models.infer_queue import InferQueue
import asyncio
from typing import ClassVar, Optional
from PIL.Image import Image


class Txt2ImgService:
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _instance: ClassVar[Optional["Txt2ImgService"]] = None

    def __init__(self, model: str):
        self._model = model
        self.queue: InferQueue = InferQueue()
        self.pipe: DiffusionPipeline | None = None

    @classmethod
    async def build(cls, model: str) -> "Txt2ImgService":
        async with cls._lock:
            if cls._instance is None:
                inst = cls(model)
                await inst._initialize()
                cls._instance = inst
            return cls._instance

    async def _initialize(self):
        def load_pipe():
            pipe = DiffusionPipeline.from_pretrained(self._model)
            pipe.to("cuda")
            self.pipe = pipe

        await asyncio.to_thread(load_pipe)

    def _infer_sync(self, prompt: str) -> Image:
        assert self.pipe is not None, "Pipeline not initialized yet"
        return self.pipe(prompt).images[0]

    async def queued_generate(self, prompt: str):
        return await self.queue.submit(lambda: self._infer_sync(prompt))
