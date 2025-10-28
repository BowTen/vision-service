from diffusers import DiffusionPipeline
from app.models.infer_queue import InferQueue
import asyncio
from typing import ClassVar, Optional
from PIL.Image import Image
from asyncio import Future
from typing import Any
import torch


class Txt2ImgService:
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _instance: ClassVar[Optional["Txt2ImgService"]] = None

    def __init__(
        self, model: str, batch_size: int, num_inference_steps: int, max_wait_ms: int
    ):
        self._model = model
        self.queue: InferQueue = InferQueue()
        self.pipe: DiffusionPipeline | None = None
        self.batch_size: int = batch_size
        self.num_inference_steps: int = num_inference_steps
        self.max_wait_ms: int = max_wait_ms
        self.batch_prompts: list = []
        self._future_result: Future = Future()
        self._batch_id: int = 0
        self.device_str: str = "cuda:0"

    @classmethod
    async def build(
        cls,
        model: str,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        max_wait_ms: int = 5 * 1000,
    ) -> "Txt2ImgService":
        async with cls._lock:
            if cls._instance is None:
                inst = cls(model, batch_size, num_inference_steps, max_wait_ms)
                await inst._initialize()
                cls._instance = inst
            return cls._instance

    async def _initialize(self):
        def load_pipe():
            pipe = DiffusionPipeline.from_pretrained(
                self._model, torch_dtype=torch.float16
            )
            pipe.to(self.device_str)
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode="max-autotune",
            )
            self.pipe = pipe

        await asyncio.to_thread(load_pipe)

    def _infer_sync(self, batch_prompts: list) -> Any:
        with torch.inference_mode(), torch.amp.autocast(self.device_str, torch.float16):
            return self.pipe(
                batch_prompts,
                num_inference_steps=self.num_inference_steps,
                width=1024,
                height=1024,
            )

    async def flush_batch(self) -> Future:
        batch_prompts = self.batch_prompts
        future_result = self._future_result
        self._batch_id += 1
        self.batch_prompts = []
        self._future_result = Future()

        try:
            result = await self.queue.submit(lambda: self._infer_sync(batch_prompts))
            future_result.set_result(result)
        except Exception as e:
            future_result.set_exception(e)
        finally:
            return future_result

    async def _flush_batch_later(self, batch_id: int) -> None:
        await asyncio.sleep(self.max_wait_ms / 1000.0)
        if batch_id == self._batch_id and len(self.batch_prompts) > 0:
            await self.flush_batch()

    async def queued_generate(self, prompt: str) -> Image:
        result_id = len(self.batch_prompts)
        self.batch_prompts.append(prompt)
        if len(self.batch_prompts) >= self.batch_size:
            future_result = await self.flush_batch()
            return (await future_result).images[result_id]
        else:
            if len(self.batch_prompts) == 1:
                asyncio.create_task(self._flush_batch_later(self._batch_id))
            result = await self._future_result
            return result.images[result_id]
