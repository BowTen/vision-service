from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import ClassVar, Optional, Any
from app.models.infer_queue import InferQueue
import asyncio
from asyncio import Future


class Img2TxtService:
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _instance: ClassVar[Optional["Img2TxtService"]] = None

    def __init__(
        self, model: str, batch_size: int, max_new_tokens: int, max_wait_ms: int
    ):
        self._model_path = model
        self.queue: InferQueue = InferQueue()
        self.processor: AutoProcessor | None = None
        self.model: AutoModelForVision2Seq | None = None
        self._batch_size: int = batch_size
        self._batch_messages: list = []
        self._future_result: Future = asyncio.Future()
        self._max_new_tokens: int = max_new_tokens
        self._max_wait_ms: int = max_wait_ms
        self._batch_id: int = 0

    @classmethod
    async def build(
        cls,
        model: str,
        batch_size: int = 1,
        max_new_tokens: int = 100,
        max_wait_ms: int = 5 * 1000,
    ) -> "Img2TxtService":
        async with cls._lock:
            if cls._instance is None:
                inst = cls(model, batch_size, max_new_tokens, max_wait_ms)
                await inst._initialize()
                cls._instance = inst
            return cls._instance

    async def _initialize(
        self,
    ):
        def load_model():
            processor = AutoProcessor.from_pretrained(self._model_path)
            self.processor = processor
            model = AutoModelForVision2Seq.from_pretrained(self._model_path)
            model = model.to("cuda")
            self.model = model

        await asyncio.to_thread(load_model)

    def _infer_sync(self, inputs: dict) -> Any:
        return self.model.generate(**inputs, max_new_tokens=self._max_new_tokens)

    async def _process_inputs(self, messages: list) -> dict:
        def thread_process_inputs():
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(self.model.device, non_blocking=True)
            return inputs

        return await asyncio.to_thread(thread_process_inputs)

    async def _flush_batch(self) -> Future:
        assert len(self._batch_messages) > 0
        batch_messages = self._batch_messages
        future_result = self._future_result
        self._batch_id += 1
        self._batch_messages = []
        self._future_result = asyncio.Future()
        try:
            inputs = await self._process_inputs(batch_messages)
            outputs = await self.queue.submit(lambda: self._infer_sync(inputs))
            future_result.set_result((inputs, outputs))
        except Exception as e:
            future_result.set_exception(e)
        finally:
            return future_result

    async def _flush_batch_later(self, batch_id: int):
        await asyncio.sleep(self._max_wait_ms / 1000.0)
        if batch_id == self._batch_id and len(self._batch_messages) > 0:
            await self._flush_batch()

    async def queued_generate(self, image_pth: str, prompt: str) -> str:
        assert self.processor is not None and self.model is not None, (
            "Model not initialized yet"
        )
        result_id = len(self._batch_messages)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_pth},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        self._batch_messages.append(messages)
        if len(self._batch_messages) >= self._batch_size:
            future_result = await self._flush_batch()
            (inputs, outputs) = await future_result
            return self.processor.decode(
                outputs[result_id][inputs["input_ids"].shape[-1] :]
            )
        else:
            if len(self._batch_messages) == 1:
                asyncio.create_task(self._flush_batch_later(self._batch_id))
            (inputs, outputs) = await self._future_result
            return self.processor.decode(
                outputs[result_id][inputs["input_ids"].shape[-1] :]
            )
