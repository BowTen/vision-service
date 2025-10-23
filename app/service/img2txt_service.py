from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import ClassVar, Optional
from app.models.infer_queue import InferQueue
import asyncio


class Img2TxtService:
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _instance: ClassVar[Optional["Img2TxtService"]] = None

    def __init__(self, model: str):
        self._model_path = model
        self.queue: InferQueue = InferQueue()
        self.processor: AutoProcessor | None = None
        self.model: AutoModelForVision2Seq | None = None

    @classmethod
    async def build(cls, model: str) -> "Img2TxtService":
        async with cls._lock:
            if cls._instance is None:
                inst = cls(model)
                await inst._initialize()
                cls._instance = inst
            return cls._instance

    async def _initialize(self):
        def load_model():
            processor = AutoProcessor.from_pretrained(self._model_path)
            self.processor = processor
            model = AutoModelForVision2Seq.from_pretrained(self._model_path)
            model = model.to("cuda")
            self.model = model

        await asyncio.to_thread(load_model)

    def _infer_sync(self, inputs: dict, max_new_tokens: int) -> str:
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1] :])

    def _process_inputs(self, image_pth: str, prompt: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_pth},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, non_blocking=True)
        return inputs

    async def queued_generate(
        self, image_pth: str, prompt: str, max_new_tokens: int = 100
    ) -> str:
        assert self.processor is not None and self.model is not None, (
            "Model not initialized yet"
        )
        inputs = await asyncio.to_thread(self._process_inputs, image_pth, prompt)
        return await self.queue.submit(lambda: self._infer_sync(inputs, max_new_tokens))
