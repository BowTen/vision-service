from fastapi import APIRouter, HTTPException, Depends, WebSocket
from starlette.requests import HTTPConnection
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from app.service.txt2img_service import Txt2ImgService
from io import BytesIO
import asyncio
import traceback

router = APIRouter(prefix="/txt2img", tags=["Text-to-Image"])


class Text2ImgRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty")
        return v


def get_txt2img_service(request: HTTPConnection) -> Txt2ImgService | None:
    return request.app.state.services.get("txt2img")


@router.post("/generate")
async def generate_image(
    request_body: Text2ImgRequest,
    service: Txt2ImgService | None = Depends(get_txt2img_service),
):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    prompt = request_body.prompt

    try:
        image = await service.queued_generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


# 接口规范
# 客户端发送文本帧作为 prompt
# 服务端开始生成图片，过程中每隔1s发送一个文本帧标识完成情况（0-99的数字）
# 服务端发送 100 表示生成完毕，然后立刻发送png图片的二进制数据，然后关闭连接
@router.websocket("/ws/generate")
async def websocket_generate_image(
    ws: WebSocket,
    service: Txt2ImgService | None = Depends(get_txt2img_service),
):
    await ws.accept()
    if service is None:
        await ws.close(code=1011)  # Internal Error: Service not initialized
        return

    prompt = (await ws.receive_text()).strip()
    if not prompt:
        await ws.close(code=1008)  # Policy Violation: Prompt cannot be empty
        return

    try:
        gen_task = asyncio.create_task(service.queued_generate(prompt))
        progress = 0
        while not gen_task.done():
            await ws.send_text(str(progress))
            # TODO: 获取真实进度
            progress = min(progress + 4, 99)
            await asyncio.sleep(1)

        image = await gen_task
        await ws.send_text("100")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        await ws.send_bytes(buffer.read())
        await ws.close()
    except Exception as e:
        error_trace = traceback.format_exc()
        await ws.close(code=1011)  # Internal Error: Generation failed
        print(f"WebSocket generation error: {e}\n{error_trace}")
