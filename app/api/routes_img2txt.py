from pydantic import BaseModel
from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    Form,
    File,
    UploadFile,
    WebSocket,
)
from starlette.requests import HTTPConnection
from app.service.img2txt_service import Img2TxtService
import tempfile
import mimetypes
import traceback

router = APIRouter(prefix="/img2txt", tags=["Image-to-Text"])


class Img2TxtResponse(BaseModel):
    text: str


def get_img2txt_service(request: HTTPConnection) -> Img2TxtService | None:
    return request.app.state.services.get("img2txt")


@router.post("/generate", response_model=Img2TxtResponse)
async def generate_text(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    service: Img2TxtService | None = Depends(get_img2txt_service),
):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if not (image.content_type and image.content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    suffix = mimetypes.guess_extension(image.content_type) or ".png"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp_file:
            tmp_file.write(await image.read())
            tmp_file.flush()
            text = await service.queued_generate(tmp_file.name, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return Img2TxtResponse(text=text)


# 接口规范
# 客户端发送图片二进制数据帧
# 客户端发送文本帧作为 prompt
# 服务端开始生成文本，过程中可能发送任意数量文本帧表示生成内容
# 服务端生成完毕直接正常关闭连接
@router.websocket("/ws/generate")
async def websocket_generate_text(
    ws: WebSocket, service: Img2TxtService | None = Depends(get_img2txt_service)
):
    await ws.accept()

    if service is None:
        await ws.close(code=1011)  # Internal Error: Service not initialized
        return

    image_bytes = await ws.receive_bytes()
    prompt = (await ws.receive_text()).strip()
    if not prompt:
        await ws.close(code=1008)  # Policy Violation: Prompt cannot be empty
        return
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_file.flush()

            # TODO: 实现流式输出
            gen_txt = await service.queued_generate(tmp_file.name, prompt)

            await ws.send_text(gen_txt)
        await ws.close()
    except Exception as e:
        error_trace = traceback.format_exc()
        await ws.close(code=1011)  # Internal Error: Generation failed
        print(f"WebSocket generation error: {e}\n{error_trace}")
