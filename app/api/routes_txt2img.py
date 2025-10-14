from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from app.service.txt2img_service import Txt2ImgService
from io import BytesIO

router = APIRouter(prefix="/txt2img", tags=["Text-to-Image"])


class Text2ImgRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty")
        return v


def get_txt2img_service(request: Request) -> Txt2ImgService:
    service = request.app.state.services.get("txt2img")
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    return service


@router.post("/generate")
async def generate_image(
    request_body: Text2ImgRequest,
    service: Txt2ImgService = Depends(get_txt2img_service),
):
    prompt = request_body.prompt

    try:
        image = await service.queued_generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
