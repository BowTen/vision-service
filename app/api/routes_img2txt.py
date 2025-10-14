from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, Request, Form, File, UploadFile
from app.service.img2txt_service import Img2TxtService
import tempfile
import mimetypes

router = APIRouter(prefix="/img2txt", tags=["Image-to-Text"])


class Img2TxtResponse(BaseModel):
    text: str


def get_img2txt_service(request: Request) -> Img2TxtService:
    service = request.app.state.services.get("img2txt")
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    return service


@router.post("/generate", response_model=Img2TxtResponse)
async def generate_text(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    service: Img2TxtService = Depends(get_img2txt_service),
):
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
