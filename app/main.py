from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.config import settings
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    app.state.services = {}
    app.state.html = {}
    if settings.service_mode == "txt2img":
        from app.service.txt2img_service import Txt2ImgService

        service = await Txt2ImgService.build(
            model=settings.hf_home + "/" + settings.txt2img_model,
            batch_size=settings.txt2img_batch_size,
            num_inference_steps=settings.txt2img_infer_steps,
            max_wait_ms=settings.txt2img_max_wait_ms,
        )
        app.state.services["txt2img"] = service
    elif settings.service_mode == "img2txt":
        from app.service.img2txt_service import Img2TxtService

        service = await Img2TxtService.build(
            model=settings.hf_home + "/" + settings.img2txt_model,
            max_new_tokens=100,
            max_wait_ms=settings.img2txt_max_wait_ms,
        )
        app.state.services["img2txt"] = service

    txt2img_html = Path("app/static/txt2img_test.html").read_text(encoding="utf-8")
    img2txt_html = Path("app/static/img2txt_test.html").read_text(encoding="utf-8")

    app.state.html["txt2img"] = txt2img_html
    app.state.html["img2txt"] = img2txt_html

    yield

    # 关闭


app = FastAPI(title="vision-service", version="0.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 选择服务模式
if settings.service_mode == "txt2img":
    from app.api.routes_txt2img import router as service_router

    @app.get("/static/txt2img/page", include_in_schema=False)
    async def txt2img_page():
        return HTMLResponse(app.state.html["txt2img"])
elif settings.service_mode == "img2txt":
    from app.api.routes_img2txt import router as service_router

    @app.get("/static/img2txt/page", include_in_schema=False)
    async def img2txt_page():
        return HTMLResponse(app.state.html["img2txt"])
else:
    raise ValueError(f"Unsupported service mode: {settings.service_mode}")

app.include_router(service_router)


@app.get("/")
async def root():
    return {"service_mode": settings.service_mode, "status": "running"}
