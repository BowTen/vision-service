from fastapi import FastAPI
from app.config import settings
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    app.state.services = {}
    if settings.service_mode == "txt2img":
        from app.service.txt2img_service import Txt2ImgService

        service = await Txt2ImgService.build(
            model=settings.hf_home + "/" + settings.txt2img_model
        )
        app.state.services["txt2img"] = service
    elif settings.service_mode == "img2txt":
        from app.service.img2txt_service import Img2TxtService

        service = await Img2TxtService.build(
            model=settings.hf_home + "/" + settings.img2txt_model
        )
        app.state.services["img2txt"] = service

    yield

    # 关闭


app = FastAPI(title="vision-service", version="0.1", lifespan=lifespan)

# 选择服务模式
if settings.service_mode == "txt2img":
    from app.api.routes_txt2img import router as service_router
elif settings.service_mode == "img2txt":
    from app.api.routes_img2txt import router as service_router
else:
    raise ValueError(f"Unsupported service mode: {settings.service_mode}")

app.include_router(service_router)


@app.get("/")
async def root():
    return {"service_mode": settings.service_mode, "status": "running"}
