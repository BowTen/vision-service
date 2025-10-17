import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import asyncio


# -------- Fake Services --------
class FakeTxt2ImgService:
    async def queued_generate(self, prompt: str):
        await asyncio.sleep(2)
        # 返回一个 2x2 的简单 PNG 图像
        img = Image.new("RGB", (2, 2), color=(255, 0, 0))
        return img


class FakeTxt2ImgServiceError(FakeTxt2ImgService):
    async def queued_generate(self, prompt: str):
        await asyncio.sleep(2)
        raise RuntimeError("simulate generation failure")


class FakeImg2TxtService:
    async def queued_generate(self, image_path: str, prompt: str):
        await asyncio.sleep(2)
        return f"TEXT({prompt})"


class FakeImg2TxtServiceError(FakeImg2TxtService):
    async def queued_generate(self, image_path: str, prompt: str):
        await asyncio.sleep(2)
        raise RuntimeError("simulate img2txt failure")


# -------- Fixtures: Routers Only (Not full app.main) --------
@pytest.fixture
def app_txt2img():
    # 只测试 txt2img 路由层
    from app.api.routes_txt2img import router as txt2img_router

    app = FastAPI()
    app.state.services = {}
    app.include_router(txt2img_router)
    return app


@pytest.fixture
def app_img2txt():
    from app.api.routes_img2txt import router as img2txt_router

    app = FastAPI()
    app.state.services = {}
    app.include_router(img2txt_router)
    return app


@pytest.fixture
def client_txt2img(app_txt2img):
    # 默认植入可用的 Fake 服务
    app_txt2img.state.services["txt2img"] = FakeTxt2ImgService()
    return TestClient(app_txt2img)


@pytest.fixture
def client_img2txt(app_img2txt):
    app_img2txt.state.services["img2txt"] = FakeImg2TxtService()
    return TestClient(app_img2txt)


# -------- Helper: 构造内存 PNG 文件 --------
@pytest.fixture
def sample_png_bytes():
    img = Image.new("RGB", (3, 3), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
