from fastapi import FastAPI
import pytest_asyncio
import pytest
from app.config import settings
from app.api.routes_img2txt import router as img2txt_router
from app.service.img2txt_service import Img2TxtService
from fastapi.testclient import TestClient
from torch.profiler import profile, record_function, ProfilerActivity


@pytest_asyncio.fixture
async def app_img2txt():
    app = FastAPI()
    app.state.services = {}
    service = await Img2TxtService.build(
        model=settings.hf_home + "/" + settings.img2txt_model
    )
    app.state.services["img2txt"] = service
    app.include_router(img2txt_router)
    return app


@pytest.fixture
def sample_png_bytes():
    image_path = "./astronaut_rides_horse.png"
    with open(image_path, "rb") as f:
        sample_png_bytes = f.read()
    return sample_png_bytes


def test_profile_generate(app_img2txt, sample_png_bytes):
    client = TestClient(app_img2txt)
    files = {
        "image": ("test.png", sample_png_bytes, "image/png"),
    }
    data = {
        "prompt": "描述这张图片",
    }

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        with record_function("post_img2txt_generate"):
            resp = client.post("/img2txt/generate", data=data, files=files)
    prof.export_chrome_trace("img2txt_trace.json")

    assert resp.status_code == 200
    body = resp.json()
    print(body)
