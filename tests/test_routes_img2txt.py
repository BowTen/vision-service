import pytest


# 成功生成文本
def test_img2txt_generate_success(client_img2txt, sample_png_bytes):
    files = {
        "image": ("test.png", sample_png_bytes, "image/png"),
    }
    data = {
        "prompt": "describe this image",
    }
    resp = client_img2txt.post("/img2txt/generate", data=data, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body
    assert body["text"].startswith("TEXT(describe this image)")


# 空白 prompt -> 400
def test_img2txt_generate_empty_prompt(client_img2txt, sample_png_bytes):
    files = {"image": ("a.png", sample_png_bytes, "image/png")}
    data = {"prompt": "   "}
    resp = client_img2txt.post("/img2txt/generate", data=data, files=files)
    assert resp.status_code == 400
    assert "Prompt cannot be empty" in resp.json()["detail"]


# 非图片文件 -> 400
def test_img2txt_generate_non_image_file(client_img2txt):
    fake_bytes = b"not an image"
    files = {
        "image": ("file.txt", fake_bytes, "text/plain"),
    }
    data = {"prompt": "hello"}
    resp = client_img2txt.post("/img2txt/generate", data=data, files=files)
    assert resp.status_code == 400
    assert "Uploaded file is not an image" in resp.json()["detail"]


# 服务未初始化 -> 503
def test_img2txt_service_not_initialized(app_img2txt, sample_png_bytes):
    from fastapi.testclient import TestClient

    client = TestClient(app_img2txt)  # 未放入服务
    files = {"image": ("t.png", sample_png_bytes, "image/png")}
    data = {"prompt": "desc"}
    resp = client.post("/img2txt/generate", data=data, files=files)
    assert resp.status_code == 503
    assert "Service not initialized" in resp.json()["detail"]


# 内部异常 -> 500
def test_img2txt_internal_exception(app_img2txt, sample_png_bytes):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeImg2TxtServiceError

    app_img2txt.state.services["img2txt"] = FakeImg2TxtServiceError()
    client = TestClient(app_img2txt)
    files = {"image": ("x.png", sample_png_bytes, "image/png")}
    data = {"prompt": "boom"}
    resp = client.post("/img2txt/generate", data=data, files=files)
    assert resp.status_code == 500
    assert "Generation failed" in resp.json()["detail"]


# 多次顺序请求
@pytest.mark.parametrize("prompt", ["a", "b", "c"])
def test_img2txt_multiple_sequential_requests(client_img2txt, sample_png_bytes, prompt):
    files = {"image": ("z.png", sample_png_bytes, "image/png")}
    data = {"prompt": prompt}
    resp = client_img2txt.post("/img2txt/generate", data=data, files=files)
    assert resp.status_code == 200
    assert resp.json()["text"].startswith(f"TEXT({prompt})")
