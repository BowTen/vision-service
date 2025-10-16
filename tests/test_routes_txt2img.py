import pytest


# 成功生成图片
def test_txt2img_generate_success(client_txt2img):
    resp = client_txt2img.post("/txt2img/generate", json={"prompt": "a cat"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    data = resp.content
    # PNG 文件头应为 89 50 4E 47
    assert data.startswith(b"\x89PNG"), "返回内容不是有效 PNG"


# 空白 prompt -> 422 (Pydantic 验证错误)
def test_txt2img_generate_empty_prompt_validation_error(client_txt2img):
    resp = client_txt2img.post("/txt2img/generate", json={"prompt": "   "})
    assert resp.status_code == 422
    body = resp.json()
    # 检查错误字段
    assert body["detail"][0]["type"].startswith("value_error")


# 服务未初始化 -> 503
def test_txt2img_service_not_initialized(app_txt2img):
    from fastapi.testclient import TestClient

    client = TestClient(app_txt2img)
    # 不植入服务
    resp = client.post("/txt2img/generate", json={"prompt": "hello"})
    assert resp.status_code == 503
    assert "Service not initialized" in resp.json()["detail"]


# 内部异常 -> 500
def test_txt2img_internal_exception(app_txt2img):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeTxt2ImgServiceError

    app_txt2img.state.services["txt2img"] = FakeTxt2ImgServiceError()
    client = TestClient(app_txt2img)
    resp = client.post("/txt2img/generate", json={"prompt": "boom"})
    assert resp.status_code == 500
    assert "Generation failed" in resp.json()["detail"]


# 多次请求顺序性（不做并发，仅路由层烟囱测试）
@pytest.mark.parametrize("prompt", ["cat", "dog", "tree"])
def test_txt2img_multiple_sequential_requests(client_txt2img, prompt):
    resp = client_txt2img.post("/txt2img/generate", json={"prompt": prompt})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
