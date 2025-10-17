from starlette.websockets import WebSocketDisconnect

# 接口规范
# 客户端发送文本帧作为 prompt
# 服务端开始生成图片，过程中每隔1s发送一个文本帧标识完成情况（0-99的数字）
# 服务端发送 100 表示生成完毕，然后立刻发送png图片的二进制数据，然后关闭连接


# 成功流程：prompt -> progress -> 100 -> image bytes
def test_ws_txt2img_success(client_txt2img):
    with client_txt2img.websocket_connect("/txt2img/ws/generate") as ws:
        ws.send_text("a cat")

        # 验证进度帧不减少于0且不大于100
        last_progress = 0
        while True:
            progress_str = ws.receive_text()
            progress = int(progress_str)
            assert progress >= last_progress
            assert progress <= 100
            last_progress = progress
            if progress_str == "100":
                break

        # 接收png格式二进制数据
        image_bytes = ws.receive_bytes()
        assert len(image_bytes) > 0
        from PIL import Image
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes))
        assert img.format == "PNG"


# 空 prompt
def test_ws_txt2img_empty_prompt(client_txt2img):
    with client_txt2img.websocket_connect("/txt2img/ws/generate") as ws:
        ws.send_text("          ")

        try:
            ws.receive_text()
        except WebSocketDisconnect as e:
            assert e.code == 1008  # Policy Violation: Prompt cannot be empty


# 内部异常
def test_ws_txt2img_internal_error(app_txt2img):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeTxt2ImgServiceError

    app_txt2img.state.services["img2txt"] = FakeTxt2ImgServiceError()
    client = TestClient(app_txt2img)

    with client.websocket_connect("/txt2img/ws/generate") as ws:
        ws.send_text("a cat")

        try:
            ws.receive_text()
        except WebSocketDisconnect as e:
            assert e.code == 1011  # Internal Error: Generation failed
