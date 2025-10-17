from starlette.websockets import WebSocketDisconnect

# 接口规范
# 客户端发送图片二进制数据帧
# 客户端发送文本帧作为 prompt
# 服务端开始生成文本，过程中可能发送任意数量文本帧表示生成内容
# 服务端生成完毕直接正常关闭连接


# 成功流程：image_bytes + prompt -> result text
def test_ws_img2txt_success(client_img2txt, sample_png_bytes):
    with client_img2txt.websocket_connect("/img2txt/ws/generate") as ws:
        ws.send_bytes(sample_png_bytes)
        ws.send_text("Describe the image")

        received_texts = ""
        while True:
            try:
                received_texts += ws.receive_text()
            except WebSocketDisconnect as e:
                assert e.code == 1000  # 正常关闭
                break

        assert len(received_texts) > 0
        assert received_texts == "TEXT(Describe the image)"


# 空 prompt
def test_ws_img2txt_empty_prompt(client_img2txt, sample_png_bytes):
    with client_img2txt.websocket_connect("/img2txt/ws/generate") as ws:
        ws.send_bytes(sample_png_bytes)
        ws.send_text("          ")

        try:
            ws.receive_text()
        except WebSocketDisconnect as e:
            assert e.code == 1008  # Policy Violation: Prompt cannot be empty


# 内部异常
def test_ws_img2txt_internal_error(app_img2txt, sample_png_bytes):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeImg2TxtServiceError

    app_img2txt.state.services["img2txt"] = FakeImg2TxtServiceError()
    client = TestClient(app_img2txt)

    with client.websocket_connect("/img2txt/ws/generate") as ws:
        ws.send_bytes(sample_png_bytes)
        ws.send_text("This will cause an error")

        try:
            ws.receive_text()
        except WebSocketDisconnect as e:
            assert e.code == 1011  # Internal Error: Generation failed
