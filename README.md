# vision-service

FastAPI 异步图片生成与理解服务（Python 3.10）。  

- 文生图（Text-to-Image）：使用 **diffusers** 部署 **HunyuanDiT 1.5B** 推理
- 图片理解（Image-to-Text）：使用 **transformers** 部署 **Qwen2.5VL-3B** 推理
- 支持 HTTP 与 WebSocket 接口。
- 可通过 Docker 镜像（已发布到 GHCR）部署并挂载本地/共享模型目录。
- 图片理解服务使用**分桶批量推理**

---

## 目录结构（示例）

```
vision-service/
├── app/
│   ├── config.py              # 配置文件
│   ├── main.py                # FastAPI 入口
│   ├── api/
│   │   ├── routes_text2img.py
│   │   └── routes_img2text.py
│   ├── models
│   │   └── infer_queue.py     # 推理队列
│   ├── service
│   │   ├── img2txt_service.py
│   │   └── txt2img_service.py
│   └── static
│       ├── img2txt_test.html  # 演示页面
│       └── txt2img_test.html  # 演示页面
├── tests/
│   ├── test_infer_queue.py
│   ├── test_routes_img2txt.py
│   ├── test_routes_txt2img.py
│   ├── test_smoke.py
│   ├── test_ws_img2txt.py
│   └── test_ws_txt2img.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Docker 部署

### 拉取镜像

```bash
docker pull ghcr.io/bowten/vision-service:latest
```

### 运行容器并挂载模型

```bash
sudo docker run --rm -d --gpus all \
  --name vision-service \
  -p 8000:8000 \
  -e SERVICE_MODE=txt2img \
  -e SERVICE_PORT=8000 \
  -v /path_prefix/models:/data/hf-cache \
  ghcr.io/bowten/vision-service:latest
```

- 服务每次启动时通过SERVICE_MODE环境变量设置为只支持一种服务，可设置为 `txt2img` 或 `img2txt`
- 服务使用cuda推理，需要保证容器内可见cuda 
- 部署后可访问 `host:port/static/txt2img/page` 查看演示页面