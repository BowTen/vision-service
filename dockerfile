FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS base

WORKDIR /vision-service

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
ENV HUGGINGFACE_HUB_CACHE=/data/hf-cache \
    HF_HOME=/data/hf-cache

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev     # 生产环境依赖

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests

# 应用运行模式： txt2img 或 img2txt
ENV SERVICE_MODE=txt2img
ENV SERVICE_PORT=8000
EXPOSE 8000

RUN chmod +x scripts/entrypoint.sh
ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["run"]