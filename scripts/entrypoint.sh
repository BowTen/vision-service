#!/usr/bin/env bash
uv run uvicorn app.main:app --host 0.0.0.0 --port $SERVICE_PORT