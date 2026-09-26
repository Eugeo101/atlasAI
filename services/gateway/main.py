import logging
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")
load_dotenv(override=True)


def get_vllm_base_url() -> str:
    url = os.getenv("VLLM_BASE_URL")
    return url


REQUEST_TIMEOUT = float(os.getenv("GATEWAY_TIMEOUT", "60.0"))

app = FastAPI(title="Real Estate AI Gateway", version="0.1.0")


@app.get("/health")
async def health():
    base_url = get_vllm_base_url()
    if not base_url or not base_url.startswith(("http://", "https://")):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Invalid VLLM_BASE_URL configured: '{base_url}'"},
        )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/health")
        if resp.status_code == 200:
            return {"status": "ok", "vllm": "reachable"}
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "vllm": "unreachable", "vllm_status_code": resp.status_code},
        )
    except httpx.RequestError as e:
        logger.error(f"vLLM health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "vllm": "unreachable", "error": str(e)},
        )


@app.post("/v1/chat")
async def chat(payload: dict):
    base_url = get_vllm_base_url()
    if not base_url or not base_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid VLLM_BASE_URL environment variable: '{base_url}'",
        )

    target_url = f"{base_url}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(target_url, json=payload)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.RequestError as e:
        logger.error(f"vLLM request failed: {e}")
        raise HTTPException(status_code=502, detail=f"vLLM unreachable at {target_url}: {e}")