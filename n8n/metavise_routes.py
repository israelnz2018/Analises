import os
import json
import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Optional, List

router = APIRouter(prefix="/metavise")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01"
}

async def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(ANTHROPIC_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

class RawClaudeRequest(BaseModel):
    system: str
    user: str
    max_tokens: Optional[int] = 2000

@router.post("/claude")
async def metavise_claude(req: RawClaudeRequest):
    try:
        result = await call_claude(
            system_prompt=req.system,
            user_prompt=req.user,
            max_tokens=req.max_tokens or 2000
        )
        return {"success": True, "text": result}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"Claude API error {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/health")
async def metavise_health():
    return {
        "status": "ok",
        "service": "Metavise Claude Proxy",
        "has_api_key": bool(ANTHROPIC_API_KEY),
        "model": CLAUDE_MODEL
    }

import subprocess
import tempfile
import os
import uuid
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel as _BaseModel

# Diretório temporário pra arquivos de vídeo (ephemeral no Railway, OK)
CROP_TMP_DIR = "/tmp/metavise_crops"
os.makedirs(CROP_TMP_DIR, exist_ok=True)


class CropRequest(_BaseModel):
    videoUrl: str
    aspectRatio: str = "1:1"
    cropOffset: int = 0


def _get_dimensions(aspect_ratio: str):
    if aspect_ratio == "1:1":
        return 1080, 1080
    if aspect_ratio == "16:9":
        return 1920, 1080
    if aspect_ratio == "9:16":
        return 1080, 1920
    if aspect_ratio == "4:5":
        return 1080, 1350
    return 1080, 1080


def _build_ffmpeg_filter(aspect_ratio: str, crop_offset: int):
    w, h = _get_dimensions(aspect_ratio)
    x_expr = f"((in_w-out_w)/2)+((in_w-out_w)*({crop_offset}/100))"
    y_expr = f"((in_h-out_h)/2)+((in_h-out_h)*({crop_offset}/100))"
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h}:{x_expr}:{y_expr},setsar=1"
    )


@router.post("/crop")
async def metavise_crop(req: CropRequest):
    """Baixa vídeo, corta com FFmpeg, retorna o arquivo MP4 cortado."""
    job_id = uuid.uuid4().hex[:12]
    input_path = os.path.join(CROP_TMP_DIR, f"{job_id}_input.mp4")
    output_path = os.path.join(CROP_TMP_DIR, f"{job_id}_output.mp4")

    try:
        print(f"[Crop {job_id}] Downloading: {req.videoUrl[:80]}...")
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            r = await client.get(req.videoUrl)
            r.raise_for_status()
            with open(input_path, "wb") as f:
                f.write(r.content)
        size_mb = os.path.getsize(input_path) / 1024 / 1024
        print(f"[Crop {job_id}] Downloaded {size_mb:.1f}MB")

        ffmpeg_filter = _build_ffmpeg_filter(req.aspectRatio, req.cropOffset)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", ffmpeg_filter,
            "-c:v", "libx264",
            "-preset", "superfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        print(f"[Crop {job_id}] FFmpeg filter: {ffmpeg_filter}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            print(f"[Crop {job_id}] FFmpeg ERROR: {proc.stderr[-500:]}")
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg failed: {proc.stderr[-300:]}",
            )
        out_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"[Crop {job_id}] Done, output {out_size_mb:.1f}MB")

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"cropped_{job_id}.mp4",
            background=None,
        )

    except httpx.HTTPStatusError as e:
        print(f"[Crop {job_id}] Download failed: {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"Failed to download source video: {e.response.status_code}")
    except subprocess.TimeoutExpired:
        print(f"[Crop {job_id}] FFmpeg timeout")
        raise HTTPException(status_code=504, detail="FFmpeg timed out (video too long?)")
    except Exception as e:
        print(f"[Crop {job_id}] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass
