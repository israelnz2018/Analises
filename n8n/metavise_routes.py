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

⚠️ NÃO MUDE NADA ALÉM DO QUE ESTÁ ESPECIFICADO ABAIXO.

Estamos adicionando UM novo endpoint ao projeto Analises (FastAPI no Railway). O objetivo é cortar vídeos do HeyGen para 1:1 (ou outros aspect ratios) usando FFmpeg, porque o FFmpeg não funciona no Cloud Run do Google AI Studio onde o Metavise roda.

═══════════════════════════════════════════════════════════════
PASSO 1 — Garantir que ffmpeg está disponível no container do Railway
═══════════════════════════════════════════════════════════════

Railway por padrão usa o Nixpacks builder, que detecta Python automaticamente mas NÃO instala ffmpeg. Precisamos adicionar.

ARQUIVO: nixpacks.toml (na raiz do projeto)

Se o arquivo NÃO existe, CRIE com este conteúdo:

[phases.setup]
nixPkgs = ["python311", "ffmpeg"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

Se o arquivo JÁ existe, ADICIONE "ffmpeg" no array nixPkgs (não mude mais nada).

═══════════════════════════════════════════════════════════════
PASSO 2 — Adicionar httpx ao requirements.txt (se ainda não estiver lá)
═══════════════════════════════════════════════════════════════

ARQUIVO: requirements.txt

VERIFIQUE se "httpx" já está listado. Se NÃO estiver, adicione UMA linha:

httpx

NÃO mude mais nada no arquivo.

═══════════════════════════════════════════════════════════════
PASSO 3 — Adicionar o endpoint de crop ao metavise_routes.py
═══════════════════════════════════════════════════════════════

ARQUIVO: n8n/metavise_routes.py (mesmo arquivo onde está o /metavise/claude)

ADICIONE no FINAL do arquivo (depois de tudo que já existe), exatamente este código:


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
    aspectRatio: str = "1:1"  # "1:1", "16:9", "9:16", "4:5"
    cropOffset: int = 0       # -50 a 50


def _get_dimensions(aspect_ratio: str):
    if aspect_ratio == "1:1":
        return 1080, 1080
    if aspect_ratio == "16:9":
        return 1920, 1080
    if aspect_ratio == "9:16":
        return 1080, 1920
    if aspect_ratio == "4:5":
        return 1080, 1350
    return 1080, 1080  # default square


def _build_ffmpeg_filter(aspect_ratio: str, crop_offset: int):
    w, h = _get_dimensions(aspect_ratio)
    # crop_offset: -50 a 50, controla deslocamento horizontal/vertical do crop
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
        # 1. Baixar o vídeo
        print(f"[Crop {job_id}] Downloading: {req.videoUrl[:80]}...")
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            r = await client.get(req.videoUrl)
            r.raise_for_status()
            with open(input_path, "wb") as f:
                f.write(r.content)
        size_mb = os.path.getsize(input_path) / 1024 / 1024
        print(f"[Crop {job_id}] Downloaded {size_mb:.1f}MB")

        # 2. Rodar FFmpeg
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

        # 3. Retornar o arquivo cortado
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"cropped_{job_id}.mp4",
            background=None,  # mantém o arquivo até resposta enviar
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
        # Limpa o input. Output é deletado depois pela próxima request ou reinício do container.
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass


═══════════════════════════════════════════════════════════════
PASSO 4 — NÃO MEXA EM MAIS NADA
═══════════════════════════════════════════════════════════════

- NÃO mude main.py
- NÃO mude o endpoint /metavise/claude existente
- NÃO mude /metavise/health existente
- NÃO renomeie nada, NÃO refatore nada

═══════════════════════════════════════════════════════════════
DEPOIS DE FAZER DEPLOY NO RAILWAY
═══════════════════════════════════════════════════════════════

Confirme listando:
1. Que arquivos foram criados/alterados (caminho exato + linhas)
2. Que o deploy no Railway foi bem-sucedido
3. Que o endpoint responde testando manualmente:
   GET https://analises-production.up.railway.app/metavise/health
   (deve continuar funcionando, sem mudanças)

⚠️ NÃO MUDE NADA ALÉM DO QUE ESTÁ ESPECIFICADO ACIMA.
