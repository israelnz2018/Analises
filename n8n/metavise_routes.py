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
