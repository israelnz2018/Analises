import os
import json
import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Optional

router = APIRouter(prefix="/claude")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01"
}

async def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(ANTHROPIC_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

# ─── Modelos de request ────────────────────────────────────────────

class ToolDataRequest(BaseModel):
    toolId: str
    toolName: str
    previousToolName: Optional[str] = None
    previousToolData: Optional[Any] = None
    projectInfo: Optional[dict] = None
    allProjectData: Optional[Any] = None

class ReportRequest(BaseModel):
    toolName: str
    toolData: Any
    projectName: str

class MentorRequest(BaseModel):
    message: str
    currentPhase: str
    currentTool: str
    projectData: Optional[Any] = None
    history: Optional[list] = []

# ─── Rota 1: Gerar dados da ferramenta ────────────────────────────

@router.post("/tool-data")
async def generate_tool_data(req: ToolDataRequest):
    try:
        project_context = ""
        if req.allProjectData:
            project_context = f"""
CONTEXTO COMPLETO DO PROJETO "{req.projectInfo.get('name', '') if req.projectInfo else ''}":
{json.dumps(req.allProjectData, ensure_ascii=False, indent=2)}
"""
        elif req.previousToolData:
            project_context = f"""
DADOS DA FERRAMENTA ANTERIOR ("{req.previousToolName}"):
{json.dumps(req.previousToolData, ensure_ascii=False, indent=2)}
"""

        system_prompt = """Você é um consultor sênior Master Black Belt em Lean Six Sigma.
Use os dados já preenchidos nas ferramentas anteriores para pré-preencher a próxima ferramenta.
REGRAS CRÍTICAS:
1. Use APENAS informações do contexto fornecido — nunca invente dados.
2. Mantenha consistência absoluta com fases anteriores.
3. Retorne EXCLUSIVAMENTE um objeto JSON válido sem explicações e sem markdown.
4. Se um campo não puder ser inferido, use string vazia.
5. Qualidade de consultoria sênior."""

        user_prompt = f"""
Projeto: "{req.projectInfo.get('name', 'Projeto de Melhoria') if req.projectInfo else 'Projeto de Melhoria'}"

{project_context}

FERRAMENTA A PREENCHER: "{req.toolName}" (ID: {req.toolId})

Retorne EXCLUSIVAMENTE o JSON preenchido, sem explicações, sem markdown, sem backticks.
"""

        result = await call_claude(system_prompt, user_prompt)
        clean = result.replace("```json", "").replace("```", "").strip()
        return {"success": True, "data": json.loads(clean)}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Rota 2: Gerar relatório da ferramenta ────────────────────────

@router.post("/report")
async def generate_report(req: ReportRequest):
    try:
        system_prompt = """Você é um consultor sênior Master Black Belt em Lean Six Sigma especializado 
em geração de relatórios executivos profissionais.
Analise os dados de uma ferramenta de qualidade e gere um RELATÓRIO EXECUTIVO em Markdown.
REGRAS:
1. Use APENAS os dados fornecidos.
2. Melhore a redação para padrão executivo.
3. Seja conciso — cabe em uma página A4.
4. Use tabelas, negrito, títulos hierárquicos.
5. Termine com "Próximos Passos Recomendados" com 2 a 3 ações concretas.
6. Idioma: Português do Brasil."""

        user_prompt = f"""
PROJETO: {req.projectName}
FERRAMENTA: {req.toolName}
DADOS: {json.dumps(req.toolData, ensure_ascii=False, indent=2)}

Gere o relatório executivo agora.
"""

        result = await call_claude(system_prompt, user_prompt, max_tokens=2048)
        return {"success": True, "report": result}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Rota 3: Mentor LBW ───────────────────────────────────────────

@router.post("/mentor")
async def mentor_chat(req: MentorRequest):
    try:
        phase_context = {
            "PreDefinir": "O usuário está na fase Pré-Definir, identificando oportunidades de melhoria.",
            "Define": "O usuário está na fase Definir, estruturando o problema e objetivos.",
            "Measure": "O usuário está na fase Medir, mapeando o processo e coletando dados.",
            "Analyze": "O usuário está na fase Analisar, identificando causas raiz.",
            "Improve": "O usuário está na fase Melhorar, desenvolvendo soluções.",
            "Control": "O usuário está na fase Controlar, sustentando os ganhos.",
        }

        system_prompt = f"""Você é o Mentor LBW — um consultor sênior Master Black Belt em Lean Six Sigma 
com 20 anos de experiência.
Seja direto e técnico. Use sempre os dados do projeto do usuário.
Responda em português do Brasil.

CONTEXTO ATUAL: {phase_context.get(req.currentPhase, "")}
Ferramenta ativa: {req.currentTool}

DADOS DO PROJETO:
{json.dumps(req.projectData, ensure_ascii=False, indent=2) if req.projectData else "Nenhum dado disponível"}"""

        messages = req.history or []
        messages.append({"role": "user", "content": req.message})

        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": messages
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(ANTHROPIC_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()
            answer = data["content"][0]["text"]

        return {"success": True, "message": answer}

    except Exception as e:
        return {"success": False, "error": str(e)}
