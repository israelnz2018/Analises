import os
import json
import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Optional

router = APIRouter(prefix="/claude")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL_FAST = "claude-sonnet-4-6"
CLAUDE_MODEL_BEST = "claude-opus-4-6"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01"
}

async def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 4096, model: str = None) -> str:
    if model is None:
        model = CLAUDE_MODEL_FAST
    payload = {
        "model": model,
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

# ─── Estruturas JSON esperadas por ferramenta ──────────────────────

TOOL_STRUCTURES = {
    "sipoc": """{
  "suppliers": ["Fornecedor 1", "Fornecedor 2"],
  "inputs": ["Entrada 1", "Entrada 2"],
  "process": ["Passo 1", "Passo 2", "Passo 3", "Passo 4", "Passo 5"],
  "outputs": ["Saída principal", "Saída secundária"],
  "customers": ["Cliente interno/externo"]
}""",
    "brainstorming": """{
  "ideas": [
    {"id": "1", "text": "x1: Ideia técnica curta", "category": "Método", "author": "IA LBW", "votes": 0},
    {"id": "2", "text": "x2: Outra ideia", "category": "Mão de Obra", "author": "IA LBW", "votes": 0}
  ],
  "brainstormingType": "Causas do problema",
  "brainstormingTopic": "Tema baseado no problema identificado"
}""",
    "gut": """{
  "columns": [
    {"id": "description", "label": "Problema / Oportunidade", "isScore": false},
    {"id": "gravidade", "label": "Gravidade", "isScore": true},
    {"id": "urgencia", "label": "Urgência", "isScore": true},
    {"id": "tendencia", "label": "Tendência", "isScore": true}
  ],
  "opportunities": [
    {"id": "1", "description": "Título do projeto", "gravidade": 5, "urgencia": 3, "tendencia": 5}
  ]
}""",
    "rab": """{
  "columns": [
    {"id": "description", "label": "Problema / Oportunidade", "isScore": false},
    {"id": "rapidez", "label": "Rapidez", "isScore": true},
    {"id": "autonomia", "label": "Autonomia", "isScore": true},
    {"id": "beneficio", "label": "Benefício", "isScore": true}
  ],
  "opportunities": [
    {"id": "1", "description": "Título do projeto", "rapidez": 5, "autonomia": 3, "beneficio": 5}
  ]
}""",
    "brief": """{
  "answers": {
    "q1": "Nome do processo",
    "q2": "Problema com dados quantitativos",
    "q3": "Pessoas e áreas envolvidas",
    "q4": "O que está errado com exemplos",
    "q5": "Riscos se não resolvido",
    "q6": "O que se quer melhorar",
    "q7": "Meta SMART: reduzir X de A para B em Y meses",
    "q8": "Benefícios financeiros e operacionais",
    "q10": "Próximos passos imediatos",
    "q12": "Recursos necessários"
  }
}""",
    "charter": """{
  "title": "Verbo + indicador + processo",
  "date": "DD/MM/AAAA",
  "rev": "00",
  "area": "Área responsável",
  "leader": "",
  "champion": "",
  "problemDefinition": "Problema com baseline quantitativo",
  "problemHistory": "Histórico e riscos",
  "goalDefinition": "Meta SMART completa",
  "kpi": "Y primário: indicador | Y secundário: indicador",
  "scopeIn": "O que está dentro do escopo",
  "scopeOut": "O que está fora do escopo",
  "businessContributions": "1. Benefício financeiro. 2. Operacional. 3. Cliente.",
  "stakeholders": [
    {"role": "Líder:", "name": "", "definition": "R", "measurement": "A", "analysis": "R", "improvement": "R", "control": "R"},
    {"role": "Patrocinador:", "name": "", "definition": "A", "measurement": "I", "analysis": "I", "improvement": "A", "control": "I"}
  ]
}""",
    "measureIshikawa": """{
  "categories": ["Método", "Máquina", "Medida", "Meio Ambiente", "Mão de Obra", "Material"],
  "causes": {
    "Método": ["x1: Causa curta máximo 6 palavras"],
    "Máquina": [],
    "Medida": [],
    "Meio Ambiente": [],
    "Mão de Obra": [],
    "Material": []
  },
  "problem": "Problema central do projeto"
}""",
    "measureMatrix": """{
  "outputs": [
    {"name": "Y principal — Indicador", "importance": 10}
  ],
  "causes": [
    {"id": "X01", "name": "Causa da Espinha de Peixe", "scores": [9], "effort": 1, "selected": false}
  ]
}""",
    "dataCollection": """{
  "items": [
    {
      "id": "1",
      "data": {
        "variable": "ID - Nome da variável",
        "priority": "Alta",
        "operationalDefinition": "O QUE MEDIR: procedimento técnico",
        "msa": "Sim",
        "method": "Quantitativa",
        "stratification": "Por turno, operador",
        "responsible": "Responsável",
        "when": "Frequência",
        "howMany": "Quantidade"
      }
    }
  ]
}""",
    "fiveWhys": """{
  "chains": [
    {
      "id": "1",
      "problem": "Problema central",
      "whys": ["Por que 1", "Por que 2", "Por que 3", "Por que 4", "Por que 5"],
      "rootCause": "Causa raiz identificada"
    }
  ]
}""",
    "fmea": """{
  "items": [
    {
      "id": "1",
      "processStep": "Etapa do processo",
      "failureMode": "Como pode falhar",
      "failureEffect": "Impacto da falha",
      "severity": 7,
      "causes": "Causas da falha",
      "occurrence": 5,
      "controls": "Controles atuais",
      "detection": 4,
      "actions": "Ações recomendadas"
    }
  ]
}""",
    "plan5w2h": """{
  "actions": [
    {
      "id": "1",
      "variable": "Causa origem",
      "what": "O que será feito",
      "why": "Por que resolve",
      "where": "Onde executar",
      "when": "DD/MM/AAAA",
      "who": "Responsável",
      "how": "Como executar",
      "howMuch": "Custo estimado",
      "status": {"state": "green", "progress": "0%"}
    }
  ]
}""",
    "sop": """{
  "title": "Título do POP",
  "objective": "Objetivo do procedimento",
  "scope": "Abrangência",
  "responsibilities": "Responsáveis",
  "steps": [
    {"id": "1", "title": "Título do passo", "description": "Descrição detalhada", "warning": ""}
  ],
  "frequency": "Frequência de revisão",
  "kpis": "Indicadores associados"
}""",
    "effortImpact": """{
  "actions": [
    {"id": "1", "label": "X1", "description": "Descrição da ação", "effort": 3, "impact": 5}
  ]
}""",
    "directObservation": """{
  "observations": [
    {
      "id": "1",
      "variable": "Variável qualitativa",
      "operationalDefinition": "Definição operacional",
      "identifiedCause": false,
      "observationDescription": "",
      "images": [],
      "aiSuggestions": {
        "trueHypothesis": "Situação que CONFIRMA a causa raiz",
        "falseHypothesis": "Situação onde nenhum desvio foi encontrado"
      }
    }
  ]
}""",
    "dataNature": """{
  "analyses": [
    {
      "id": "1",
      "variableY": {"name": "Nome Y", "type": "Contínuo", "description": "Por que é Y"},
      "variableX": {"name": "Nome X", "type": "Discreto", "description": "Por que é X"},
      "quadrant": "Y Contínuo / X Discreto",
      "recommendedTools": ["Box Plot", "ANOVA"],
      "explanation": "Explicação técnica da recomendação"
    }
  ]
}""",
    "stakeholders": """{
  "stakeholders": [
    {"role": "Líder:", "name": "", "definition": "R", "measurement": "A", "analysis": "R", "improvement": "R", "control": "R"}
  ]
}""",
}

# ─── Instruções específicas por ferramenta ─────────────────────────

TOOL_SPECIFIC_INSTRUCTIONS = {
    "sipoc": """
ATENÇÃO — SIPOC:
- suppliers: quem fornece entradas para o processo
- inputs: materiais, informações ou recursos que entram
- process: exatamente 5 passos principais do processo
- outputs: resultados ou produtos do processo
- customers: quem recebe as saídas
Use dados reais do contexto do projeto.
""",
    "brainstorming": """
ATENÇÃO — BRAINSTORMING:
- Gere mínimo 12 ideias distribuídas nos 6Ms: Método, Mão de Obra, Material, Máquina, Meio Ambiente, Medição
- Prefixe cada ideia com "x1:", "x2:", etc.
- Ideias curtas e técnicas — máximo 8 palavras cada
- Baseie nas informações do projeto
""",
    "measureIshikawa": """
ATENÇÃO — ESPINHA DE PEIXE:
- Use ideias do Brainstorming como causas se disponíveis
- Distribua nos 6Ms corretamente
- Frases EXTREMAMENTE curtas — máximo 6 palavras por causa
- O "problem" deve ser o problema central do projeto
""",
    "measureMatrix": """
ATENÇÃO — MATRIZ CAUSA E EFEITO:
- outputs: use os KPIs do projeto como Y's com importance 10
- causes: use as causas da Espinha de Peixe como X's
- scores: correlação 0=sem relação, 1=fraca, 3=média, 9=forte
- O array "scores" deve ter o mesmo tamanho que "outputs"
""",
    "gut": """
ATENÇÃO — MATRIZ GUT:
- Use os projetos da Ideia de Projetos como opportunities
- Pontuações APENAS: 1, 3 ou 5
- gravidade: 5=Extremamente Grave, 3=Grave, 1=Leve
- urgencia: 5=Imediata, 3=O mais rápido, 1=Pode esperar
- tendencia: 5=Piorar rápido, 3=Irá piorar, 1=Não piora
""",
    "rab": """
ATENÇÃO — MATRIZ RAB:
- Use os projetos da Ideia de Projetos como opportunities
- Pontuações APENAS: 1, 3 ou 5
- rapidez: 5=Imediato/1 mês, 3=1-3 meses, 1=+3 meses
- autonomia: 5=Total, 3=Apoio de outras áreas, 1=Depende de terceiros
- beneficio: 5=Impacto Estratégico, 3=Impacto na Área, 1=Impacto no Processo
""",
    "charter": """
ATENÇÃO — PROJECT CHARTER:
- title: começa com Reduzir/Aumentar/Melhorar/Otimizar — SEM "Lean Six Sigma"
- goalDefinition: formato SMART obrigatório com baseline e target
- scopeIn e scopeOut: ambos obrigatórios
- NÃO invente nomes de pessoas para stakeholders
""",
    "fiveWhys": """
ATENÇÃO — 5 PORQUÊS:
- Use as causas mais críticas da Espinha de Peixe ou Matriz C&E
- Cada "por que" deve aprofundar o anterior
- rootCause deve ser uma causa sistêmica real
""",
    "fmea": """
ATENÇÃO — FMEA:
- Baseie os modos de falha nas causas da Espinha de Peixe
- severity (1-10): impacto no cliente
- occurrence (1-10): frequência da falha
- detection (1-10): dificuldade de detectar
- RPN = severity × occurrence × detection
""",
    "plan5w2h": """
ATENÇÃO — PLANO DE AÇÃO 5W2H:
- Baseie as ações nas causas confirmadas (FMEA, 5 Porquês)
- "what": verbo + objeto + resultado esperado
- "who": cargo/função, não nome genérico
- "when": datas realistas no formato DD/MM/AAAA
""",
    "dataNature": """
ATENÇÃO — NATUREZA DOS DADOS:
- Y Contínuo + X Contínuo → Regressão Linear, Dispersão
- Y Contínuo + X Discreto → Box Plot, ANOVA, Teste T
- Y Discreto + X Contínuo → Regressão Logística
- Y Discreto + X Discreto → Qui-quadrado, Pareto
""",
}

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

        structure = TOOL_STRUCTURES.get(req.toolId, "{}")
        specific_instruction = TOOL_SPECIFIC_INSTRUCTIONS.get(req.toolId, "")

        system_prompt = """Você é um consultor sênior Master Black Belt em Lean Six Sigma.
Use os dados já preenchidos nas ferramentas anteriores para pré-preencher a próxima ferramenta.
REGRAS CRÍTICAS:
1. Use APENAS informações do contexto fornecido — nunca invente dados.
2. Mantenha consistência absoluta com fases anteriores.
3. Retorne EXCLUSIVAMENTE um objeto JSON válido sem explicações e sem markdown.
4. Se um campo não puder ser inferido, use string vazia "".
5. Qualidade de consultoria sênior.
6. Responda em português do Brasil."""

        user_prompt = f"""
Projeto: "{req.projectInfo.get('name', 'Projeto de Melhoria') if req.projectInfo else 'Projeto de Melhoria'}"

{project_context}

FERRAMENTA A PREENCHER: "{req.toolName}" (ID: {req.toolId})

{specific_instruction}

ESTRUTURA JSON ESPERADA (use exatamente esta estrutura):
{structure}

Retorne EXCLUSIVAMENTE o JSON preenchido com dados reais do projeto.
Sem explicações, sem markdown, sem backticks.
"""

        result = await call_claude(system_prompt, user_prompt)
        clean = result.replace("```json", "").replace("```", "").strip()

        # Remove possível texto antes do JSON
        if clean.startswith("{") is False and "{" in clean:
            clean = clean[clean.index("{"):]
        if clean.startswith("[") is False and "[" in clean:
            if clean.index("[") < clean.index("{") if "{" in clean else True:
                clean = clean[clean.index("["):]

        parsed = json.loads(clean)
        return {"success": True, "data": parsed}

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON inválido retornado pelo Claude: {str(e)}", "raw": result[:500] if 'result' in locals() else ""}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Rota 2: Gerar relatório da ferramenta ────────────────────────

@router.post("/report")
async def generate_report(req: ReportRequest):
    try:
        # Usa Opus para relatório consolidado, Sonnet para individual
        model = CLAUDE_MODEL_BEST if req.toolName == "consolidated" else CLAUDE_MODEL_FAST

        tool_specific = {
            "Brainstorming": "Gere tabela Markdown: | Nº | Categoria | Ideia | Prioridade |. Top 3 ideias em destaque.",
            "Espinha de Peixe": "Estruture pelos 6Ms. Destaque as 3 causas mais críticas. Tabela de priorização.",
            "Plano de Ação 5W2H": "Tabela: | O Quê | Por Quê | Onde | Quando | Quem | Como | Quanto | Status |",
            "Project Charter": "Documento executivo: Problema, Meta SMART, Escopo, Equipe, Benefícios.",
            "SIPOC": "Tabela SIPOC completa. Análise dos pontos críticos do fluxo.",
            "FMEA": "Tabela FMEA ordenada por RPN decrescente. Ações prioritárias.",
        }

        specific = tool_specific.get(req.toolName, f"Relatório executivo profissional para {req.toolName}.")

        system_prompt = """Você é um consultor sênior Master Black Belt em Lean Six Sigma especializado
em geração de relatórios executivos profissionais.
REGRAS:
1. Use APENAS os dados fornecidos.
2. Melhore a redação para padrão executivo de consultoria.
3. Seja conciso — cabe em uma página A4.
4. Use tabelas Markdown, negrito, títulos hierárquicos.
5. Termine com "## Próximos Passos Recomendados" com 2 a 3 ações concretas.
6. Idioma: Português do Brasil."""

        user_prompt = f"""
PROJETO: {req.projectName}
FERRAMENTA: {req.toolName}
DADOS: {json.dumps(req.toolData, ensure_ascii=False, indent=2)}

INSTRUÇÃO ESPECÍFICA: {specific}

Gere o relatório executivo agora.
"""

        result = await call_claude(system_prompt, user_prompt, max_tokens=2048, model=model)
        return {"success": True, "report": result}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Rota 3: Mentor LBW ───────────────────────────────────────────

@router.post("/mentor")
async def mentor_chat(req: MentorRequest):
    try:
        phase_context = {
            "PreDefinir": "O usuário está na fase Pré-Definir, identificando e priorizando oportunidades de melhoria.",
            "Define": "O usuário está na fase Definir, estruturando o escopo, problema e objetivos do projeto.",
            "Measure": "O usuário está na fase Medir, mapeando o processo e coletando dados da situação atual.",
            "Analyze": "O usuário está na fase Analisar, identificando causas raiz do problema com dados.",
            "Improve": "O usuário está na fase Melhorar, desenvolvendo e implementando soluções.",
            "Control": "O usuário está na fase Controlar, sustentando os ganhos e padronizando melhorias.",
        }

        system_prompt = f"""Você é o Mentor LBW — um consultor sênior Master Black Belt em Lean Six Sigma
com 20 anos de experiência em projetos de melhoria de processos.

PRINCÍPIOS:
- Seja direto e técnico. Evite respostas genéricas.
- Use sempre os dados do projeto do usuário para personalizar cada resposta.
- Quando sugerir uma próxima ação, seja específico.
- Use linguagem de consultoria executiva — profissional mas acessível.
- Responda em português do Brasil.

CONTEXTO ATUAL: {phase_context.get(req.currentPhase, "")}
Ferramenta ativa: {req.currentTool}

DADOS DO PROJETO:
{json.dumps(req.projectData, ensure_ascii=False, indent=2) if req.projectData else "Nenhum dado disponível ainda."}"""

        messages = list(req.history) if req.history else []
        messages.append({"role": "user", "content": req.message})

        payload = {
            "model": CLAUDE_MODEL_FAST,
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

# ─── Rota 4: Sugestões contextuais do Mentor ──────────────────────

@router.post("/mentor-suggestions")
async def mentor_suggestions(req: dict):
    try:
        current_phase = req.get("currentPhase", "")
        current_tool = req.get("currentTool", "")
        completed_tools = req.get("completedTools", [])
        project_data = req.get("projectData", {})

        system_prompt = """Você é o Mentor LBW. Gere exatamente 3 sugestões de perguntas curtas e relevantes
que um profissional faria neste momento do projeto DMAIC.
As sugestões devem ser:
- Específicas para a fase e ferramenta atual
- Baseadas nos dados já preenchidos
- Máximo 8 palavras cada
Retorne EXCLUSIVAMENTE um array JSON com 3 strings. Sem explicações.
Exemplo: ["Como escrever uma meta SMART?", "Qual o próximo passo?", "Como calcular o impacto?"]"""

        user_prompt = f"""
Fase atual: {current_phase}
Ferramenta atual: {current_tool}
Ferramentas concluídas: {', '.join(completed_tools)}
Contexto: {json.dumps(project_data, ensure_ascii=False)}
Gere as 3 sugestões agora.
"""

        result = await call_claude(system_prompt, user_prompt, max_tokens=200)
        clean = result.replace("```json", "").replace("```", "").strip()
        suggestions = json.loads(clean)
        return {"success": True, "suggestions": suggestions}

    except Exception:
        # Fallback por fase
        fallbacks = {
            "PreDefinir": ["Como priorizar os projetos?", "O que é a Matriz GUT?", "Como validar uma ideia?"],
            "Define": ["Como escrever uma meta SMART?", "O que colocar no escopo?", "Como calcular o impacto?"],
            "Measure": ["Como mapear o processo?", "Quais dados coletar?", "O que é MSA?"],
            "Analyze": ["Como identificar a causa raiz?", "Quando usar o 5 Porquês?", "Como usar o Ishikawa?"],
            "Improve": ["Como priorizar as soluções?", "O que é um piloto?", "Como fazer o FMEA?"],
            "Control": ["Como sustentar os ganhos?", "O que é um POP?", "Como monitorar o KPI?"],
        }
        phase = req.get("currentPhase", "")
        return {"success": True, "suggestions": fallbacks.get(phase, ["Qual o próximo passo?", "Como posso melhorar?", "O que é importante aqui?"])}
