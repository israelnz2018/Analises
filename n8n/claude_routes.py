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

TOOL_STRUCTURES = {}
TOOL_SPECIFIC_INSTRUCTIONS = {}

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

# ════════════════════════════════════════════════════════════════
# FERRAMENTAS — ESTRUTURAS E INSTRUÇÕES
# Para atualizar uma ferramenta: apague o bloco dela e cole o novo
# Para adicionar ferramenta nova: cole o bloco no final desta seção
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════
# FERRAMENTA: SIPOC
# ════════════════════════════════════════
TOOL_STRUCTURES["sipoc"] = """{
  "suppliers": ["Fornecedor 1", "Fornecedor 2"],
  "inputs": ["Entrada 1", "Entrada 2"],
  "process": ["Passo 1", "Passo 2", "Passo 3", "Passo 4", "Passo 5"],
  "outputs": ["Saida principal", "Saida secundaria"],
  "customers": ["Cliente interno/externo"]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["sipoc"] = """
ATENCAO - SIPOC:
- suppliers: quem fornece entradas para o processo
- inputs: materiais, informacoes ou recursos que entram
- process: exatamente 5 passos principais do processo
- outputs: resultados ou produtos do processo
- customers: quem recebe as saidas
Use dados reais do contexto do projeto.
"""

# ════════════════════════════════════════
# FERRAMENTA: BRAINSTORMING
# ════════════════════════════════════════
TOOL_STRUCTURES["brainstorming"] = """{
  "ideas": [
    {"id": "1", "text": "x1: Ideia tecnica curta", "category": "Metodo", "author": "IA LBW", "votes": 0},
    {"id": "2", "text": "x2: Outra ideia", "category": "Mao de Obra", "author": "IA LBW", "votes": 0}
  ],
  "brainstormingType": "Causas do problema",
  "brainstormingTopic": "Tema baseado no problema identificado"
}"""
TOOL_SPECIFIC_INSTRUCTIONS["brainstorming"] = """
ATENCAO - BRAINSTORMING:
- Gere minimo 12 ideias distribuidas nos 6Ms: Metodo, Mao de Obra, Material, Maquina, Meio Ambiente, Medicao
- Prefixe cada ideia com x1:, x2:, etc.
- Ideias curtas e tecnicas - maximo 8 palavras cada
- Baseie nas informacoes do projeto
"""

# ════════════════════════════════════════
# FERRAMENTA: MATRIZ GUT
# ════════════════════════════════════════
TOOL_STRUCTURES["gut"] = """{
  "columns": [
    {"id": "description", "label": "Problema / Oportunidade", "isScore": false},
    {"id": "gravidade", "label": "Gravidade", "isScore": true},
    {"id": "urgencia", "label": "Urgencia", "isScore": true},
    {"id": "tendencia", "label": "Tendencia", "isScore": true}
  ],
  "opportunities": [
    {"id": "1", "description": "Titulo do projeto", "gravidade": 5, "urgencia": 3, "tendencia": 5}
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["gut"] = """
ATENCAO - MATRIZ GUT:
- Use os projetos da Ideia de Projetos como opportunities
- Pontuacoes APENAS: 1, 3 ou 5
- gravidade: 5=Extremamente Grave, 3=Grave, 1=Leve
- urgencia: 5=Imediata, 3=O mais rapido, 1=Pode esperar
- tendencia: 5=Piorar rapido, 3=Ira piorar, 1=Nao piora
"""

# ════════════════════════════════════════
# FERRAMENTA: MATRIZ RAB
# ════════════════════════════════════════
TOOL_STRUCTURES["rab"] = """{
  "columns": [
    {"id": "description", "label": "Problema / Oportunidade", "isScore": false},
    {"id": "rapidez", "label": "Rapidez", "isScore": true},
    {"id": "autonomia", "label": "Autonomia", "isScore": true},
    {"id": "beneficio", "label": "Beneficio", "isScore": true}
  ],
  "opportunities": [
    {"id": "1", "description": "Titulo do projeto", "rapidez": 5, "autonomia": 3, "beneficio": 5}
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["rab"] = """
ATENCAO - MATRIZ RAB:
- Use os projetos da Ideia de Projetos como opportunities
- Pontuacoes APENAS: 1, 3 ou 5
- rapidez: 5=Imediato/1 mes, 3=1-3 meses, 1=+3 meses
- autonomia: 5=Total, 3=Apoio de outras areas, 1=Depende de terceiros
- beneficio: 5=Impacto Estrategico, 3=Impacto na Area, 1=Impacto no Processo
"""

# ════════════════════════════════════════
# FERRAMENTA: ENTENDENDO O PROBLEMA (BRIEF)
# ════════════════════════════════════════
TOOL_STRUCTURES["brief"] = """{
  "answers": {
    "q1": "Nome do processo",
    "q2": "Problema com dados quantitativos",
    "q3": "Pessoas e areas envolvidas",
    "q4": "O que esta errado com exemplos",
    "q5": "Riscos se nao resolvido",
    "q6": "O que se quer melhorar",
    "q7": "Meta SMART: reduzir X de A para B em Y meses",
    "q8": "Beneficios financeiros e operacionais",
    "q10": "Proximos passos imediatos",
    "q12": "Recursos necessarios"
  }
}"""
TOOL_SPECIFIC_INSTRUCTIONS["brief"] = """
ATENCAO - ENTENDENDO O PROBLEMA:
- Use o projeto selecionado como foco central
- q7 deve ser uma meta SMART com indicador, baseline e target
- q2 deve ter pelo menos um numero ou percentual
- Linguagem tecnica e executiva
"""

# ════════════════════════════════════════
# FERRAMENTA: PROJECT CHARTER
# ════════════════════════════════════════
TOOL_STRUCTURES["charter"] = """{
  "title": "Verbo + indicador + processo sem Lean Six Sigma",
  "date": "DD/MM/AAAA",
  "rev": "00",
  "area": "Area responsavel",
  "leader": "",
  "champion": "",
  "problemDefinition": "Problema com baseline quantitativo",
  "problemHistory": "Historico e riscos",
  "goalDefinition": "Meta SMART completa com baseline e target",
  "kpi": "Y primario: indicador | Y secundario: indicador",
  "scopeIn": "O que esta dentro do escopo",
  "scopeOut": "O que esta fora do escopo",
  "businessContributions": "1. Beneficio financeiro. 2. Operacional. 3. Cliente.",
  "team": [
    {"role": "Black Belt", "name": "", "definition": "A", "measurement": "A", "analysis": "A", "improvement": "A", "control": "A"},
    {"role": "Champion", "name": "", "definition": "P", "measurement": "", "analysis": "", "improvement": "P", "control": "P"},
    {"role": "Patrocinador / Sponsor", "name": "", "definition": "P", "measurement": "", "analysis": "", "improvement": "P", "control": "P"}
  ]
  
}"""
TOOL_SPECIFIC_INSTRUCTIONS["charter"] = """
ATENCAO - PROJECT CHARTER:
- title: comeca com Reduzir/Aumentar/Melhorar/Otimizar - SEM Lean Six Sigma
- goalDefinition: formato SMART obrigatorio com baseline e target numericos
- scopeIn e scopeOut: ambos obrigatorios
- NAO invente nomes de pessoas
- problemDefinition deve ter pelo menos um numero
- team[].role: usar APENAS estes papeis Lean Six Sigma:
  Patrocinador / Sponsor, Champion Executive, Champion,
  Process Owner, Master Black Belt (MBB), Black Belt,
  Green Belt, Yellow Belt, White Belt, Team Member / SME,
  Gestor de Area Impactada, Operador / Frontline,
  Cliente / Usuario Final, Fornecedor / Suporte, Outro
- team[].definition/measurement/analysis/improvement/control:
  "A" = participa ativamente nesta fase DMAIC
  "P" = apenas informado/consultado nesta fase
  ""  = nao participa nesta fase
- REGRA: quem tem pelo menos um "A" = faz parte do time do projeto
         quem tem apenas "P" ou vazio = e um impactado
"""

# ════════════════════════════════════════
# FERRAMENTA: ESPINHA DE PEIXE (ISHIKAWA)
# ════════════════════════════════════════
TOOL_STRUCTURES["measureIshikawa"] = """{
  "categories": ["Metodo", "Maquina", "Medida", "Meio Ambiente", "Mao de Obra", "Material"],
  "causes": {
    "Metodo": ["x1: Causa curta maximo 6 palavras"],
    "Maquina": [],
    "Medida": [],
    "Meio Ambiente": [],
    "Mao de Obra": [],
    "Material": []
  },
  "problem": "Problema central do projeto"
}"""
TOOL_SPECIFIC_INSTRUCTIONS["measureIshikawa"] = """
ATENCAO - ESPINHA DE PEIXE:
- Use ideias do Brainstorming como causas se disponiveis
- Distribua nos 6Ms corretamente
- Frases EXTREMAMENTE curtas - maximo 6 palavras por causa
- O problem deve ser o problema central do projeto
- Minimo 2 causas por categoria
"""

# ════════════════════════════════════════
# FERRAMENTA: MATRIZ CAUSA E EFEITO
# ════════════════════════════════════════
TOOL_STRUCTURES["measureMatrix"] = """{
  "outputs": [
    {"name": "Y principal - Indicador", "importance": 10}
  ],
  "causes": [
    {"id": "X01", "name": "Causa da Espinha de Peixe", "scores": [9], "effort": 1, "selected": false}
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["measureMatrix"] = """
ATENCAO - MATRIZ CAUSA E EFEITO:
- outputs: use os KPIs do projeto como Y com importance 10
- causes: use as causas da Espinha de Peixe como X
- scores: correlacao 0=sem relacao, 1=fraca, 3=media, 9=forte
- O array scores deve ter o mesmo tamanho que outputs
"""

# ════════════════════════════════════════
# FERRAMENTA: PLANO DE COLETA DE DADOS
# ════════════════════════════════════════
TOOL_STRUCTURES["dataCollection"] = """{
  "items": [
    {
      "id": "1",
      "data": {
        "variable": "ID - Nome da variavel",
        "priority": "Alta",
        "operationalDefinition": "O QUE MEDIR: procedimento tecnico",
        "msa": "Sim",
        "method": "Quantitativa",
        "stratification": "Por turno, operador",
        "responsible": "Responsavel",
        "when": "Frequencia",
        "howMany": "Quantidade"
      }
    }
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["dataCollection"] = """
ATENCAO - PLANO DE COLETA:
- Use APENAS causas com selected=true da Matriz C&E se disponivel
- Quantitativa: envolve numeros, tempos, dimensoes
- Qualitativa: envolve auditoria visual, Sim/Nao
- operationalDefinition no formato: O QUE MEDIR: procedimento tecnico
"""

# ════════════════════════════════════════
# FERRAMENTA: 5 PORQUES
# ════════════════════════════════════════
TOOL_STRUCTURES["fiveWhys"] = """{
  "chains": [
    {
      "id": "1",
      "problem": "Problema central",
      "whys": ["Por que 1", "Por que 2", "Por que 3", "Por que 4", "Por que 5"],
      "rootCause": "Causa raiz identificada"
    }
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["fiveWhys"] = """
ATENCAO - 5 PORQUES:
- Use as causas mais criticas da Espinha de Peixe ou Matriz C&E
- Cada por que deve aprofundar o anterior
- rootCause deve ser uma causa sistemica real
- Gere uma cadeia por causa principal identificada
"""

# ════════════════════════════════════════
# FERRAMENTA: FMEA
# ════════════════════════════════════════
TOOL_STRUCTURES["fmea"] = """{
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
      "actions": "Acoes recomendadas"
    }
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["fmea"] = """
ATENCAO - FMEA:
- Baseie os modos de falha nas causas da Espinha de Peixe
- severity (1-10): impacto no cliente
- occurrence (1-10): frequencia da falha
- detection (1-10): dificuldade de detectar
- RPN = severity x occurrence x detection
- Ordene por RPN decrescente
"""

# ════════════════════════════════════════
# FERRAMENTA: PLANO DE ACAO 5W2H
# ════════════════════════════════════════
TOOL_STRUCTURES["plan5w2h"] = """{
  "actions": [
    {
      "id": "1",
      "variable": "Causa origem",
      "what": "O que sera feito",
      "why": "Por que resolve",
      "where": "Onde executar",
      "when": "DD/MM/AAAA",
      "who": "Responsavel",
      "how": "Como executar",
      "howMuch": "Custo estimado",
      "status": {"state": "green", "progress": "0%"}
    }
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["plan5w2h"] = """
ATENCAO - PLANO DE ACAO 5W2H:
- Baseie as acoes nas causas confirmadas (FMEA, 5 Porques)
- what: verbo + objeto + resultado esperado
- who: cargo/funcao, nao nome generico
- when: datas realistas no formato DD/MM/AAAA
"""

# ════════════════════════════════════════
# FERRAMENTA: POP (PROCEDIMENTO OPERACIONAL)
# ════════════════════════════════════════
TOOL_STRUCTURES["sop"] = """{
  "title": "Titulo do POP",
  "objective": "Objetivo do procedimento",
  "scope": "Abrangencia",
  "responsibilities": "Responsaveis",
  "steps": [
    {"id": "1", "title": "Titulo do passo", "description": "Descricao detalhada", "warning": ""}
  ],
  "frequency": "Frequencia de revisao",
  "kpis": "Indicadores associados"
}"""
TOOL_SPECIFIC_INSTRUCTIONS["sop"] = """
ATENCAO - POP:
- Baseie os passos nas acoes do Plano de Acao 5W2H
- steps deve ter entre 5 e 10 passos claros e executaveis
- kpis: use os indicadores do Project Charter
- responsibilities: use os responsaveis do Plano de Acao
"""

# ════════════════════════════════════════
# FERRAMENTA: ESFORCO X IMPACTO
# ════════════════════════════════════════
TOOL_STRUCTURES["effortImpact"] = """{
  "actions": [
    {"id": "1", "label": "X1", "description": "Descricao da acao", "effort": 3, "impact": 5}
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["effortImpact"] = """
ATENCAO - ESFORCO X IMPACTO:
- Use as ideias do Brainstorming como actions
- effort: 1=Muito Baixo, 2=Baixo, 3=Medio, 4=Alto, 5=Muito Alto
- impact: 1=Muito Baixo, 2=Baixo, 3=Medio, 4=Alto, 5=Muito Alto
- label: use os prefixos x1, x2, etc. do Brainstorming
"""

# ════════════════════════════════════════
# FERRAMENTA: OBSERVACAO DIRETA (GEMBA)
# ════════════════════════════════════════
TOOL_STRUCTURES["directObservation"] = """{
  "observations": [
    {
      "id": "1",
      "variable": "Variavel qualitativa",
      "operationalDefinition": "Definicao operacional",
      "identifiedCause": false,
      "observationDescription": "",
      "images": [],
      "aiSuggestions": {
        "trueHypothesis": "Situacao que CONFIRMA a causa raiz",
        "falseHypothesis": "Situacao onde nenhum desvio foi encontrado"
      }
    }
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["directObservation"] = """
ATENCAO - OBSERVACAO DIRETA:
- Use APENAS variaveis QUALITATIVAS do Plano de Coleta
- observationDescription: sempre string vazia (usuario preenche)
- trueHypothesis: simulacao realistica que CONFIRMA a causa raiz
- falseHypothesis: situacao onde nenhum desvio foi encontrado
"""

# ════════════════════════════════════════
# FERRAMENTA: NATUREZA DOS DADOS
# ════════════════════════════════════════
TOOL_STRUCTURES["dataNature"] = """{
  "analyses": [
    {
      "id": "1",
      "variableY": {"name": "Nome Y", "type": "Continuo", "description": "Por que e Y"},
      "variableX": {"name": "Nome X", "type": "Discreto", "description": "Por que e X"},
      "quadrant": "Y Continuo / X Discreto",
      "recommendedTools": ["Box Plot", "ANOVA"],
      "explanation": "Explicacao tecnica da recomendacao"
    }
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["dataNature"] = """
ATENCAO - NATUREZA DOS DADOS:
- Y Continuo + X Continuo: Regressao Linear, Dispersao
- Y Continuo + X Discreto: Box Plot, ANOVA, Teste T
- Y Discreto + X Continuo: Regressao Logistica
- Y Discreto + X Discreto: Qui-quadrado, Pareto
- Use variaveis do Plano de Coleta como base
"""

# ════════════════════════════════════════
# FERRAMENTA: ANALISE DE STAKEHOLDERS
# ════════════════════════════════════════
TOOL_STRUCTURES["stakeholders"] = """{
  "stakeholders": [
    {
      "id": "1",
      "name": "Nome extraido do Charter ou contexto do projeto",
      "area": "Area/Departamento",
      "role": "Champion Executive",
      "power": "Alto",
      "interest": "Alto",
      "currentEngagement": "Apoiador",
      "notes": "Observacoes sobre a relacao deste stakeholder com o projeto"
    }
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["stakeholders"] = """
ATENCAO - ANALISE DE STAKEHOLDERS:

FONTES DE DADOS - use nesta ordem:
1. Nomes do Project Charter (lider, champion, equipe)
2. Areas mencionadas no Brief e SIPOC
3. Funcoes implicitas no processo descrito

POWER (Poder):
- Alto: autoridade para aprovar, vetar ou alocar recursos
- Medio: influencia decisoes mas nao as toma sozinho
- Baixo: executa mas nao decide

INTEREST (Interesse):
- Alto: diretamente afetado pelo resultado do projeto
- Medio: afetado indiretamente
- Baixo: pouca relacao com o escopo

CURRENT ENGAGEMENT (Engajamento Atual):
- Lider: defende e lidera ativamente
- Apoiador: favoravel e colaborativo
- Neutro: nem favoravel nem contrario
- Resistente: apresenta obstaculos
- Desconhece: ainda nao foi apresentado ao projeto

ROLES DISPONIVEIS:
Champion Executive, Champion, MBB, Black Belt, Green Belt,
Yellow Belt, Process Owner, Team Member, Sponsor,
Focal Point, Customers, Outro

REGRAS CRITICAS:
1. Use APENAS nomes e areas mencionados no contexto
2. Se nao houver nomes, use cargo/funcao como identificador
3. Gere entre 4 e 8 stakeholders relevantes
4. Lider do projeto: power=Alto, interest=Alto
5. Champion: power=Alto, role=Champion Executive
6. Inclua pelo menos um Resistente ou Neutro para realismo
7. notes: informacao especifica sobre a relacao com o projeto
"""
# ════════════════════════════════════════
# FERRAMENTA: MAPA DE PROCESSO
# ════════════════════════════════════════
TOOL_STRUCTURES["processMap"] = """{
  "nodes": [
    {
      "id": "1",
      "type": "start",
      "position": {"x": 50, "y": 100},
      "data": {"label": "Inicio", "fontSize": 12}
    },
    {
      "id": "2",
      "type": "step",
      "position": {"x": 220, "y": 80},
      "data": {"label": "Nome da atividade", "fontSize": 12, "area": "Nome da area"}
    },
    {
      "id": "3",
      "type": "decision",
      "position": {"x": 420, "y": 75},
      "data": {"label": "Decisao?", "fontSize": 11}
    },
    {
      "id": "4",
      "type": "step",
      "position": {"x": 600, "y": 40},
      "data": {"label": "Atividade sim", "fontSize": 12, "area": "Nome da area"}
    },
    {
      "id": "5",
      "type": "step",
      "position": {"x": 600, "y": 160},
      "data": {"label": "Atividade nao", "fontSize": 12, "area": "Nome da area"}
    },
    {
      "id": "6",
      "type": "end",
      "position": {"x": 800, "y": 100},
      "data": {"label": "Fim", "isEnd": true, "fontSize": 12}
    }
  ],
  "edges": [
    {"id": "e1-2", "source": "1", "target": "2"},
    {"id": "e2-3", "source": "2", "target": "3"},
    {"id": "e3-4", "source": "3", "target": "4", "label": "Sim"},
    {"id": "e3-5", "source": "3", "target": "5", "label": "Nao"},
    {"id": "e4-6", "source": "4", "target": "6"},
    {"id": "e5-6", "source": "5", "target": "6"}
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["processMap"] = """
ATENCAO - MAPA DE PROCESSO:

FONTES DE DADOS - use nesta ordem:
1. SIPOC - use os 5 passos do process[] como base das atividades
2. Brief - use o nome do processo e o problema para contexto
3. Charter - use as areas para identificar quem executa cada passo

REGRAS DE POSICIONAMENTO:
- Inicio sempre em x=50, y=100
- Cada atividade avanca 180px no eixo x
- Atividades na mesma area ficam no mesmo nivel y
- Atividades de areas diferentes ficam em y diferentes (separar 120px)
- Decisoes ficam no centro entre duas atividades

TIPOS DE NOS:
- start: apenas um no inicio do processo
- end: apenas um no final do processo
- step: atividades normais do processo (retangulo)
- decision: pontos de aprovacao ou verificacao (losango)

REGRAS DE CONEXAO (edges):
- Conexoes normais: sem label
- Saida Sim da decisao: label="Sim", style.stroke="#0F6E56"
- Saida Nao da decisao: label="Nao", style.stroke="#993C1D"
- Cada no deve ter pelo menos uma conexao de entrada e uma de saida
- Exceto start (so saida) e end (so entrada)

REGRAS CRITICAS:
1. Use EXATAMENTE os passos do SIPOC como atividades
2. Identifique pontos de aprovacao no processo e crie nos decision
3. Gere entre 5 e 12 nos no total
4. Posicione os nos de forma que nao se sobreponham
5. O campo area em data indica qual departamento executa aquela atividade
6. IDs devem ser strings numericas simples: "1", "2", "3"...
7. IDs das edges devem seguir o padrao: "e1-2", "e2-3", etc.
"""
# ════════════════════════════════════════
# FERRAMENTA: STAKEHOLDER & ADKAR
# ════════════════════════════════════════
TOOL_STRUCTURES["stakeholderAdkar"] = """{
  "stakeholders": [
    {
      "id": "1",
      "name": "Maria Silva",
      "area": "Diretoria Industrial",
      "role": "Patrocinador / Sponsor",
      "type": "Core Team",
      "power": "Alto",
      "interest": "Alto",
      "currentEngagement": "Apoiador",
      "desiredEngagement": "Apoiador",
      "awareness": "Verde",
      "desire": "Verde",
      "knowledge": "Amarelo",
      "ability": "Vermelho",
      "reinforcement": "Vermelho",
      "barrier": "",
      "channel": "Reuniao 1:1",
      "frequency": "Semanal",
      "owner": "",
      "customAction": "",
      "notes": ""
    }
  ]
}"""
TOOL_SPECIFIC_INSTRUCTIONS["stakeholderAdkar"] = """
ATENCAO - STAKEHOLDER & ADKAR:
Voce e um especialista em Lean Six Sigma + Prosci ADKAR.

═════════════════════════════════════════════════════════════════
PRINCIPIO FUNDAMENTAL
═════════════════════════════════════════════════════════════════

Esta ferramenta e usada na fase DEFINE do DMAIC.
Em Define o projeto esta apenas comecando.
A unica acao valida e de COMUNICACAO E ENGAJAMENTO INICIAL.

Toda saida desta ferramenta deve responder UMA pergunta:
"Como cada stakeholder vai ficar SABENDO e QUERENDO participar
do projeto que esta comecando agora?"

Pense como Project Communication Manager, nao como engenheiro
de processo. Suas sugestoes sao sempre conversas, reunioes,
apresentacoes, alinhamentos — nunca atividades tecnicas.

═════════════════════════════════════════════════════════════════
REGRA #1 - QUEM INCLUIR
═════════════════════════════════════════════════════════════════

UNICA FONTE: charter.stakeholders[] do Project Charter
(ou projectCharterPMI.stakeholders[] se charter nao existir).

PROCESSO:
1. Pegue o array charter.stakeholders[].
2. Filtre apenas itens com "name" preenchido (diferente de ""
   e diferente de null).
3. Gere exatamente 1 stakeholder por nome encontrado.

Cada stakeholder e UMA pessoa real do Charter, com nome real.
Se o Charter tem 4 nomes preenchidos, a saida tem 4 stakeholders.
Se tem 7, a saida tem 7. Nunca mais, nunca menos.

Nao adicione pessoas que nao estao no Charter.
Nao use "(a definir)" — se nao tem nome, simplesmente nao inclua.
Cada stakeholder e UMA pessoa real, nunca um grupo
("Equipe de Operadores", "Analistas de Processo" sao grupos,
nao pessoas, e nao devem aparecer).

═════════════════════════════════════════════════════════════════
REGRA #2 - MAPEAR role DO CHARTER PARA role DA FERRAMENTA
═════════════════════════════════════════════════════════════════

O Charter usa roles antigos (com dois pontos). Mapeie assim:

| role no Charter         | role na ferramenta            |
|-------------------------|-------------------------------|
| Líder:                  | Black Belt                    |
| Patrocinador:           | Patrocinador / Sponsor        |
| Dono do Processo:       | Process Owner                 |
| Champion:               | Champion                      |
| Validação Técnica:      | Master Black Belt (MBB)       |
| Validação Financeira:   | Team Member / SME             |
| Membro da Equipe:       | Team Member / SME             |
| Outros:                 | Outro                         |

Se o Charter ja usa um dos 15 papeis Lean Six Sigma diretamente
(Black Belt, Green Belt, Champion, etc), use exatamente esse.

═════════════════════════════════════════════════════════════════
REGRA #3 - CAMPO name vs area
═════════════════════════════════════════════════════════════════

CAMPO "name" = NOME EXATO conforme charter.stakeholders[].name.
- Use o nome exatamente como esta no Charter.
- name e sempre o nome de UMA pessoa.
- Cargo, funcao ou area pertencem aos campos role e area,
  nunca ao campo name.

CAMPO "area" = DEPARTAMENTO / AREA.
- Inferir do contexto (charter.area, Brief, SIPOC).

EXEMPLO CORRETO:
{ "name": "Israel Cavalcanti de Souza",
  "area": "Pintura Automotiva",
  "role": "Black Belt" }

═════════════════════════════════════════════════════════════════
REGRA #4 - CAMPO customAction
═════════════════════════════════════════════════════════════════

NAO gere o campo customAction.
O frontend calcula a acao recomendada automaticamente.
Deixe customAction como string vazia "" em todos os stakeholders.

═════════════════════════════════════════════════════════════════
REGRA #5 - DEFINIR type A PARTIR DOS CAMPOS A/P DO CHARTER
═════════════════════════════════════════════════════════════════

Cada item do charter.stakeholders[] tem os campos:
definition, measurement, analysis, improvement, control
com valores "A" (Ativo), "P" (Passivo), "I" (Informado) ou "".

- Pelo menos um "A" → type = "Core Team"
- Apenas "P", "I" ou vazio → type = "Impactado"

═════════════════════════════════════════════════════════════════
REGRA #6 - desiredEngagement (pelo role mapeado)
═════════════════════════════════════════════════════════════════

| role                         | desiredEngagement |
|------------------------------|-------------------|
| Patrocinador / Sponsor       | Neutro            |
| Champion Executive           | Lider             |
| Champion                     | Lider             |
| Process Owner                | Apoiador          |
| Master Black Belt (MBB)      | Lider             |
| Black Belt                   | Lider             |
| Green Belt                   | Lider             |
| Yellow Belt                  | Lider             |
| White Belt                   | Apoiador          |
| Team Member / SME            | Apoiador          |
| Gestor de Area Impactada     | Apoiador          |
| Operador / Frontline         | Apoiador          |
| Cliente / Usuario Final      | Neutro            |
| Fornecedor / Suporte         | Neutro            |
| Outro                        | Neutro            |

═════════════════════════════════════════════════════════════════
REGRA #7 - currentEngagement
═════════════════════════════════════════════════════════════════

Niveis PMI: Lider, Apoiador, Neutro, Resistente, Desconhece.

Em Define (projeto comecando):
- Core Team: Apoiador ou Neutro
- Impactados: Neutro, Resistente ou Desconhece

═════════════════════════════════════════════════════════════════
REGRA #8 - SEMAFORO ADKAR
═════════════════════════════════════════════════════════════════

Valores validos: "Vermelho", "Amarelo", "Verde".

Em Define:
- Core Team: awareness e desire = Verde ou Amarelo
- Impactados: awareness = Vermelho ou Amarelo
- ability e reinforcement = Vermelho para todos
  (ainda nao implementou nada)

═════════════════════════════════════════════════════════════════
REGRA #9 - CHANNEL e FREQUENCY
═════════════════════════════════════════════════════════════════

| Quadrante                    | channel              | frequency  |
|------------------------------|----------------------|------------|
| Gerenciar de Perto (P+I alto)| Reuniao 1:1          | Semanal    |
| Manter Satisfeito (P alto)   | Steering Committee   | Mensal     |
| Manter Informado (I alto)    | Status Report        | Quinzenal  |
| Monitorar (P+I baixo)        | Comunicado Geral     | Marcos     |
"""
# ════════════════════════════════════════
# FERRAMENTA: DESIRE CHECK (MEASURE)
# ════════════════════════════════════════
TOOL_STRUCTURES["desireCheck"] = """{
  "actions": [
    {
      "stakeholderId": "1",
      "stakeholderName": "Nome da pessoa",
      "role": "Champion",
      "type": "Core Team",
      "desireCurrent": "Vermelho",
      "barrier": "Nao entende o impacto na sua area",
      "action": "Reuniao 1:1 para apresentar o impacto direto",
      "owner": "Black Belt",
      "dueDate": "DD/MM/AAAA",
      "status": "Pendente"
    }
  ],
  "tollgateDecision": "Aprovado com restrições",
  "tollgateNotes": ""
}"""
TOOL_SPECIFIC_INSTRUCTIONS["desireCheck"] = """
ATENCAO - DESIRE CHECK (TOLL-GATE MEASURE):
Voce e um especialista em Lean Six Sigma + Prosci ADKAR.

CONTEXTO:
Esta ferramenta e o toll-gate de entrada na fase Measure.
O Belt precisa garantir que as pessoas QUEREM cooperar antes
de mapear o processo e coletar dados.
Sem Desire, o Belt nao consegue medir nada — as pessoas nao
abrem dados, nao participam do mapeamento, nao deixam observar.

FONTE DE DADOS:
Leia os stakeholders de allProjectData.stakeholderAdkar.stakeholders
Se nao existir, leia de allProjectData.stakeholders.stakeholders

GERE actions PARA TODOS OS STAKEHOLDERS — nao filtre ninguem.
Ordene assim:
1. Vermelho (prioridade maxima)
2. Amarelo (atencao)
3. Verde (monitorar — pode regredir)

PARA CADA STAKEHOLDER:
- stakeholderId: id do stakeholder na ferramenta-mae
- stakeholderName: nome da ferramenta-mae
- role: papel da ferramenta-mae
- type: Core Team ou Impactado da ferramenta-mae
- desireCurrent: valor atual do campo "desire" da ferramenta-mae
- barrier: usar o barrier da ferramenta-mae se existir.
  Se vazio e desire for Vermelho/Amarelo, inferir a resistencia
  mais provavel baseada no papel e contexto do projeto.
  Se desire for Verde, deixar vazio.
- action: sugerir acao concreta para destravar ou monitorar.
  Vermelho: acao urgente e especifica
  Amarelo: acao de engajamento
  Verde: "Monitorar — verificar na proxima reuniao de fase"
- owner: quem deve executar a acao (BB/GB para acoes tecnicas,
  Sponsor para acoes estrategicas, gestor direto para operacionais)
- dueDate: deixar vazio (Belt preenche)
- status: sempre "Pendente" ao gerar

TOLLGATE:
- tollgateDecision: avaliar baseado no percentual de Verde no Core Team:
  >= 70% Verde no Core Team → "Aprovado"
  50-69% Verde no Core Team → "Aprovado com restrições"
  < 50% Verde no Core Team → "Bloqueado"
- tollgateNotes: escrever 2-3 linhas explicando a decisao e
  os principais riscos identificados no Desire do time.
"""
# ════════════════════════════════════════
# FERRAMENTA: KNOWLEDGE CHECK (ANALYZE)
# ════════════════════════════════════════
TOOL_STRUCTURES["knowledgeCheck"] = """{
  "coreTeam": [
    {
      "stakeholderId": "1",
      "stakeholderName": "Nome da pessoa",
      "role": "Champion",
      "type": "Core Team",
      "knowledgeCurrent": "Amarelo",
      "technicalUnderstanding": "Amarelo",
      "agreesWithConclusions": "Parcialmente",
      "objection": "Questiona se a amostra é representativa",
      "response": "Apresentar série histórica de 12 meses com mesma tendência",
      "action": "Reunião técnica para revisar metodologia juntos",
      "owner": "Black Belt",
      "dueDate": "",
      "status": "Pendente"
    }
  ],
  "stakeholders": [
    {
      "stakeholderId": "3",
      "stakeholderName": "Nome da pessoa",
      "role": "Gestor de Área Impactada",
      "type": "Impactado",
      "knowledgeCurrent": "Vermelho",
      "agreesWithConclusions": "Não",
      "objection": "Acredita que o problema é pontual, não sistêmico",
      "response": "Mostrar dados dos últimos 18 meses com frequência e custo acumulado",
      "action": "Apresentação executiva com dashboard histórico",
      "owner": "Black Belt",
      "dueDate": "",
      "status": "Pendente"
    }
  ],
  "tollgateDecision": "Aprovado com restrições",
  "tollgateNotes": ""
}"""
TOOL_SPECIFIC_INSTRUCTIONS["knowledgeCheck"] = """
ATENCAO - KNOWLEDGE CHECK (TOLL-GATE ANALYZE):
Voce e um especialista em Lean Six Sigma + Prosci ADKAR.

CONTEXTO:
Esta ferramenta e o toll-gate de entrada na fase Improve.
O Belt apresentou as analises — causas raiz, variaveis criticas,
correlacoes. Precisa garantir que todos ENTENDEM e ACEITAM
as conclusoes antes de propor solucoes.

Duas dimensoes avaliadas:
1. Entendimento tecnico (principalmente Core Team)
2. Aceitacao das conclusoes (todos os stakeholders)

IMPORTANTE: discordancia nao e um problema — e esperado.
O Belt confronta com fatos e dados historicos.
O campo "response" deve conter a evidencia que responde
a objecao.

FONTE DE DADOS:
- Stakeholders: allProjectData.stakeholderAdkar.stakeholders
- Analises disponiveis: allProjectData (fases Measure e Analyze)

SEPARACAO:
- coreTeam[]: stakeholders com type = "Core Team"
- stakeholders[]: stakeholders com type = "Impactado"

PARA CADA PESSOA:

knowledgeCurrent (estado atual do K no ADKAR):
- Vermelho: nao recebeu informacao sobre as analises
- Amarelo: entende parcialmente mas tem duvidas ou objecoes
- Verde: entende e aceita as conclusoes

technicalUnderstanding (so para Core Team):
- Vermelho: nao entende a metodologia usada
- Amarelo: entende o conceito mas nao os detalhes
- Verde: consegue explicar a analise para outro colega

agreesWithConclusions:
- "Sim": concorda com as conclusoes
- "Parcialmente": concorda em parte, tem ressalvas
- "Não": discorda das conclusoes

objection (se agreesWithConclusions != "Sim"):
- Descrever a objecao especifica baseada no papel da pessoa
  e no contexto do projeto
- Exemplos:
  Gestor: "Acredita que o problema e pontual"
  Operador: "Acha que a causa e outra — equipamento antigo"
  Sponsor: "Questiona o ROI da solucao"
- Se agreesWithConclusions = "Sim": deixar vazio

response (se agreesWithConclusions != "Sim"):
- Sugerir a evidencia ou argumento para responder a objecao
- Sempre baseado em dados e fatos historicos
- Ex: "Mostrar serie historica de 18 meses"
      "Apresentar benchmarking do setor"
      "Comparar com dados de outra planta"
- Se agreesWithConclusions = "Sim": deixar vazio

action:
- Sim: "Validado — manter alinhamento"
- Parcialmente/Nao: acao concreta para resolver a objecao
  antes de apresentar a solucao

owner:
- Objecoes tecnicas: Black Belt
- Objecoes estrategicas: Sponsor ou Champion
- Objecoes operacionais: Gestor de Area

status: sempre "Pendente" ao gerar

TOLLGATE:
- tollgateDecision: avaliar baseado em agreesWithConclusions
  do Core Team:
  Todos "Sim" no Core Team → "Aprovado"
  Algum "Parcialmente" no Core Team → "Aprovado com restrições"
  Algum "Não" no Core Team → "Bloqueado"
- tollgateNotes: 2-3 linhas explicando as principais objecoes
  e como o Belt planeja responder antes de avançar para Improve
"""
# ════════════════════════════════════════
# FERRAMENTA: ABILITY CHECK (IMPROVE)
# ════════════════════════════════════════
TOOL_STRUCTURES["abilityCheck"] = """{
  "solutionValidation": [
    {
      "stakeholderId": "1",
      "stakeholderName": "Nome da pessoa",
      "role": "Champion",
      "type": "Core Team",
      "abilityCurrent": "Amarelo",
      "agreesWithSolution": "Parcialmente",
      "objection": "Preocupação com o tempo de implementação",
      "response": "Apresentar cronograma detalhado com marcos semanais",
      "action": "Reunião para revisar cronograma juntos",
      "owner": "Black Belt",
      "dueDate": "",
      "status": "Pendente"
    }
  ],
  "planApproval": [
    {
      "stakeholderId": "3",
      "stakeholderName": "Nome da pessoa",
      "role": "Gestor de Área Impactada",
      "type": "Impactado",
      "approvesP lan": "Com ressalvas",
      "resourceConcern": "Custo de treinamento não previsto no budget",
      "objection": "Não tem budget para treinar a equipe agora",
      "response": "Apresentar ROI — payback em 3 meses cobre o custo",
      "environment": "Piloto Produção",
      "pilotSize": "2 turnos, linha 3 apenas",
      "action": "Apresentação executiva com análise de ROI",
      "owner": "Sponsor",
      "dueDate": "",
      "status": "Pendente"
    }
  ],
  "implementation": [
    {
      "stakeholderId": "4",
      "stakeholderName": "Nome da pessoa",
      "role": "Operador / Frontline",
      "type": "Impactado",
      "abilityCurrent": "Vermelho",
      "training": "Não realizado",
      "practiced": "Não",
      "difficulty": "",
      "supportAction": "Treinamento hands-on na linha com BB presente",
      "owner": "Black Belt",
      "dueDate": "",
      "status": "Pendente"
    }
  ],
  "tollgateDecision": "Aprovado com restrições",
  "tollgateNotes": ""
}"""
TOOL_SPECIFIC_INSTRUCTIONS["abilityCheck"] = """
ATENCAO - ABILITY CHECK (TOLL-GATE IMPROVE):
Voce e um especialista em Lean Six Sigma + Prosci ADKAR.

CONTEXTO:
Esta ferramenta cobre 3 momentos criticos da fase Improve:
1. O time concorda com as solucoes propostas?
2. Os gestores aprovam o plano de implementacao?
3. Os afetados conseguem executar o novo processo?

FONTE DE DADOS:
- Stakeholders: allProjectData.stakeholderAdkar.stakeholders
- Separar por type: Core Team vs Impactado
- Gestores = Impactados com role "Gestor de Area Impactada"
  ou power "Alto" ou "Medio"

MOMENTO 1 — solutionValidation (Core Team apenas):

abilityCurrent:
- Vermelho: nao entende ou nao consegue executar a solucao
- Amarelo: entende mas tem duvidas ou objecoes
- Verde: entende e apoia a solucao

agreesWithSolution:
- "Sim": concorda com a solucao proposta
- "Parcialmente": concorda em parte, tem ressalvas
- "Não": discorda da solucao

objection (se != "Sim"):
- Descrever objecao especifica baseada no papel e contexto
- Exemplos:
  BB/GB: "Solucao tecnicamente fragil em pico de producao"
  Process Owner: "Impacto nao mapeado no processo X"
  Champion: "Risco reputacional com cliente nao endereçado"

response (se != "Sim"):
- Argumento tecnico ou dado que responde a objecao

action:
- Sim: "Validado — avançar para piloto"
- Parcialmente/Nao: acao concreta para resolver antes do piloto

MOMENTO 2 — planApproval (Impactados com poder):

approvesP lan:
- "Sim": aprova o plano completo
- "Com ressalvas": aprova com condições
- "Não": nao aprova

resourceConcern:
- Descrever o recurso ou custo questionado se houver
- Ex: "Custo de treinamento", "Alocacao de 2 operadores",
  "Parada de linha para piloto"

objection:
- Objecao especifica ao plano

response:
- Argumento ou dado que responde (ROI, benchmarking, etc)

environment:
- "Controlado": lab, area isolada, sem impacto producao
- "Piloto Produção": subconjunto da producao real
- "Produção direta": implementacao completa imediata
- Inferir pelo contexto do projeto — projetos de alto risco
  ou alta complexidade tendem a Piloto Producao

pilotSize (se environment = "Piloto Produção"):
- Descrever escopo minimo do piloto
- Ex: "1 turno, linha 1", "10% do volume", "Area X apenas"
- Principio: menor possivel para validar sem parar producao

MOMENTO 3 — implementation (Impactados diretos):

abilityCurrent:
- Vermelho: nao praticou ainda
- Amarelo: praticou mas comete erros
- Verde: executa sozinho sem apoio

training:
- "Não realizado": ainda nao foi treinado
- "Realizado": passou pelo treinamento
- "Não necessário": mudanca simples, nao requer treinamento

practiced:
- "Não": ainda nao praticou na realidade
- "Sim com dificuldade": praticou mas precisa de apoio
- "Sim sem dificuldade": executa de forma independente

difficulty (se practiced != "Não"):
- Descrever a dificuldade principal encontrada na pratica
- Se "Sim sem dificuldade": deixar vazio

supportAction:
- Acao de suporte especifica para desenvolver a Ability
- Vermelho: treinamento hands-on obrigatorio
- Amarelo: acompanhamento no gemba, checklist de apoio
- Verde: "Monitorar — verificar na proxima semana"

TOLLGATE:
- tollgateDecision:
  Todos concordam + todos aprovam → "Aprovado"
  Alguma ressalva mas sem bloqueio → "Aprovado com restrições"
  Algum "Não" em qualquer momento → "Bloqueado"
- tollgateNotes: 2-3 linhas sobre os principais riscos
  nos 3 momentos e condicoes para avancar para Control
"""

# ════════════════════════════════════════
# FERRAMENTA: REINFORCEMENT CHECK (CONTROL)
# ════════════════════════════════════════
TOOL_STRUCTURES["reinforcementCheck"] = """{
  "sustainability": [
    {
      "stakeholderId": "1",
      "stakeholderName": "Nome da pessoa",
      "role": "Champion",
      "type": "Core Team",
      "reinforcementCurrent": "Amarelo",
      "sustainingFor": "2-4 semanas",
      "regressed": "Não",
      "regressionCause": "",
      "action": "Verificar na reunião semanal de fase",
      "owner": "Black Belt",
      "dueDate": "",
      "status": "Pendente"
    }
  ],
  "postProject": {
    "processOwner": "",
    "controlIndicator": "",
    "monitoringFrequency": "Mensal",
    "reopenCriteria": ""
  },
  "closureDecision": "Encerrado com monitoramento",
  "closureNotes": ""
}"""
TOOL_SPECIFIC_INSTRUCTIONS["reinforcementCheck"] = """
ATENCAO - REINFORCEMENT CHECK (TOLL-GATE CONTROL):
Voce e um especialista em Lean Six Sigma + Prosci ADKAR.

CONTEXTO:
Esta e a ultima ferramenta ADKAR. O projeto esta encerrando.
O Belt esta saindo de cena. A unica pergunta e:
a mudanca vai sobreviver 30 dias sem o Belt?

As pessoas ja aceitaram a mudanca — chegaram ate aqui.
O risco agora e regressao: voltar ao jeito antigo quando
a pressao do projeto some.

FONTE DE DADOS:
- Stakeholders: allProjectData.stakeholderAdkar.stakeholders
- Usar todos os stakeholders

PARA CADA STAKEHOLDER:

reinforcementCurrent:
- Vermelho: voltou ao processo antigo, indicador regrediu
- Amarelo: faz o novo processo as vezes mas regride sob pressao
- Verde: sustenta ha mais de 30 dias sem acompanhamento

sustainingFor:
- "< 2 semanas": implementacao muito recente
- "2-4 semanas": em andamento, ainda nao confirmado
- "+ 30 dias": sustentado — criterio de sucesso
- "Regrediu": voltou ao processo antigo

regressed:
- "Sim": voltou ao processo antigo
- "Não": mantendo o novo processo

regressionCause (se regressed = "Sim"):
- Descrever a causa mais provavel da regressao
- Exemplos:
  "Pressao de producao fez voltar ao atalho antigo"
  "Novo funcionario nao foi treinado no processo novo"
  "Sistema nao foi atualizado para suportar novo fluxo"
- Se regressed = "Não": deixar vazio

action:
- "+ 30 dias" e nao regrediu: "Validado — encerrar acompanhamento"
- "2-4 semanas": "Verificar novamente em 2 semanas"
- "< 2 semanas": "Agendar verificacao em 30 dias"
- Regrediu: acao corretiva especifica e urgente

REGRA DE reinforcementCurrent automatica:
- sustainingFor = "+ 30 dias" e regressed = "Não" → Verde
- sustainingFor = "2-4 semanas" → Amarelo
- sustainingFor = "< 2 semanas" → Amarelo
- regressed = "Sim" → Vermelho

POST PROJECT:
- processOwner: nome do Process Owner do charter se disponivel
- controlIndicator: KPI principal do projeto (do charter.kpi)
- monitoringFrequency: "Mensal" como default
- reopenCriteria: criterio objetivo para reabrir o projeto
  Ex: "Se indicador regredir mais de 20% por 2 meses consecutivos"

CLOSURE DECISION:
- Todos Verde e "+ 30 dias" → "Encerrado"
- Maioria Verde mas algum Amarelo → "Encerrado com monitoramento"
- Qualquer Vermelho ou regrediu → "Reabrir projeto"

closureNotes: 2-3 linhas sobre o status geral de sustentacao
e recomendacoes para o Process Owner apos encerramento.
"""
# ════════════════════════════════════════════════════════════════
# FIM DAS FERRAMENTAS
# ════════════════════════════════════════════════════════════════

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

        structure = TOOL_STRUCTURES.get(req.toolId, "{}")
        specific_instruction = TOOL_SPECIFIC_INSTRUCTIONS.get(req.toolId, "")

        system_prompt = """Voce e um consultor senior Master Black Belt em Lean Six Sigma.
Use os dados ja preenchidos nas ferramentas anteriores para pre-preencher a proxima ferramenta.
REGRAS CRITICAS:
1. Use APENAS informacoes do contexto fornecido - nunca invente dados.
2. Mantenha consistencia absoluta com fases anteriores.
3. Retorne EXCLUSIVAMENTE um objeto JSON valido sem explicacoes e sem markdown.
4. Se um campo nao puder ser inferido, use string vazia.
5. Qualidade de consultoria senior.
6. Responda em portugues do Brasil."""

        user_prompt = f"""
Projeto: "{req.projectInfo.get('name', 'Projeto de Melhoria') if req.projectInfo else 'Projeto de Melhoria'}"

{project_context}

FERRAMENTA A PREENCHER: "{req.toolName}" (ID: {req.toolId})

{specific_instruction}

ESTRUTURA JSON ESPERADA (use exatamente esta estrutura):
{structure}

Retorne EXCLUSIVAMENTE o JSON preenchido com dados reais do projeto.
Sem explicacoes, sem markdown, sem backticks.
"""

        result = await call_claude(system_prompt, user_prompt)
        clean = result.replace("```json", "").replace("```", "").strip()

        if not clean.startswith("{") and "{" in clean:
            clean = clean[clean.index("{"):]
        if not clean.startswith("[") and "[" in clean:
            if "{" not in clean or clean.index("[") < clean.index("{"):
                clean = clean[clean.index("["):]

        parsed = json.loads(clean)
        return {"success": True, "data": parsed}

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON invalido: {str(e)}", "raw": result[:500] if 'result' in locals() else ""}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Rota 2: Gerar relatorio da ferramenta ────────────────────────

@router.post("/report")
async def generate_report(req: ReportRequest):
    try:
        model = CLAUDE_MODEL_BEST if req.toolName == "consolidated" else CLAUDE_MODEL_FAST

        tool_specific = {
            "Brainstorming": "Gere tabela Markdown: | No | Categoria | Ideia | Prioridade |. Top 3 ideias em destaque.",
            "Espinha de Peixe": "Estruture pelos 6Ms. Destaque as 3 causas mais criticas. Tabela de priorizacao.",
            "Plano de Acao 5W2H": "Tabela: | O Que | Por Que | Onde | Quando | Quem | Como | Quanto | Status |",
            "Project Charter": "Documento executivo: Problema, Meta SMART, Escopo, Equipe, Beneficios.",
            "SIPOC": "Tabela SIPOC completa. Analise dos pontos criticos do fluxo.",
            "FMEA": "Tabela FMEA ordenada por RPN decrescente. Acoes prioritarias.",
            "Analise de Stakeholders": "Tabela com nome, poder, interesse, engajamento atual e plano de acao por stakeholder.",
            "Stakeholder & ADKAR": "Tabela com nome, tipo (Core/Impactado), classificacao Poder x Interesse, semaforo ADKAR (A-D-K-A-R) e plano de acao por stakeholder. Destaque o GAP entre Core Team e Impactados.",
        }

        specific = tool_specific.get(req.toolName, f"Relatorio executivo profissional para {req.toolName}.")

        system_prompt = """Voce e um consultor senior Master Black Belt em Lean Six Sigma especializado
em geracao de relatorios executivos profissionais.
REGRAS:
1. Use APENAS os dados fornecidos.
2. Melhore a redacao para padrao executivo de consultoria.
3. Seja conciso - cabe em uma pagina A4.
4. Use tabelas Markdown, negrito, titulos hierarquicos.
5. Termine com ## Proximos Passos Recomendados com 2 a 3 acoes concretas.
6. Idioma: Portugues do Brasil."""

        user_prompt = f"""
PROJETO: {req.projectName}
FERRAMENTA: {req.toolName}
DADOS: {json.dumps(req.toolData, ensure_ascii=False, indent=2)}

INSTRUCAO ESPECIFICA: {specific}

Gere o relatorio executivo agora.
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
            "PreDefinir": "O usuario esta na fase Pre-Definir, identificando e priorizando oportunidades de melhoria.",
            "Define": "O usuario esta na fase Definir, estruturando o escopo, problema e objetivos do projeto.",
            "Measure": "O usuario esta na fase Medir, mapeando o processo e coletando dados da situacao atual.",
            "Analyze": "O usuario esta na fase Analisar, identificando causas raiz do problema com dados.",
            "Improve": "O usuario esta na fase Melhorar, desenvolvendo e implementando solucoes.",
            "Control": "O usuario esta na fase Controlar, sustentando os ganhos e padronizando melhorias.",
        }

        system_prompt = f"""Voce e o Mentor LBW - um consultor senior Master Black Belt em Lean Six Sigma
com 20 anos de experiencia em projetos de melhoria de processos.

PRINCIPIOS:
- Seja direto e tecnico. Evite respostas genericas.
- Use sempre os dados do projeto do usuario para personalizar cada resposta.
- Quando sugerir uma proxima acao, seja especifico.
- Use linguagem de consultoria executiva - profissional mas acessivel.
- Responda em portugues do Brasil.

CONTEXTO ATUAL: {phase_context.get(req.currentPhase, "")}
Ferramenta ativa: {req.currentTool}

DADOS DO PROJETO:
{json.dumps(req.projectData, ensure_ascii=False, indent=2) if req.projectData else "Nenhum dado disponivel ainda."}"""

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

# ─── Rota 4: Sugestoes contextuais do Mentor ──────────────────────

@router.post("/mentor-suggestions")
async def mentor_suggestions(req: dict):
    try:
        current_phase = req.get("currentPhase", "")
        current_tool = req.get("currentTool", "")
        completed_tools = req.get("completedTools", [])
        project_data = req.get("projectData", {})

        system_prompt = """Voce e o Mentor LBW. Gere exatamente 3 sugestoes de perguntas curtas e relevantes
que um profissional faria neste momento do projeto DMAIC.
As sugestoes devem ser:
- Especificas para a fase e ferramenta atual
- Baseadas nos dados ja preenchidos
- Maximo 8 palavras cada
Retorne EXCLUSIVAMENTE um array JSON com 3 strings. Sem explicacoes.
Exemplo: ["Como escrever uma meta SMART?", "Qual o proximo passo?", "Como calcular o impacto?"]"""

        user_prompt = f"""
Fase atual: {current_phase}
Ferramenta atual: {current_tool}
Ferramentas concluidas: {', '.join(completed_tools)}
Contexto: {json.dumps(project_data, ensure_ascii=False)}
Gere as 3 sugestoes agora.
"""

        result = await call_claude(system_prompt, user_prompt, max_tokens=200)
        clean = result.replace("```json", "").replace("```", "").strip()
        suggestions = json.loads(clean)
        return {"success": True, "suggestions": suggestions}

    except Exception:
        fallbacks = {
            "PreDefinir": ["Como priorizar os projetos?", "O que e a Matriz GUT?", "Como validar uma ideia?"],
            "Define": ["Como escrever uma meta SMART?", "O que colocar no escopo?", "Como calcular o impacto?"],
            "Measure": ["Como mapear o processo?", "Quais dados coletar?", "O que e MSA?"],
            "Analyze": ["Como identificar a causa raiz?", "Quando usar o 5 Porques?", "Como usar o Ishikawa?"],
            "Improve": ["Como priorizar as solucoes?", "O que e um piloto?", "Como fazer o FMEA?"],
            "Control": ["Como sustentar os ganhos?", "O que e um POP?", "Como monitorar o KPI?"],
        }
        phase = req.get("currentPhase", "")
        return {"success": True, "suggestions": fallbacks.get(phase, ["Qual o proximo passo?", "Como posso melhorar?", "O que e importante aqui?"])}
