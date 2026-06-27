# agente.py - migrado de OpenAI para Claude (REST) + memória SQLite

import os
import sqlite3
import requests

# Configuração do Claude (via REST, mesmo padrão do claude_routes.py).
# Modelo Sonnet (leve/rápido) — NÃO usar Opus/4.8 aqui por ser potente demais para Q&A.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("AGENTE_CLAUDE_MODEL", "claude-sonnet-4-6")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
}

# ✅ Função para inicializar banco SQLite
def init_db():
    conn = sqlite3.connect("memoria_ia.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT,        -- 'analise' ou 'grafico'
            conteudo TEXT     -- texto ou base64 do grafico
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ✅ Função para salvar conversa
def salvar_conversa(tipo: str, conteudo: str):
    conn = sqlite3.connect("memoria_ia.db")
    c = conn.cursor()
    c.execute('INSERT INTO conversas (tipo, conteudo) VALUES (?, ?)', (tipo, conteudo))
    conn.commit()
    conn.close()

# ✅ Função para recuperar as últimas N conversas
def recuperar_ultimas_conversas(tipo: str, limite: int = 5):
    conn = sqlite3.connect("memoria_ia.db")
    c = conn.cursor()
    c.execute('SELECT conteudo FROM conversas WHERE tipo=? ORDER BY id DESC LIMIT ?', (tipo, limite))
    resultados = c.fetchall()
    conn.close()
    return [r[0] for r in resultados][::-1]  # retorna na ordem cronológica

# ✅ Função principal de pergunta IA
def perguntar_ia(pergunta: str, texto_analise: str) -> str:
    # 🔧 Usa diretamente a última análise recebida como contexto principal
    prompt = (
        "Você é um analista estatístico especialista em análises de dados reais.\n\n"
        "Aqui estão os resultados da última análise realizada pelo sistema:\n\n"
        f"{texto_analise}\n\n"
        "Baseado nestes resultados, responda de forma direta e técnica à pergunta do aluno, "
        "interpretando os dados apresentados sem explicações genéricas de estatística. "
        "Fale como um analista responderia a um engenheiro de processos.\n\n"
        "Se necessrio, recomende o uso de alguma ferramenta grafica ou estatistica adicional par auxiliar o aluno.\n\n"
        f"Pergunta do aluno: {pergunta}"
    )

    # Chama o Claude (REST)
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 500,
        "temperature": 0.2,
        "system": "Você é um analista estatístico especialista em análises de processos.",
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(ANTHROPIC_URL, headers=ANTHROPIC_HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    resposta_final = data["content"][0]["text"]

    # 🔧 Opcional: salvar pergunta e resposta no histórico (ajuste se quiser manter histórico separado)
    salvar_conversa("analise", f"Pergunta: {pergunta}\nResposta: {resposta_final}")

    return resposta_final




